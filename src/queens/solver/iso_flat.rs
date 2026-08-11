//! `IsoFlat` -- [`Fused`](super::Fused)'s selective single graph-iso key (the merge of
//! [`IsoBurr`](super::IsoBurr) at [`Incremental`](super::Incremental)'s nodes/sec) over a
//! **flat lockless [`QueensTt`]** instead of the log-structured [`BurrStore`].
//!
//! `fused` is eviction-free by *freezing* solved entries into immutable BuRR segments, but
//! a miss then walks the whole segment cascade -- so throughput decays from the ~30 M/s
//! memtable regime out of the gate to ~10 M/s once the cascade builds. A flat table never
//! decays: a miss is one O(1) probe forever (the TT probe is ~1% of cycles -- the search is
//! per-node compute-bound, not memory-bound). So iso-flat sustains the fast regime.
//!
//! Same per-node kernel as `fused`/`burr`/`incremental`: the 8 dihedral orientations of the
//! available mask carried down the DFS (`child_orient`, ~62 cyc/move, no re-fold) and a
//! **single** selective key -- the cheap L1-resident tiny graph-iso canon for small
//! fragmented graphs (`popcount <= iso_max_avail`, the transposition-rich deep nodes; the WL
//! canon above the tiny table), else the incremental D4 `lex_min8`, in disjoint tagged
//! namespaces. `iso_max_avail` dials the throughput/merge/fit trilemma: low (≤7, tiny-table
//! only -- the default) avoids all live WL → fast; raising it merges more at WL cost. Sound
//! because the key choice is a pure function of the position (its available popcount) and
//! transpositions are strictly intra-ply.
//!
//! The flat table normally evicts at n=16; the selective merge keeps the resident set
//! smaller than D4, trading some merge for the cheap key. The full-merge (pure-iso, fits-but-
//! WL-bound) end is reachable by raising `iso_max_avail`, but is not the throughput default.

use super::graph::{small_canon_table, tiny_key_from_adj, TINY_TABLE_SLOTS};
use super::incremental::{build_att, child_orient, lex_min8, orient_of};
use super::*;
use crate::queens::dense::MAX_DENSE_K;
use crate::queens::tt::Probe3;
use rayon::prelude::*;
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const ORACLE_FLUSH: u64 = 1 << 14;

#[derive(Default)]
struct OracleAcc {
    attempts: u64,
    hits: u64,
    comp_hits: u64,
    comp_misses: u64,
}

thread_local! {
    /// `QUEENS_SKIP18`: set once per root (in `first_player_wins`'s `resolve`, on the worker that runs
    /// that root) to true iff this root's first-move square is one of the configured slow deep roots.
    /// The whole subtree of a slow root then skips pc==18 TT work; fast roots (and the control) keep it.
    /// Per-worker + steal-off ⇒ one root per worker at a time, so the flag stays correct for the run.
    static IN_SKIP18_ROOT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// `QUEENS_SCHED`: set true on the elder-brother thread iff it is running the slow solo root
    /// (sq 0), so its depth-1 `kids.iter().any` loop records the 2nd-ply move schedule. Every other
    /// root (set false in its own `resolve`) and every deeper node leave it false. Per-worker; sq-0's
    /// depth-1 loop is sequential on the one calling thread, so the flag stays correct.
    static IN_SCHED_ROOT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// The first-move square of the root this worker is currently resolving (set in `resolve`,
    /// like [`IN_SKIP18_ROOT`]). Only read on the cold [`killer_loop`](Fused::killer_loop) paths
    /// for refutation attribution; `u32::MAX` = not inside a root.
    static CUR_ROOT_SQ: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) };
}

/// `QUEENS_KILLER`: global per-square refutation tally for the 2nd-ply killer heuristic. When a
/// root's depth-1 `.any()` finds its refuting reply, that square's count is bumped; later roots
/// prepend the top-`killer_k` already-successful squares present in their own move set to their
/// otherwise-unchanged 2nd-ply order. Cross-root signal: the depth-1 loops run concurrently, so a
/// slow root picks up killers published by roots that finished earlier — attacking the measured
/// critical path (one slow root's sequential 2nd-ply loop ≈ 95% of the n=16 wall; the front-load
/// oracle showed −72% isolated / −13% full-run on such a root). Verdict-preserving: any
/// permutation of an `.any()` over the same reply set proves the same value. Squares are in each
/// root's canonical orientation frame (roots are D4-orbit representatives, so frames are
/// comparable in practice; the A/B is the arbiter). Relaxed ordering — a stale read only costs
/// a missed reordering opportunity, never soundness.
///
/// One table per odd (prove-win) ply of the parallel upper tree: index 0 = depth 1 (the 2nd-ply
/// loop, the record lever), 1 = depth 3, 2 = depth ≥5 (`QUEENS_KILLER_DEEP` extends the jumps to
/// depths 3+; below `min_avail` the deep kernel takes over, so deeper odd plies never get here).
static KILLER_HITS: [[std::sync::atomic::AtomicU32; 256]; 3] =
    [const { [const { std::sync::atomic::AtomicU32::new(0) }; 256] }; 3];

/// `M_DHIST` (`QUEENS_DHIST=1`) deep-kernel cutoff history: per-square tally of every fused-descent
/// cutoff (the square whose child proved LOSS) across the whole run — the killer signal extended
/// below ply 5, applied as a *tiebreak within equal-degree groups* of the dynamic move sort (never
/// a jump, so the degree ordering's forcing-first structure is untouched). Gated by the rank
/// report's measured headroom: ordering_loss = 52.6% of child-exams, r≥3 = 37.1% of nodes.
/// Relaxed atomics; readers tolerate staleness; 16 cache lines spread the write traffic.
static DEEP_HIST: [std::sync::atomic::AtomicU32; 256] =
    [const { std::sync::atomic::AtomicU32::new(0) }; 256];

/// One sq-0 2nd-ply move's in-situ schedule record (`QUEENS_SCHED`). `t_*_us` are µs since the
/// search t0; `nodes` is the cumulative-node delta over the move's subtree (flush-coarse, exact to
/// ~FLUSH_NODES); `child_pc` is the available count after the move (= its number of 3rd-ply child
/// moves); `won` marks the refutation (the `.any()` cut). Cold — one per explored 2nd-ply move.
#[derive(Clone, Copy)]
struct SchedRec {
    sq: u32,
    t_enter_us: u64,
    t_exit_us: u64,
    nodes: u64,
    child_pc: u32,
    won: bool,
}

thread_local! {
    static ORACLE_ACC: RefCell<OracleAcc> = const { RefCell::new(OracleAcc {
        attempts: 0,
        hits: 0,
        comp_hits: 0,
        comp_misses: 0,
    }) };
}

/// Histogram slot count: one per available-popcount, `0..=256` (the n=16 board has
/// `16*16 = 256` squares). Indexed by `avail.popcount()`.
const MAXPC: usize = 257;

/// Production-window measurement mode for [`wins_inc`](IsoFlat::wins_inc), a `const MODE`
/// monomorphisation (resolved once per subtree handoff, never per node). The two measurement
/// modes are mutually exclusive and both only apply to the `!ORACLE && !COUNT && WINDOW`
/// combo, so folding them into one `u8` keeps the generic count down vs two bools.
const M_NORMAL: u8 = 0; // plain solve / flat TT (the A/B control — byte-identical hot path)
const M_HIST: u8 = 1; // tally flat-TT puts by popcount (`QUEENS_PC_HIST=1`)
const M_SEG: u8 = 2; // route by per-popcount band (`QUEENS_TT_SEGMENT=1`)
const M_PROF: u8 = 3; // stratified profiler: rdtsc-time the TT get/put per pc (`QUEENS_PROF=1`)
const M_WAVE: u8 = 4; // ETC + sorted-batch child probe (`QUEENS_WAVE=1`); A'' Phase-1 experiment
const M_SIZE: u8 = 5; // A'' Phase-2a offload sizing: tap the recurse-arm probe stream (`QUEENS_SIZE=1`)
const M_SIZE_WAVE: u8 = 6; // as M_SIZE but ON TOP of the M_WAVE ETC cut — sizes the post-cut residual (`QUEENS_SIZE=2`)
const M_WAVE_B: u8 = 7; // A'' Phase-2b-0 de-risk: descend children in TT-slot order, not move order (`QUEENS_WAVE_B=1`)
const M_L0: u8 = 8; // A'' Phase-2b dedup: M_WAVE + a per-worker L0 probe cache (`QUEENS_L0=1`)
const M_WAVE_C: u8 = 9; // M_WAVE + cascade-reorder (recurse arm first in the pc-cascade) (`QUEENS_WAVE_C=1`)
const M_ORD: u8 = 10; // dynamic move ordering: descend by current available-block degree, not static q.order (`QUEENS_ORD=1`)
const M_ORD_W: u8 = 11; // dynamic move ordering + the M_WAVE ETC cut on top (`QUEENS_ORD=2`)
const M_DECPROBE: u8 = 12; // M_ORD_W + tap the connected-component decomposability of every pc 9..16 getK node (`QUEENS_DECPROBE=1`)
const M_RANK: u8 = 13; // M_ORD_W + tap the first-losing-child cutoff rank (ETC vs descent rank vs no-cut) per pc (`QUEENS_RANK=1`)
const M_COLD: u8 = 14; // M_ORD_W + tap the entry-probe hit/miss (cold-compute) fraction per pc, per-worker (`QUEENS_COLD=1`)
const M_HITKEY: u8 = 15; // M_ORD_W + DUMP each pc≥17 entry probe's canonical key + avail bits (all hits, 1/64 misses) to a file for offline structural study of the 0.2% deep-tail hits (`QUEENS_HITKEY=1`)
const M_DHIST: u8 = 16; // M_ORD_W + deep cutoff-history tiebreak in the dynamic move sort (`QUEENS_DHIST=1`)
const M_KPROBE: u8 = 17; // M_ORD_W + tap every getK entry's labelled code: HLL distinct + memo-sim hit rate (`QUEENS_KPROBE=1`)
const M_RANK_O: u8 = 18; // M_ORD + the M_RANK rank tap: rank capture under dynamic ordering, no ETC (`QUEENS_RANK=1 QUEENS_ORD=1`)
const M_RANK_WV: u8 = 19; // M_WAVE + the M_RANK rank tap: rank capture under static order + ETC (`QUEENS_RANK=1 QUEENS_ORD=0`)
const M_RANK_N: u8 = 20; // M_NORMAL + the M_RANK rank tap: static order, no ETC (`QUEENS_RANK=1 QUEENS_ORD=0 QUEENS_WAVE=0`). NOTE: unlike the byte-identical M_NORMAL control, node_pc is live here, so the production skip18 default applies — set QUEENS_SKIP18=0 for a skip-free capture.

/// The `M_RANK` tap family: the same first-losing-child rank tally captured on each ordering
/// base (the M1 per-variant capture). `MODE` is a const generic ⇒ the predicate folds to a
/// compile-time constant and every tap DCEs off the non-rank instantiations.
const fn mode_rank(mode: u8) -> bool {
    matches!(mode, M_RANK | M_RANK_O | M_RANK_WV | M_RANK_N)
}

/// Max recurse-arm children [`wins_inc`](IsoFlat::wins_inc)'s `M_WAVE` ETC pre-pass batches per
/// node (the sorted-wave window). The deep-tail nodes the lever targets fan out to a handful of
/// recurse children, so 32 covers them; a wider near-root node simply ETC-probes its first 32 and
/// the rest fall through to the normal descent (correctness-neutral — the cut is only ever an
/// *early* return). Stack-bounded (`[u64; 32]×2` per frame) so the recursive `wins_inc` is safe.
const WAVE_CAP: usize = 32;

/// `rdtsc` time-stamp counter read (`constant_tsc`/`nonstop_tsc` on this box, so it tracks
/// wall cycles). Used only on the `M_PROF` measurement path to stratify TT get/put latency
/// by popcount; never on the production hot path.
#[inline(always)]
fn rdtsc() -> u64 {
    // SAFETY: `_rdtsc` is unconditionally available on x86_64 and has no preconditions.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// Per-worker, non-atomic stratified profile accumulator for `QUEENS_PROF` — TT get/put cycle
/// totals and counts by available-popcount. Merged into [`IsoFlat::prof`] at drain (the
/// parallel twin of [`PC_HIST_ACC`]). Zero cost off the `M_PROF` monomorphisation.
struct ProfAcc {
    get_cyc: [u64; MAXPC],
    get_n: [u64; MAXPC],
    put_cyc: [u64; MAXPC],
    nodes: [u64; MAXPC],
}

thread_local! {
    /// Per-worker, **non-atomic** per-popcount flat-TT put tally for the
    /// `QUEENS_PC_HIST` segmented-TT sizing measurement. Each `wins_inc` expansion
    /// (one flat-TT put, always `pc >= 9` in production iso-window) bumps a plain
    /// integer here; merged into the shared [`IsoFlat::pc_hist`] at drain. Empty cost
    /// on the production (`HIST = false`) path — the bump is monomorphised away.
    static PC_HIST_ACC: RefCell<[u64; MAXPC]> = const { RefCell::new([0u64; MAXPC]) };
}

thread_local! {
    /// Per-worker stratified TT-latency profile (`QUEENS_PROF`); merged into the shared
    /// [`IsoFlat::prof`] at drain. Empty cost off the `M_PROF` path.
    static PROF_ACC: RefCell<ProfAcc> = const {
        RefCell::new(ProfAcc {
            get_cyc: [0; MAXPC],
            get_n: [0; MAXPC],
            put_cyc: [0; MAXPC],
            nodes: [0; MAXPC],
        })
    };
}

/// HyperLogLog register width (`2^p`) for the `M_SIZE` global-distinct (dedup-ceiling) estimate.
/// p=16 ⇒ 64 KB shared / 64 KB per-worker thread-local, ≈0.4% std err — ample to size the
/// duplicate fraction of a billions-probe stream.
const SIZE_HLL_P: u32 = 16;
const SIZE_HLL_M: usize = 1 << SIZE_HLL_P;
/// Per-worker cap on the slot-sorted-locality sample (routes). Bounds the memory the cold sort
/// touches; the giant-root tail is one worker, so its sample is a contiguous probe-stream prefix.
const SIZE_SAMPLE_CAP: usize = 4_000_000;

/// `M_KPROBE` band range: getK entries are the descent's pc 9..=`DK` cheap arms; `DK` ≤ 20.
const KPROBE_BANDS: usize = 12; // pc 9..=20, indexed pc-9
/// Per-band HyperLogLog register width for the `M_KPROBE` distinct-code estimate. p=14 ⇒ 16 KB
/// per band (12 bands ≈ 196 KB shared + per worker), ≈0.8% std err — ample for a repeat-rate ratio.
const KPROBE_HLL_P: u32 = 14;
const KPROBE_HLL_M: usize = 1 << KPROBE_HLL_P;
/// `M_KPROBE` simulated code-keyed memo sizes (direct-mapped, 8 B tag/slot, shared across workers
/// like a real memo would be): a small L3-resident-scale table and a large DRAM-scale table. The
/// pair brackets the realizable hit rate between "cache-cheap" and "big but latency-priced".
const KPROBE_SIM_S_BITS: u32 = 20; // 2^20 slots = 8 MiB
const KPROBE_SIM_L_BITS: u32 = 26; // 2^26 slots = 512 MiB

/// Per-worker, non-atomic accumulator for the `QUEENS_SIZE` (A'' Phase-2a) offload-sizing probe:
/// the recurse-arm probe-stream width per available-popcount, a HyperLogLog of the probed
/// canonical keys (the global distinct ⇒ dedup ceiling), and a bounded route sample for the
/// post-sort row-buffer-locality check. Merged into the shared [`IsoFlat`] state at drain (the
/// parallel twin of [`PROF_ACC`]). Zero cost off the `M_SIZE` monomorphisation.
struct SizeAcc {
    w: [u64; MAXPC],       // per-pc probe count (frontier width)
    hll: [u8; SIZE_HLL_M], // thread-local HLL register slice over probed keys (merged by max)
    sample: Vec<u64>,      // contiguous route sample for the slot-sorted locality check
    // Recency-cache simulation (sidecar viability): a per-worker direct-mapped cache of `2^RC_BITS`
    // u32 tags over the entry-probe stream. A "hit" = the same key recurred within the worker's
    // recency window ⇒ a cache-resident sidecar of this size would serve it without a DRAM probe.
    // Bounds the realistic DRAM-cut (≤ the 26.5% global dedup ceiling). Lazily sized on first feed.
    rc: Vec<u32>,
    rc_hits: u64,
    rc_probes: u64,
    rc_per_pc_hits: [u64; MAXPC], // hits attributed to the probed key's band, to find where it pays
    // Temporal bands: hit/probe counts in fixed `RC_WINDOW`-probe windows of this worker's stream,
    // so the report shows whether reuse rises in the late tail (steady-state) vs cold-start.
    rc_win_h: [u64; RC_WINDOWS],
    rc_win_p: [u64; RC_WINDOWS],
}

/// Temporal-band window size (probes) and count for the recency sim. 4M-probe windows × 24 covers
/// the slowest worker's ~58M-probe stream; the last window absorbs the overflow.
const RC_WINDOW: u64 = 4_000_000;
const RC_WINDOWS: usize = 24;

/// Per-worker recency-sim result drained for the report (probes, hits, and the temporal-window
/// curves). The top-probe entries are the 2 slow roots / giant-root tail.
struct RcWorker {
    probes: u64,
    hits: u64,
    win_h: [u64; RC_WINDOWS],
    win_p: [u64; RC_WINDOWS],
}

/// Recency-cache size for the sidecar-viability sim: `2^RC_BITS` u32 tags per worker
/// (default 22 = 4M entries = 16 MB ≈ L3-resident). `QUEENS_RC_BITS` overrides.
fn rc_bits() -> u32 {
    env_u32("QUEENS_RC_BITS", 22).clamp(10, 27)
}

thread_local! {
    /// Per-worker `M_SIZE` accumulator; merged into the shared [`IsoFlat::size_*`] state at drain.
    /// Empty cost off the `M_SIZE` path (this thread-local is only touched on that monomorphisation).
    static SIZE_ACC: RefCell<SizeAcc> = const {
        RefCell::new(SizeAcc {
            w: [0; MAXPC],
            hll: [0; SIZE_HLL_M],
            sample: Vec::new(),
            rc: Vec::new(),
            rc_hits: 0,
            rc_probes: 0,
            rc_per_pc_hits: [0; MAXPC],
            rc_win_h: [0; RC_WINDOWS],
            rc_win_p: [0; RC_WINDOWS],
        })
    };
}

/// `M_DECPROBE` (`QUEENS_DECPROBE=1`) per-pc-band connected-component decomposability tally over the
/// pc 9..16 getK nodes (indexed by `node_pc`). Merged into the shared [`IsoFlat::dec_*`] state at drain
/// (the parallel twin of [`PROF_ACC`]). Zero cost off the `M_DECPROBE` monomorphisation.
struct DecAcc {
    nodes: [u64; MAXPC],      // getK nodes seen per pc
    ncomp_sum: [u64; MAXPC],  // Σ #components (for the mean)
    ge2: [u64; MAXPC],        // #nodes with ≥2 components
    all_le8: [u64; MAXPC],    // #nodes whose every component is ≤8 (table-resolvable)
    all_le_km1: [u64; MAXPC], // #nodes whose every component is ≤(k-1) (drops one getK layer)
    // max-component-size distribution per pc, bucketed: msz_dist[pc][s] = #nodes with max-comp-size==s.
    // s ranges 0..=16 (a pc==16 node's max comp is ≤16); index 0 unused.
    msz_dist: [[u64; 17]; MAXPC],
}

/// `M_KPROBE` (`QUEENS_KPROBE=1`) per-band getK-entry tally (indexed pc−9): entry count, a
/// thread-local HLL register slice per band over the entry's labelled `(pc, code)` key (distinct ⇒
/// the memo-hit ceiling `entries/distinct`), and hit counters for the two shared simulated
/// direct-mapped memo tables. Merged into the shared [`IsoFlat::kprobe_*`] state at drain. Zero
/// cost off the `M_KPROBE` monomorphisation.
struct KprobeAcc {
    entries: [u64; KPROBE_BANDS],
    sim_s_hits: [u64; KPROBE_BANDS],
    sim_l_hits: [u64; KPROBE_BANDS],
    hll: [[u8; KPROBE_HLL_M]; KPROBE_BANDS],
    // `QUEENS_KPROBE=2` only: per-band registers over the CANONICAL (iso-merged `comp_canon`) key —
    // the Tier-C1 go/no-go quantity (canonical distinct = the value-table footprint; entries /
    // canonical distinct = the amortisation multiplicity). Untouched at level 1.
    hll_c: [[u8; KPROBE_HLL_M]; KPROBE_BANDS],
}

/// `M_RANK` (`QUEENS_RANK=1`) per-pc-band first-losing-child cutoff-rank tally (indexed by `node_pc`).
/// Counts where the OR-node's first proven-loss child landed: the ETC pre-pass ("rank -1"), the
/// descent at 0-based rank `r` (`rank_dist`, last bucket absorbs the tail), or no cut at all (a LOSS
/// node, full scan). Merged into the shared [`IsoFlat::rank_*`] at drain. Zero cost off `M_RANK`.
struct RankAcc {
    nodes: [u64; MAXPC],   // expanded OR-nodes seen per pc (the descent-loop ones)
    etc_cut: [u64; MAXPC], // cut by the ETC pre-pass (before the descent)
    // Σ ETC probes issued (= Σ nw over the `nw >= 2` batch) per pc. Tier-A go/no-go: probes-per-cut
    // = etc_probes / etc_cut. A band with many probes but ~0 cuts is wasted ETC work (cold mass) —
    // gate the batch off below that crossover (the descent finds the same cuts later, node-identical).
    etc_probes: [u64; MAXPC],
    no_cut: [u64; MAXPC], // reached the end with no cut (a LOSS node)
    // Sum of degrees (= `moves.len()`, the available-move count) over the no-cut LOSS nodes — a LOSS
    // node examines *every* child, so this is the children-examined total they contribute to `E`
    // (the `degree*nocut` term). Per pc; only the no-cut nodes bump it.
    no_cut_deg: [u64; MAXPC],
    // descent rank distribution: rank_dist[pc][r] = #nodes whose first descent cutoff was at rank r;
    // r in 0..RANK_BUCKETS, the last bucket absorbs rank ≥ RANK_BUCKETS-1.
    rank_dist: [[u64; RANK_BUCKETS]; MAXPC],
}

/// Descent-rank histogram bucket count for [`RankAcc`]; the last bucket is "rank ≥ this−1".
const RANK_BUCKETS: usize = 24;

/// `M_COLD` (`QUEENS_COLD=1`) per-pc-band entry-probe hit/miss tally (indexed by `node_pc`). A node's
/// entry get into the flat TT either HITs (a transposition/re-probe — warm work) or MISSes (the node
/// expands — cold compute). The miss% per pc is the cold-compute fraction the memory-side prefetch/
/// pre-warm levers target; a substantially-warm tail (low miss%) ⇒ nothing to pre-warm. Kept PER
/// WORKER (non-atomic) and drained as a [`ColdWorker`] so the report can isolate the giant-root tail
/// (the top-probe worker) from the aggregate. Zero cost off the `M_COLD` monomorphisation.
struct ColdAcc {
    hits: [u64; MAXPC],
    misses: [u64; MAXPC],
}

/// Per-worker `M_COLD` result drained for the report: this worker's per-pc entry-probe hit/miss
/// arrays plus its total probe count (the top-probe worker is the giant-root tail). The arrays are
/// boxed so a `ColdWorker` is cheap to move into the shared `Vec` (drained off the hot path).
struct ColdWorker {
    probes: u64,
    hits: Box<[u64; MAXPC]>,
    misses: Box<[u64; MAXPC]>,
}

/// `M_HITKEY` (`QUEENS_HITKEY=1`) one captured pc≥17 entry probe: the node's canonical key and its
/// own available-set bits, plus the probe outcome. From `avail` (a board-square bitset) + the board
/// side `n` the OFFLINE study reconstructs the exact conflict graph and every cheating-free structural
/// feature; `key` is the canonical identity (D4-merged) so recurrences of the same node collapse and
/// PV-overlap is an exact-key check. The point: compare the feature distribution of the rare HITs (the
/// 0.2% deep-tail transpositions — high-value re-probes) against a miss sample to find what predicts a
/// recurrence WITHOUT consulting the verdict / future probe stream (the prefetch/pin/order lever).
#[derive(Clone, Copy)]
struct HitRec {
    key: Bits,
    avail: Bits,
    pc: u16,
    hit: bool,
}

/// Per-worker `M_HITKEY` accumulator: the captured records and a miss counter for the 1/64 sampling
/// (hits are rare ⇒ kept in full; misses are the bulk ⇒ sampled). Drained into [`IsoFlat::hitkey_recs`]
/// and written to a file post-solve. Empty cost off the `M_HITKEY` monomorphisation.
struct HitKeyAcc {
    recs: Vec<HitRec>,
    miss_seen: u64,
}

/// 1-in-N miss sampling for `M_HITKEY` (hits captured in full; misses are ~500× more numerous).
const HITKEY_MISS_SAMPLE: u64 = 64;

thread_local! {
    /// Per-worker `M_DECPROBE` accumulator; merged into the shared [`IsoFlat::dec_*`] at drain.
    /// Empty cost off the `M_DECPROBE` path.
    static DEC_ACC: RefCell<DecAcc> = const {
        RefCell::new(DecAcc {
            nodes: [0; MAXPC],
            ncomp_sum: [0; MAXPC],
            ge2: [0; MAXPC],
            all_le8: [0; MAXPC],
            all_le_km1: [0; MAXPC],
            msz_dist: [[0; 17]; MAXPC],
        })
    };
    /// Per-worker `M_KPROBE` accumulator; merged into the shared [`IsoFlat::kprobe_*`] at drain.
    /// Empty cost off the `M_KPROBE` path.
    static KPROBE_ACC: RefCell<KprobeAcc> = const {
        RefCell::new(KprobeAcc {
            entries: [0; KPROBE_BANDS],
            sim_s_hits: [0; KPROBE_BANDS],
            sim_l_hits: [0; KPROBE_BANDS],
            hll: [[0; KPROBE_HLL_M]; KPROBE_BANDS],
            hll_c: [[0; KPROBE_HLL_M]; KPROBE_BANDS],
        })
    };
    /// Per-worker `M_RANK` accumulator; merged into the shared [`IsoFlat::rank_*`] at drain.
    /// Empty cost off the `M_RANK` path.
    static RANK_ACC: RefCell<RankAcc> = const {
        RefCell::new(RankAcc {
            nodes: [0; MAXPC],
            etc_cut: [0; MAXPC],
            etc_probes: [0; MAXPC],
            no_cut: [0; MAXPC],
            no_cut_deg: [0; MAXPC],
            rank_dist: [[0; RANK_BUCKETS]; MAXPC],
        })
    };
    /// Per-worker `M_COLD` accumulator; drained into the shared per-worker [`IsoFlat::cold_workers`]
    /// list at drain. Empty cost off the `M_COLD` path.
    static COLD_ACC: RefCell<ColdAcc> = const {
        RefCell::new(ColdAcc {
            hits: [0; MAXPC],
            misses: [0; MAXPC],
        })
    };
    /// Per-worker `M_HITKEY` accumulator; drained into [`IsoFlat::hitkey_recs`] at solve end.
    /// Empty cost off the `M_HITKEY` path.
    static HITKEY_ACC: RefCell<HitKeyAcc> = const {
        RefCell::new(HitKeyAcc {
            recs: Vec::new(),
            miss_seen: 0,
        })
    };
}

/// A'' Phase-2b dedup (`M_L0`): per-worker **L0 probe cache** — a small direct-mapped cache of solved
/// `(route, fp) → val` in front of the multi-GB flat TT. The order-INDEPENDENT half of the sorted-wave
/// prize (no move-ordering tax): a recurring key (a transposition, or the `M_WAVE` ETC-then-descent
/// re-probe) is served from this ~1 MB L2/L3-resident table (~10 cyc) instead of a cold flat-TT DRAM
/// probe (~176 cyc). Sound because a queens position's win/loss is **immutable** (key-determined), so a
/// cached entry is never stale; the 55-bit `fp` tag matches the TT's own collision tolerance (no new
/// false hits). Indexed by the LOW bits of `route` (the TT's `fastrange` uses the HIGH bits ⇒ the two
/// indexings are decorrelated). Touched only on the `M_L0` monomorphisation (gated `QUEENS_L0=1`).
const L0_BITS: u32 = 17;
const L0_SIZE: usize = 1 << L0_BITS;
const L0_FP_MASK: u64 = (1u64 << 55) - 1; // mirrors `Slot::fp_mask()` (fp = bits [9..64))

thread_local! {
    /// Entry encoding (one `u64`): `0` = empty; else `1 | (val << 1) | ((fp & L0_FP_MASK) << 2)`
    /// (bit 0 = used, bit 1 = val, bits [2..57) = fp tag). Zero-cost off the `M_L0` path.
    static L0_CACHE: RefCell<[u64; L0_SIZE]> = const { RefCell::new([0u64; L0_SIZE]) };
}

/// L0 probe-cache lookup (see [`L0_CACHE`]). `Some(val)` on a tag-matched hit, else `None`.
#[inline]
fn l0_get(route: u64, fp: u64) -> Option<u8> {
    let idx = (route as usize) & (L0_SIZE - 1);
    L0_CACHE.with(|c| {
        let e = c.borrow()[idx];
        (e & 1 != 0 && (e >> 2) == (fp & L0_FP_MASK)).then_some(((e >> 1) & 1) as u8)
    })
}

/// Insert a solved `(route, fp) → val` into the L0 probe cache (direct-mapped, evict-on-collision).
#[inline]
fn l0_put(route: u64, fp: u64, val: u8) {
    let idx = (route as usize) & (L0_SIZE - 1);
    let e = 1 | ((val as u64 & 1) << 1) | ((fp & L0_FP_MASK) << 2);
    L0_CACHE.with(|c| c.borrow_mut()[idx] = e);
}

// ---- Raw-pointer L0 sidecar (QUEENS_SIDECAR) — the handoff's untried M_L0 angle -------------
// Same direct-mapped 1 MB (L2-resident) cache, but accessed via a **raw pointer fetched once per
// node** (no per-probe `.with`/`RefCell` borrow — the overhead the M_L0 build measured at +6%
// cyc/node). The node's entry-get + its puts reuse the one pointer. Exact 55-bit fp tag ⇒ a hit is
// the correct fixed value ⇒ node-count-neutral. Tests whether the M_L0 negative was the access
// overhead or whether the repeats are genuinely TT-warm (a sidecar hit then saving no DRAM).
thread_local! {
    static RAW_L0: std::cell::UnsafeCell<Vec<u64>> = const { std::cell::UnsafeCell::new(Vec::new()) };
}

/// Fetch this worker's raw L0 base pointer (one TLS lookup per node), lazily sizing on first use.
#[inline]
fn raw_l0_ptr() -> *mut u64 {
    RAW_L0.with(|c| {
        // SAFETY: single-threaded per-worker access; the Vec lives for the thread and is never
        // resized after the first node, so the pointer stays valid for the node's duration.
        let v = unsafe { &mut *c.get() };
        if v.len() != L0_SIZE {
            *v = vec![0u64; L0_SIZE];
        }
        v.as_mut_ptr()
    })
}

/// Raw L0 lookup on a pre-fetched base pointer (no TLS, no borrow). `Some(val)` on a tag hit.
/// # Safety: `base` is a valid `L0_SIZE`-element buffer from [`raw_l0_ptr`].
#[inline]
unsafe fn raw_l0_get(base: *mut u64, route: u64, fp: u64) -> Option<u8> {
    let e = *base.add((route as usize) & (L0_SIZE - 1));
    (e & 1 != 0 && (e >> 2) == (fp & L0_FP_MASK)).then_some(((e >> 1) & 1) as u8)
}

/// Raw L0 store on a pre-fetched base pointer (no TLS, no borrow).
/// # Safety: `base` is a valid `L0_SIZE`-element buffer from [`raw_l0_ptr`].
#[inline]
unsafe fn raw_l0_put(base: *mut u64, route: u64, fp: u64, val: u8) {
    *base.add((route as usize) & (L0_SIZE - 1)) =
        1 | ((val as u64 & 1) << 1) | ((fp & L0_FP_MASK) << 2);
}

/// One suspended ancestor on the explicit upper-tree search stack of
/// [`wins_inc_iter`](IsoFlat::wins_inc_iter). Plain `Copy` POD so the thread-local arena reuses
/// storage with no per-node allocation. `orient[0]` is the node's available mask (kept whole so
/// the recurse arm can derive child orientations); `moves[moves_start..moves_start+nmoves]` in the
/// shared arena is its filtered child list (exactly what the recursion passes children as
/// `pmoves`), resumed at index `mi`. No `result` field is needed: a winning child just resumes the
/// parent's loop, a losing child wins the parent outright (handled by the unwind cascade).
// hot-struct discipline (CLAUDE.md #4/#6): explicit layout, fields largest-align-descending (the
// align-8 `Bits`/`u64` block, then the `u32`s, then the lone `u8` last — so `repr(C)` introduces no
// interior padding and the frame stays the same 328 B the default repr packed it to).
#[derive(Clone, Copy)]
#[repr(C)]
struct IncFrame {
    orient: [Bits; 8],
    key: Bits,
    route: u64,
    fp: u64,
    moves_start: u32,
    nmoves: u32,
    mi: u32,
    /// Search depth of this node (handoff depth + stack height). **Even depth ⇒ a prove-loss node**
    /// (every child must be searched ⇒ publishing its children for work-stealing is zero-speculation,
    /// exactly as `par_wins_inc` fans its even plies). Also gives a stolen child its own root depth.
    /// Dead unless `const STEAL`.
    depth: u32,
    /// Work-stealing: how many of this frame's children have been published as `rayon` scope tasks
    /// (capped at the idle-core count). Dead unless `const STEAL`.
    published: u32,
    /// ABDADA two-pass deferral state (one of [`PASS0`]/[`PASS0_DEF`]/[`PASS1`]). In pass 0 an
    /// in-flight child is *skipped* (left in place, flips the frame to [`PASS0_DEF`]) so this
    /// worker keeps useful work going while the child's owner finishes it. When the move list is
    /// exhausted with deferrals outstanding ([`PASS0_DEF`]) the frame flips to [`PASS1`] and
    /// re-scans from the start: children whose owners finished are now cheap TT hits, and any
    /// still in-flight are expanded by this worker (the progress guarantee). Always [`PASS0`] for
    /// the non-ABDADA (`const ABDADA == false`) instantiation — dead state the compiler elides.
    /// Last field (`u8`): keeps the `repr(C)` layout padding-free.
    pass: u8,
}

// #7: a field-add that grows the per-node arena frame fails the build instead of silently
// widening the stride. 256 (`[Bits;8]`) + 32 (`key`) + 16 (`route`/`fp`) + 20 (5×`u32`) + 1 = 325 → 328.
const _: () =
    assert!(std::mem::size_of::<IncFrame>() == 328 && std::mem::align_of::<IncFrame>() == 8);

/// [`IncFrame::pass`] states for the ABDADA two-pass deferral (see that field).
const PASS0: u8 = 0;
const PASS0_DEF: u8 = 1;
const PASS1: u8 = 2;

/// Work-stealing: how many expanded nodes between samples of the (atomic) `steal_armed` flag in the
/// hot loop. Large enough that the per-node cost is a single local increment + compare (the atomic
/// load is amortised ~1/this), small enough that a long tail handoff arms within a few thousand nodes
/// of the delay expiring.
const STEAL_CHECK_EVERY: u32 = 4096;

/// Per-worker reusable arena for [`wins_inc_iter`](IsoFlat::wins_inc_iter): the frame stack plus a
/// shared move buffer (each frame owns the slice `moves[moves_start..]`, reclaimed by `truncate` on
/// pop). Cleared, not freed, at each subtree handoff — zero per-node allocation.
struct IncArena {
    frames: Vec<IncFrame>,
    moves: Vec<u8>,
}

thread_local! {
    static INC_STACK: RefCell<IncArena> =
        const { RefCell::new(IncArena { frames: Vec::new(), moves: Vec::new() }) };
}

/// Compact representation of an in-band (`popcount ≤ 7`) available graph: once a node
/// drops into the iso band the whole subtree below it is a pure ≤7-vertex graph game
/// (Node Kayles), so it is built **once** at band entry and the search carries it down
/// instead of touching the 256-bit board again. Vertices are labelled `0..k0` in
/// q.order (the move order [`wins_tiny`](IsoFlat::wins_tiny) used), so the searched node
/// set stays byte-identical.
///
/// - `closed[i]` = the local vertices removed by playing `i` (its neighbours **and**
///   itself — `attack[v]` is self-blocking): the child's alive set is `alive & !closed[i]`.
/// - `adj[i]` = `closed[i]` minus the self bit: the edges, for the relabelling-invariant
///   tiny-canon edge code ([`tiny_key_from_adj`]).
///
/// Plain `[u8; 8]` data, one cache line, passed by `&` — the per-node board ops
/// (`and_not`/`popcount`/`each` over four `u64`s) and the per-child attack-row loads of
/// the old tail collapse to single-byte ops on `alive`.
#[repr(C)] // hot-struct discipline (CLAUDE.md #4/#5): explicit layout, two-per-line `[u8;8]` pair.
struct TinyGraph {
    adj: [u8; MAXV_TINY],
    closed: [u8; MAXV_TINY],
}

// #7: lock the carried in-band graph at 2×8 bytes (rule #5's {8,16,32} no-per-record-align tier).
const _: () = assert!(
    std::mem::size_of::<TinyGraph>() == 2 * MAXV_TINY && std::mem::align_of::<TinyGraph>() == 1
);

/// Local-vertex capacity for [`TinyGraph`] — the tiny-canon band tops out at 7, padded
/// to 8 for a clean stride (matches `SMALL_WORK_MAX` in `graph.rs`).
const MAXV_TINY: usize = 8;

#[inline(always)]
fn avail_has8(avail: Bits, sq: u8) -> bool {
    let sq = sq as u32;
    let word = (sq >> 6) as usize;
    let bit = sq & 63;
    // SAFETY: every caller feeds byte-compressed copies of `q.order`.
    unsafe { (*avail.0.get_unchecked(word) & (1u64 << bit)) != 0 }
}

#[inline(always)]
fn att_for(att: &[[Bits; 8]], sq: u32) -> &[Bits; 8] {
    debug_assert!((sq as usize) < att.len());
    // SAFETY: `sq` is drawn from `q.order` or its filtered subsequences, and `att` has one
    // entry per board square.
    unsafe { att.get_unchecked(sq as usize) }
}

#[inline(always)]
fn att_for8(att: &[[Bits; 8]], sq: u8) -> &[Bits; 8] {
    // SAFETY: `sq` is a byte-compressed board square from `q.order`.
    unsafe { att.get_unchecked(sq as usize) }
}

#[inline(always)]
fn att08(att: &[[Bits; 8]], sq: u8) -> Bits {
    att_for8(att, sq)[0]
}

/// Byte indices `0..=255`: word `w`'s 64-byte slice `[w*64 .. w*64+64)` is the identity
/// source vector for the [`verts_of`] `vpcompressb` extraction. 64-aligned for the load.
#[repr(align(64))]
struct IdentBytes([u8; 256]);
static IDENT_BYTES: IdentBytes = {
    let mut a = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        a[i] = i as u8;
        i += 1;
    }
    IdentBytes(a)
};

/// Scatter the set-square indices of `avail` (ascending) into `verts[..pc]`. A closure-free
/// twin of `avail.each(|v| ...)`: a plain `#[inline(always)]` fn with no `FnMut` reliably
/// inlines into the giant `wins_inc`, where the closure form was being *outlined* to a shared
/// `FnMut::call_mut` (~6.9% of n=16 search cycles — the closure is invoked once per live square,
/// per `wK_get` entry on the pc 9..16 dense majority). Output is bit-identical to `each`.
///
/// AVX512-VBMI2 `vpcompressb` per word (compress the identity byte vector by the word's bit
/// mask) replaces the serial `tzcnt`/`x &= x-1` scan — that loop's ~2-cycle-per-set-bit
/// dependent chain measured ~9% of n=16 search cycles (srcline profile, 2026-07-02).
/// `vpcompressb` preserves ascending bit order, so the output stays byte-identical.
#[inline(always)]
fn verts_of(avail: Bits, verts: &mut [u8]) {
    use std::arch::x86_64::{
        _mm512_load_si512, _mm512_mask_storeu_epi8, _mm512_maskz_compress_epi8,
    };
    let mut n = 0usize;
    for (w, &word) in avail.0.iter().enumerate() {
        // SAFETY: znver5 ⇒ AVX512-BW/VBMI2. The masked store writes exactly `popcount(word)`
        // bytes at `verts[n..]` (masked-out lanes are architecturally suppressed — no write,
        // no fault), and `Σ popcount = verts.len()` at every call site (a pc==K node), so all
        // writes are in bounds. `IDENT_BYTES` is 64-aligned for the aligned load.
        unsafe {
            let idx = _mm512_load_si512(IDENT_BYTES.0.as_ptr().add(w * 64) as *const _);
            let c = _mm512_maskz_compress_epi8(word, idx);
            let cnt = word.count_ones();
            let smask = ((1u128 << cnt) - 1) as u64;
            _mm512_mask_storeu_epi8(verts.as_mut_ptr().add(n) as *mut i8, smask, c);
            n += cnt as usize;
        }
    }
    debug_assert_eq!(n, verts.len());
}

/// Compact a board-square attack `row` against the `K` live squares of an `avail` set (whose
/// per-word values are `a`) into a `K`-bit labelled adjacency row, with one 4-word BMI2 `pext`.
/// Bit `j` of the result = "`row` hits the `j`-th live square of `avail`" — exactly the labelled
/// adjacency the `wK_get` code-build needs. `cpre = [c0,c1,c2]` are `avail`'s cumulative word
/// popcounts (`c0 = popcount(a[0])`, `c1 = c0+popcount(a[1])`, `c2 = c1+popcount(a[2])`), constant
/// across all `K` rows so the caller hoists them once. This replaces the `K·(K-1)/2` scalar
/// `Bits::get` bit-tests of the code-build — which the compiler auto-vectorizes at K≤9 but falls to
/// scalar `bt`-per-bit for K≥10 (the profile's largest compute bucket) — with `K` rows of 4 fast
/// pexts (znver5 `pext` = 3-cyc latency / 1-per-cyc), ~½ the ops on the pc 10..13 builders. The
/// live squares scatter across ≤4 of the board's 64-bit words, so each row needs all 4 words
/// extracted and stitched at `avail`'s word boundaries (the prefix popcounts).
#[inline(always)]
fn adj_row_pext(row: Bits, a: &[u64; 4], cpre: [u32; 3]) -> u64 {
    use std::arch::x86_64::_pext_u64;
    // SAFETY: production is built with target-cpu=znver5, which includes BMI2.
    unsafe {
        let r = row.0;
        _pext_u64(r[0], a[0])
            | (_pext_u64(r[1], a[1]) << cpre[0])
            | (_pext_u64(r[2], a[2]) << cpre[1])
            | (_pext_u64(r[3], a[3]) << cpre[2])
    }
}

/// `M_KPROBE`: rebuild the labelled edge code of one getK-entry node (`avail`, pc==`k`) exactly as
/// the `wN_get`/`w_wide_get` builders pack it (upper-triangular, ascending-square label order) for
/// any runtime `k` in 9..=20, into words 0..=2; word 3 carries `k` so the folded key is band-tagged
/// and bands never collide. This IS the key a code-keyed getK memo would use — two board positions
/// with the same induced labelled graph produce the same key. Cold probe path only (`QUEENS_KPROBE`
/// run); the tap DCEs off `M_KPROBE`, production never calls this.
fn kprobe_code(att: &[[Bits; 8]], avail: Bits, k: u32) -> Bits {
    debug_assert!((9..=20).contains(&k));
    debug_assert_eq!(avail.popcount(), k);
    let mut verts = [0u8; 20];
    verts_of(avail, &mut verts[..k as usize]);
    let a = &avail.0;
    let c0 = a[0].count_ones();
    let c1 = c0 + a[1].count_ones();
    let c2 = c1 + a[2].count_ones();
    let cpre = [c0, c1, c2];
    let mut words = [0u64; 4];
    let mut off = 0u32;
    for i in 0..k {
        let packed = adj_row_pext(att08(att, verts[i as usize]), a, cpre);
        let width = k - 1 - i;
        let contrib = (packed >> (i + 1)) & ((1u64 << width) - 1);
        let lo = off & 63;
        let wi = (off >> 6) as usize;
        words[wi] |= contrib << lo;
        // k=20 tops out at 190 code bits, so a straddle never reaches word 3 (the band tag).
        if lo + width > 64 {
            words[wi + 1] |= contrib >> (64 - lo);
        }
        off += width;
    }
    words[3] = k as u64;
    Bits(words)
}

/// `M_DECPROBE` (`QUEENS_DECPROBE=1`) connected-component decomposition of one pc==`k` getK node's
/// conflict graph, built straight from `avail` (the `k` live squares) + `att` — the exact labelled
/// adjacency the `wK_get` code-build forms (`adj_row_pext` per vertex, kept per-row instead of packed
/// into a code). Returns `(ncomp, max_size, all_le8, all_le_km1)`:
/// - `ncomp`     = number of connected components (isolated squares are size-1 components),
/// - `max_size`  = largest component's vertex count,
/// - `all_le8`   = every component is ≤8 vertices (⇒ table-resolvable, zero getK recursion),
/// - `all_le_km1`= every component is ≤`k-1` (⇒ drops at least one getK layer).
///
/// Components via bitmask BFS over the ≤16 `u16` adjacency rows: lowest unvisited vertex seeds a
/// frontier `|= adj[v]` to fixpoint = one component; repeat. Cold measurement only (DCEs for every
/// other MODE).
#[inline]
fn decompose_node(avail: Bits, att: &[[Bits; 8]], k: usize) -> (u32, u32, bool, bool) {
    let mut verts = [0u8; 16];
    verts_of(avail, &mut verts[..k]);
    let a = &avail.0;
    let c0 = a[0].count_ones();
    let c1 = c0 + a[1].count_ones();
    let c2 = c1 + a[2].count_ones();
    let cpre = [c0, c1, c2];
    // Labelled adjacency rows: bit j of adj[i] set iff live square i attacks live square j.
    let mut adj = [0u16; 16];
    for i in 0..k {
        let packed = adj_row_pext(att08(att, verts[i]), a, cpre) as u16;
        // Clear the self-bit (a square doesn't conflict with itself) — `adj_row_pext` includes it.
        adj[i] = packed & !(1u16 << i);
    }
    let full: u16 = if k >= 16 { u16::MAX } else { (1u16 << k) - 1 };
    let mut unvisited = full;
    let mut ncomp = 0u32;
    let mut max_size = 0u32;
    let mut all_le8 = true;
    let mut all_le_km1 = true;
    while unvisited != 0 {
        let seed = unvisited.trailing_zeros();
        let mut comp = 1u16 << seed;
        loop {
            let mut next = comp;
            let mut r = comp;
            while r != 0 {
                let v = r.trailing_zeros() as usize;
                r &= r - 1;
                next |= adj[v];
            }
            next &= full;
            if next == comp {
                break;
            }
            comp = next;
        }
        unvisited &= !comp;
        let sz = comp.count_ones();
        ncomp += 1;
        max_size = max_size.max(sz);
        all_le8 &= sz <= 8;
        all_le_km1 &= (sz as usize) < k; // every component ≤ k-1 ⇒ drops at least one getK layer
    }
    (ncomp, max_size, all_le8, all_le_km1)
}

/// Thousands-separated decimal (cold report formatting only; the `bin` has its own twin).
fn commas(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.char_indices() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// The `k`-vertex (`k ≤ 8`) labelled upper-triangular edge code of the `alive` sub-graph of a
/// dense block, in the exact bit order [`dense::DenseW8`] is built with (pairs `(x,y)`, `x<y`,
/// ascending). `closed[v]` carries `v`'s neighbours (self-blocking), and `verts[x] ≠ verts[y]`,
/// so `(closed[verts[x]] >> verts[y]) & 1` is the edge bit. Lets the block resolve any ≤8
/// descendant by one complete-table lookup instead of recomputing it.
#[inline]
fn dense_block_code(closed: &[u16; 13], alive: u16, k: usize) -> usize {
    let mut verts = [0usize; 8];
    let mut n = 0usize;
    let mut rem = alive;
    while rem != 0 {
        verts[n] = rem.trailing_zeros() as usize;
        rem &= rem - 1;
        n += 1;
    }
    let mut code = 0usize;
    let mut bit = 0usize;
    for x in 0..k {
        for &vy in verts.iter().take(k).skip(x + 1) {
            code |= (((closed[verts[x]] >> vy) & 1) as usize) << bit;
            bit += 1;
        }
    }
    code
}

/// Filter `pmoves` (the parent node's available squares, already in `q.order`) down to the
/// squares still set in `avail`, written compactly into `buf` and returned as a slice. This
/// replaces the per-node scan over all `n²` squares with a scan over the parent's
/// (monotonically shrinking) move list. It preserves the `q.order` subsequence, so the move
/// order — and therefore the searched node set — is byte-identical. `buf` is left uninit (no
/// `n²`-wide zero-init, which would cost more than the scan it removes).
#[inline]
fn filter_moves<'a>(buf: &'a mut [MaybeUninit<u8>; MAXV], pmoves: &[u8], avail: Bits) -> &'a [u8] {
    let mut nc = 0usize;
    for &sq in pmoves {
        // Branchless compaction: write `sq` unconditionally, then advance the count only if
        // it survives the filter — an unavailable `sq` is simply overwritten next iteration.
        // `avail_has8` is ~50/50 down the tree, so the old `if`-guarded write was a coin-flip
        // branch the predictor missed every other node; this trades it for one always-taken
        // L1 store (much cheaper than the misprediction). Output is byte-identical.
        // SAFETY: `pmoves` is a `q.order` subsequence (≤ MAXV entries) and `nc` never exceeds
        // the count of survivors so far, so `buf[nc]` is always in bounds.
        unsafe { buf.get_unchecked_mut(nc).write(sq) };
        nc += avail_has8(avail, sq) as usize;
    }
    // SAFETY: the loop initialised exactly `buf[..nc]` via `write`; `MaybeUninit<u8>` is
    // layout-identical to `u8` and `u8` has no invalid bit patterns, so reading that
    // prefix back as `&[u8]` (bounded by the returned `'a` borrow of `buf`) is sound.
    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, nc) }
}

/// Resolve a `u32` env knob once at construction (never per node). Used for the
/// leaf-oracle prototype thresholds.
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Print the per-root wall schedule captured by `QUEENS_ROOT_TIMING` (each root's [start, end]
/// interval in seconds since the solve began), sorted by completion. The summary answers the
/// tail question directly: **SOLO for last X s** = how long the final root ran with the others
/// finished (the single-threaded tail), and **longest = Y% of total wall** = whether one root
/// dominates (size-ordering won't help; deep parallelism would balloon) or several are
/// comparable (size-ordering shortens the tail for free).
fn print_root_timing(n: u32, moves: &[u32], starts: &[AtomicU64], ends: &[AtomicU64]) {
    let secs = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
    // (idx, square, start_s, end_s)
    let mut rows: Vec<(usize, u32, f64, f64)> = (0..starts.len())
        .map(|i| (i, moves[i], secs(&starts[i]), secs(&ends[i])))
        .collect();
    if rows.is_empty() {
        return;
    }
    let total = rows.iter().map(|r| r.3).fold(0.0_f64, f64::max);
    rows.sort_by(|a, b| a.3.total_cmp(&b.3));
    println!(
        "\n[root-timing] n={n}: {} roots, total wall {total:.1}s (sorted by end)",
        rows.len()
    );
    println!(
        "  {:>3} {:>4} {:>9} {:>9} {:>9}",
        "idx", "sq", "start", "end", "dur"
    );
    for &(i, sq, s, e) in &rows {
        println!("  {i:>3} {sq:>4} {s:>8.1}s {e:>8.1}s {:>8.1}s", e - s);
    }
    let last = rows[rows.len() - 1];
    // The final root ran alone from the next-latest end until its own end.
    let solo = if rows.len() >= 2 {
        last.3 - rows[rows.len() - 2].3
    } else {
        total
    };
    let mut durs: Vec<f64> = rows.iter().map(|r| r.3 - r.2).collect();
    durs.sort_by(|a, b| b.total_cmp(a));
    let top: Vec<String> = durs.iter().take(3).map(|d| format!("{d:.1}s")).collect();
    println!(
        "  tail root idx {} (sq {}): ran {:.1}s, ended {:.1}s — SOLO for last {solo:.1}s ({:.0}% of wall)",
        last.0,
        last.1,
        last.3 - last.2,
        last.3,
        100.0 * solo / total,
    );
    println!(
        "  longest 3 root durations: [{}] — longest = {:.0}% of total wall",
        top.join(", "),
        100.0 * durs[0] / total,
    );
}

/// Tag a complete component iso key into a namespace disjoint from the position keys
/// ([`graph_bits`]/[`d4_bits`]) so the flat TT can memoise per-component **nimbers**
/// (stored in the `val` byte, `< 16`) alongside the win/loss position entries without
/// aliasing (collisions only at the table's ~2⁻⁵⁵ fingerprint rate, like any TT entry).
#[inline]
fn comp_nimber_bits(h: u64) -> Bits {
    Bits([
        h,
        mix64(h ^ 0x4E49_4D42_4552_0001),
        0x4E49_4D42_4552_4E49,
        0x4E49_4D42_0000_0000,
    ])
}

/// **IsoFlat** -- the A3 DFS-resident kernel + flat lockless TT + single selective iso/D4 key.
pub struct IsoFlat {
    name: &'static str,
    tt: QueensTt,
    att: OnceLock<Box<[[Bits; 8]]>>,
    order8: OnceLock<Box<[u8]>>,
    /// `order_rank[sq]` = `sq`'s position in `q.order` (descending attack degree). Built
    /// once; lets the iso-band entry relabel a child's vertices into q.order with a tiny
    /// insertion sort instead of rescanning the parent's (long) move list.
    order_rank: OnceLock<Box<[u8]>>,
    /// Complete, eviction-free ≤7 win/loss table, keyed by the **labelled** dense index
    /// [`Queens::tiny_table_index`] (`OFF[k] + edge_code`) — `0` = unknown, `1` = loss,
    /// `2` = win. One byte per labelled code (~2 MB direct, no fingerprint, no collision), so
    /// a band entry is a single direct indexed load with no canon-table lookup and no flat-TT
    /// DRAM probe. Keying by the labelled (not canonical) code skips the 16 MB canon table —
    /// the win/loss is iso-invariant, so every labelling stores the same value; the slight
    /// merge loss is recomputed cheaply in the L1 [`solve_local`](Self::solve_local) memo.
    /// (A smaller L2 *hash* table was tried — it collides above ~512 K reached codes, and the
    /// collision recomputes inflate the node count and *slow* completion, so the collision-
    /// free direct table wins.) Shared lock-free: a position's value is fixed.
    tiny_tt: Box<[AtomicU8]>,
    /// Complete dense W8 labelled win/loss table (32 MiB, opt-in via the `iso-window`
    /// solver). At `popcount == 8` the whole subgame is an 8-vertex Node-Kayles graph whose
    /// value is iso-invariant, so it is looked up by the raw 28-bit labelled edge code — one
    /// TLB-friendly indexed bit load, instead of a `child_orient`/`lex_min8` D4 key + a
    /// scattered probe into the 13–17 GB flat TT (and, on a miss, expanding the whole pc==8
    /// subtree). `None` for plain `iso-flat`.
    dense8: Option<DenseW8>,
    /// Dense low-popcount **ceiling K** for the `iso-dense` solver: every pc==k node with
    /// `9 ≤ k ≤ dense_k` is resolved directly from the complete W0..W8 tables by the W_K
    /// evaluator (`W_K(G) = ∃v · ¬W_{K-1}(G∖N[v])`, one BMI2-projected child sweep, no flat-TT
    /// probe and no subtree expansion), exactly as W8 already does pc==8. `8` = off (the
    /// `iso-flat`/`iso-window` control). Set by `new_dense` (default 16 — the u128 code ceiling;
    /// `QUEENS_DENSE_K` override, clamped 9..=16; raising K pays the whole way up with the pext
    /// code-build, −30.6% n=16 wall K=12→16). Read
    /// **once at the root** to pick the `const DK` search
    /// instantiation — never a run-constant branch in the deep loop or a TT probe.
    dense_k: u32,
    tiny_canon: &'static [u64],
    par_depth: u32,
    par_min_avail: Option<u32>,
    /// `QUEENS_WARM_RESTART=1`: a two-phase solve. **Phase 1** runs the full root fan for
    /// `warm_secs` to warm the shared TT, then aborts (via a deadline-triggered unwind that skips
    /// every in-flight node's `par_tt_put`, so only *completed*, correct subtrees persist — never a
    /// wrong value). **Phase 2** re-runs the real search over the now-warm TT: the roots that
    /// finished in phase 1 are instant TT hits, and the slow (unfinished) roots restart **staggered**
    /// (`warm_stagger_ms` apart) so one warms the shared region before the next hits it, instead of
    /// racing. Phase 2 is otherwise the byte-identical search — no per-node contention added. Off ⇒
    /// no phase 1 and the single split-path bool check short-circuits (the default stays unchanged).
    warm_restart: bool,
    warm_secs: u64,
    warm_stagger_ms: u64,
    /// True only while phase 1 is running (gates the deadline check in `par_wins_inc`).
    warm_phase: AtomicBool,
    /// Set by the phase-1 watchdog thread at the `warm_secs` deadline; a split node that sees it set
    /// (while `warm_phase`) panics to unwind the warm pass. `Arc` so the watchdog can hold a clone.
    warm_deadline: Arc<AtomicBool>,
    /// `QUEENS_ROOT_TIMING=1`: record each root's wall [start, end] interval (µs since the
    /// solve began) and print the schedule post-solve. Diagnoses whether the single-threaded
    /// tail is **one dominant root** (one long interval running solo at the end → deep
    /// parallelism needed, balloons) or **several comparable roots** (→ size-ordering shortens
    /// the tail for free). Cold: two timestamps per root (≤ root_total times), zero hot-path cost.
    root_timing: bool,
    iso_max_avail: u32,
    /// `QUEENS_BLOCK_K` (default 8 = off): dense-block boundary. When `> 8`, a node dropping to
    /// `8 < pc ≤ block_k` is solved as one **local block** — the flat-TT boundary entry is merged
    /// once by its D4 key, then the whole subtree below is solved in a thread-private L1 memo over
    /// the `u16` alive mask (no per-descendant flat-TT probe), exactly as the pc≤7 band already
    /// does. Measurement prototype for the dense-blocks lever; resolved once here, never per node.
    block_k: u32,
    eff_min_avail: AtomicU32,
    root_done: AtomicU64,
    root_total: AtomicU64,
    // Lever-B leaf-oracle prototype (QUEENS_NIMBER_ORACLE=1): when a node's available
    // graph fully decomposes into connected components each ≤ `nimber_k`, resolve it by
    // decompose → per-component nimber (memoised) → XOR → win iff ≠0, with NO recursion.
    // Because max-component is monotone non-increasing down the tree, this prunes the
    // whole all-small region below its frontier (G1: ~42% of distinct nodes at ≤7, n=14).
    nimber_oracle: bool,
    counting: bool,
    /// `QUEENS_PC_HIST=1`: tally flat-TT puts by available-popcount into [`pc_hist`](Self::pc_hist)
    /// for segmented-TT band sizing (resolved once here, never per node — the hot loop is
    /// monomorphised on `const MODE = M_HIST`, selected from this at the per-subtree handoff).
    hist: bool,
    /// `QUEENS_TT_SEGMENT=1` (mirrors [`QueensTt::is_segmented`]): route flat-TT probes by
    /// per-popcount band. Resolved once; selects `const MODE = M_SEG` at the subtree handoff,
    /// so the deep hot path is fully monomorphised (the flat control stays byte-identical).
    segment: bool,
    /// `QUEENS_TT_ASSOC=1` (mirrors [`QueensTt::is_assoc`], implies [`segment`](Self::segment)):
    /// route each band probe into a cache-line set-associative bucket (8-way). Orthogonal to the
    /// search-strategy `MODE`; resolved once at construction, read by the TT-helper layout branch.
    assoc: bool,
    /// `QUEENS_SIDECAR=1`: the raw-pointer once-per-node L0 sidecar (the handoff's untried M_L0
    /// angle). The node's entry probe checks a per-worker 1 MB direct-mapped exact cache (raw ptr
    /// fetched once, no per-probe `.with`) before the TT; populated on the node's put. Off ⇒ the
    /// base-pointer fetch and the branch short-circuit.
    sidecar: bool,
    /// `QUEENS_PFDEEP=1`: gather-time recurse-child prefetch (the cheap-first PREFETCH lever). The
    /// fused descent already prefetches each recurse child ~30 cyc before recursing into it (one-ahead),
    /// hiding only a sliver of the ~165-cyc cold DRAM entry probe. Recurse children (`pc > recurse_min`)
    /// sort LAST in the degree-ordered move list, so when this is set we issue every recurse child's
    /// `prefetch_h` at GATHER time — the descent then scans the cheap getK/band children (real cycles,
    /// no TT probe) before reaching the recurse arm, overlapping that scan with the cold load. The win
    /// is at `nw == 1` (the deep-tail majority: pc≥17 nodes have too few recurse children for the ETC
    /// batch, which only prefetches at `nw >= 2`). Byte-identical node set (pure cache hint). Off ⇒ the
    /// current behaviour exactly (prefetch only inside the `nw >= 2` ETC batch).
    pf_deep: bool,
    /// `QUEENS_ETC_PC=<pc>`: gate the `nw >= 2` ETC probe batch OFF for nodes with `node_pc` below this
    /// threshold (default 0 = never gated = current behaviour, byte-identical). The gather + the
    /// gather-time prefetch are KEPT (the descent reuses the descriptors and the prefetch still warms
    /// each recurse child's entry probe) — only the eager ETC pre-pass probe loop is skipped. Tier-A
    /// lever: the `M_RANK` tap shows the ETC cuts ~0–5% of nodes in pc≤28 (pr/cut 300–5000 = the probe
    /// is near-pure waste there) and only starts paying at pc≥29 (ETC% 12–35%, pr/cut <180). Gating the
    /// cold mass off drops the redundant ETC probe while the descent still finds the same cuts lazily
    /// (warm, via the kept prefetch).
    ///
    /// **MEASURED-NEGATIVE (n=16 4-round interleaved A/B, 8 GB TT, `QUEENS_ETC_GATE=1` ⇒ gate pc<29):
    /// cyc/node −1.3% (the per-node probe saving IS real), but nodes +2.0% / total cyc +0.7% / wall
    /// +3.6% — a net LOSS.** The "node-set-identical" premise was WRONG: the ETC's value in the cold
    /// pc≤28 mass is NOT its ~0% cuts — it's the **win-child reuse** (`wv==1` skip below = eviction
    /// protection: a recurse child the ETC proved a WIN is skipped instead of re-recursed, and on a
    /// direct-mapped TT that child's slot is often evicted by the time the descent reaches it ⇒ skipping
    /// it avoids a full re-expansion). That reuse and the "wasted" cut-probe are the SAME probe, so no
    /// pc-gate can keep the protection while dropping the waste. Kept gated-off as substrate (default
    /// 0 ⇒ `node_pc >= 0` always true ⇒ byte-identical). The one untried angle: at a much larger TT
    /// (≥17 GB, less eviction ⇒ smaller +nodes) the −1.3% cyc/node might net out — but the box is
    /// memory-tight for back-to-back 17 GB A/Bs, and the ceiling is small regardless.
    etc_pc_gate: u32,
    /// Shared per-popcount flat-TT put histogram (one [`AtomicU64`] per popcount), merged
    /// from each worker's thread-local [`PC_HIST_ACC`] at drain. Only populated when `hist`.
    pc_hist: Box<[AtomicU64]>,
    /// `QUEENS_PROF=1`: stratified TT-latency profiler. Selects `const MODE = M_PROF` at the
    /// subtree handoff; `wins_inc` then `rdtsc`-times each flat-TT get/put and bins the cycles
    /// by popcount. Production (`M_NORMAL`) compiles all of it away.
    prof: bool,
    /// Shared profile accumulator (`4 * MAXPC` [`AtomicU64`]: get-cyc / get-n / put-cyc / nodes,
    /// laid out `metric * MAXPC + pc`), merged from each worker's [`PROF_ACC`] at drain.
    prof_data: Box<[AtomicU64]>,
    /// `QUEENS_WAVE=1` (A'' Phase-1): selects `const MODE = M_WAVE` at the subtree handoff, adding
    /// an **ETC (enhanced transposition cutoff) + sorted-batch** pre-pass to every `wins_inc` node —
    /// probe all recurse-arm children's TT (sorted by slot, all prefetches up front) *before*
    /// expanding any; a known-loss/empty child wins the OR node outright (cut). Verdict-preserving;
    /// changes only which children expand (gate-safe on iso-dense, no `--distinct`). Off = byte-
    /// identical `M_NORMAL`.
    wave: bool,
    /// `QUEENS_SIZE=1` (A'' Phase-2a): selects `const MODE = M_SIZE` at the subtree handoff and taps
    /// the recurse-arm probe stream — every `wins_inc` entry is one flat-TT get — to size the idle-core
    /// offload (Approach B). Cold measurement only: per-pc width into [`size_w`](Self::size_w), the
    /// probed keys into [`size_hll`](Self::size_hll) (global distinct ⇒ the dedup ceiling), and a route
    /// sample into [`size_sample`](Self::size_sample) (post-sort row-buffer locality). Off = byte-
    /// identical `M_NORMAL`; the tap DCEs to nothing on every other `MODE`.
    size: bool,
    /// `QUEENS_SIZE=2`: run the sizing tap ON TOP of the M_WAVE ETC cut (`M_SIZE_WAVE`), so the
    /// measured stream is the post-cut **residual** Approach B offloads on top of the default — vs
    /// `QUEENS_SIZE=1` (`M_SIZE`), the WAVE-off pre-cut upper bound. Implies `size`.
    size_wave: bool,
    /// Shared per-popcount recurse-arm probe count (frontier width), merged from each worker's
    /// [`SIZE_ACC`] at drain. Only populated when `size`.
    size_w: Box<[AtomicU64]>,
    /// Shared HyperLogLog over the probed canonical keys (the global distinct ⇒ dedup ceiling),
    /// merged register-wise from each worker's thread-local slice at drain. Only used when `size`.
    size_hll: Hll,
    /// Cold slot-sorted-locality sample (routes), drained from each worker's [`SIZE_ACC`]. Only
    /// populated when `size`; touched off the hot path (drain + post-solve report).
    size_sample: Mutex<Vec<u64>>,
    /// Recency-cache sidecar-viability sim: cache size `2^size_rc_bits` (`QUEENS_RC_BITS`), resolved
    /// once. Per-worker `(probes, hits)` collected at drain (the top-probe workers are the 2 slow
    /// roots — the giant-root tail that dominates the wall, where the reuse must be measured).
    size_rc_bits: u32,
    size_rc: Mutex<Vec<RcWorker>>,
    /// Aggregate recency-hit count per available-popcount band (where a sidecar would pay).
    size_rc_pc: Box<[AtomicU64]>,
    /// `QUEENS_WAVE_B=1` (A'' Phase-2b-0 de-risk): selects `const MODE = M_WAVE_B` at the subtree
    /// handoff — descend each node's children in **TT-slot order** (the single-thread sorted-frontier
    /// wave) instead of move order. Verdict-preserving (reorder never changes the OR/AND value); the
    /// node-count delta vs the WAVE-off move-order baseline is the move-ordering tax the sorted wave
    /// pays for its row-buffer locality (the 2b gate). Off = byte-identical (the sort DCEs).
    wave_b: bool,
    /// `QUEENS_L0=1` (A'' Phase-2b dedup): selects `const MODE = M_L0` = M_WAVE + the per-worker L0
    /// probe cache ([`L0_CACHE`]) layered into `mtt_get`/`mtt_put`. The order-independent dedup (no
    /// move-ordering tax): recurring keys served from L2/L3 instead of a cold flat-TT DRAM probe. Off =
    /// byte-identical M_WAVE (`mtt_get`/`mtt_put` are the identity for every non-`M_L0`/`M_SEG` mode).
    l0: bool,
    /// `QUEENS_WAVE_C=1`: selects `const MODE = M_WAVE_C` = M_WAVE with the recurse arm hoisted to the
    /// front of the fused-descent pc-cascade (a deep-tail recurse child skips the ~8 cheap-arm tests).
    /// Behaviour- and node-count-identical to M_WAVE (only branch order shifts) — a frontend micro-opt;
    /// off = byte-identical M_WAVE (the front arm DCEs).
    wave_c: bool,
    /// `QUEENS_ORD=1`: selects `const MODE = M_ORD` — dynamic move ordering by current available-block
    /// degree (`child0.popcount()` ascending) instead of the static `q.order`. Verdict-preserving; the
    /// node-count delta vs static is the ordering gain (the move-ordering lever the +94% slot-order tax
    /// indicted). Off = byte-identical (the per-node re-sort DCEs). `QUEENS_ORD=2` also runs the M_WAVE
    /// ETC cut on top (`ord_etc` ⇒ `M_ORD_W`).
    ord: bool,
    /// `QUEENS_ORD=2`: dynamic ordering **plus** the M_WAVE ETC cut (`M_ORD_W`) — does ETC still pay on
    /// top of the better ordering? Implies `ord`.
    ord_etc: bool,
    /// `QUEENS_SKIP18=1`: skip ALL transposition-table work for pc==18 nodes — don't compute the canon
    /// key (`lex_min8`→`d4_bits`→`hash128`, the ~6%-of-cycles / #1-branch-mispredict step), don't probe,
    /// don't put, don't ETC. **Safe & cascade-free because pc==18 is the shallowest recurse level: every
    /// pc==18 child is pc≤17 = a getK leaf (no further recursion)**, so a re-expanded pc==18 node only
    /// re-runs a bounded getK sweep — never an unmemoised subtree (the B2 canon-skip cascade can't happen
    /// here). The entry-probe hit rate at pc==18 is ~0.3% (HITKEY-measured), so the skipped key+probe+put
    /// is near-pure waste. A/B toggle (`=0` = byte-identical control). Verdict-preserving (the TT is only
    /// a memo); node-count rises by ~the hit rate's re-expansions (bounded getK sweeps).
    skip18: bool,
    /// `QUEENS_SKIP18_ROOTS=<sq>,<sq>`: the first-move square indices of the slow deep roots that
    /// [`skip18`](Self::skip18) applies to (their entire run). Empty ⇒ all roots. Set per root via
    /// [`IN_SKIP18_ROOT`] in `first_player_wins`.
    skip18_squares: Vec<u8>,
    /// `QUEENS_SKIP18_PCS=18,20,22` (default `{18}`): the SET of pc bands to skip all TT work for, as a
    /// bitmask (bit `pc`). A *set* (vs a contiguous range) keeps the in-between bands memoized, so a
    /// re-expanded skipped node hits its non-skipped recurse children — interrupting the cascade that a
    /// contiguous `[18..hi]` range triggers (measured +17% nodes at hi=22). `{18}` alone is cascade-free
    /// (children all getK leaves). The sweep over sets finds the most key+probe+put saved per re-exp.
    skip18_pcs: u64,
    /// `QUEENS_SKIP18_FRAC=M` (default 1 = off) + `QUEENS_SKIP18_FRAC_PCS=<set>`: for the *cascading*
    /// bands (pc 19–25), skip TT for only a `1/M` FRACTION of nodes (chosen by a pre-key hash of the raw
    /// `avail`, since the canon key is exactly what we skip), keeping the other `(M-1)/M` memoised as
    /// cascade / re-probe **anchors** — the dampened way to capture some of a cascading band's
    /// key+probe+put saving without the full re-expansion. Orientation-specific (the TT is canon-keyed
    /// ⇒ a canonical node is kept iff a kept orientation reached it); `M` tunes saving vs re-exp.
    skip18_frac: u32,
    skip18_frac_pcs: u64,
    /// `QUEENS_DECPROBE=1`: selects `const MODE = M_DECPROBE` (= M_ORD_W + a cold tap on every pc 9..16
    /// getK node's connected-component decomposition — gates the nimber-decomposition lever). Per-pc
    /// component stats merged from each worker's [`DEC_ACC`] at drain; report printed post-solve. Off =
    /// byte-identical M_ORD_W (the tap DCEs).
    decprobe: bool,
    /// `QUEENS_DHIST=1`: selects `const MODE = M_DHIST` (M_ORD_W + the deep cutoff-history
    /// tiebreak in the dynamic move sort — see [`DEEP_HIST`]). Verdict-preserving (an OR-node
    /// child permutation); node set differs from the default ⇒ A/B on nodes/total-cyc/wall.
    dhist: bool,
    /// Shared `M_DECPROBE` accumulator (6 metric arrays + the max-comp-size dist), `AtomicU64`-backed
    /// so workers fold their [`DEC_ACC`] in lock-free. Only populated when `decprobe`.
    dec_nodes: Box<[AtomicU64]>,
    dec_ncomp_sum: Box<[AtomicU64]>,
    dec_ge2: Box<[AtomicU64]>,
    dec_all_le8: Box<[AtomicU64]>,
    dec_all_le_km1: Box<[AtomicU64]>,
    dec_msz: Box<[AtomicU64]>, // laid out [pc * 17 + size]
    /// `QUEENS_KPROBE=1`: selects `const MODE = M_KPROBE` (= M_ORD_W + a cold tap on every getK
    /// entry: rebuild the labelled `(pc, code)` key, fold it into a per-band HLL (distinct ⇒ the
    /// repeat-rate ceiling `entries/distinct`) and probe two shared simulated direct-mapped memo
    /// tables (finite-size hit rates) — gates the code-keyed getK memo lever). Off = byte-identical
    /// M_ORD_W (the tap DCEs).
    kprobe: bool,
    /// `QUEENS_KPROBE=2`: also fold each getK entry's CANONICAL key (`each_comp_canon`, the
    /// iso-merged WL/IR certificate, components combined order-independently) into a second
    /// per-band HLL — the Tier-C1 canonical-value-table go/no-go (footprint + multiplicity).
    /// Much slower per entry (a WL canon per getK call); level 1 skips it.
    kprobe_canon: bool,
    /// Shared per-band `M_KPROBE` HLLs (indexed pc−9); workers fold locals in at drain.
    kprobe_hll: Vec<Hll>,
    /// Shared per-band canonical-key HLLs (level 2 only).
    kprobe_hll_c: Vec<Hll>,
    /// Shared `M_KPROBE` per-band tallies (indexed pc−9), `AtomicU64`-backed.
    kprobe_entries: Box<[AtomicU64]>,
    kprobe_s_hits: Box<[AtomicU64]>,
    kprobe_l_hits: Box<[AtomicU64]>,
    /// `M_KPROBE` simulated memo tag tables (`fp|1` per slot, 0 = empty; direct-mapped, always-
    /// replace). Shared across workers like a real memo would be; empty `Box<[]>` when off.
    kprobe_sim_s: Box<[AtomicU64]>,
    kprobe_sim_l: Box<[AtomicU64]>,
    /// `QUEENS_RANK=1`: selects `const MODE = M_RANK` (= M_ORD_W + a cold tap on the first-losing-child
    /// cutoff rank — gates the move-ordering lever). Per-pc ETC/descent-rank/no-cut stats merged from
    /// each worker's [`RANK_ACC`] at drain; report printed post-solve. Off = byte-identical M_ORD_W.
    rank: bool,
    /// Shared `M_RANK` accumulator, `AtomicU64`-backed. Only populated when `rank`.
    rank_nodes: Box<[AtomicU64]>,
    rank_etc: Box<[AtomicU64]>,
    rank_etc_probes: Box<[AtomicU64]>, // Σ ETC probes issued per pc (Tier-A probes-per-cut tap)
    rank_nocut: Box<[AtomicU64]>,
    rank_nocut_deg: Box<[AtomicU64]>, // Σ degree over no-cut LOSS nodes per pc (the E `degree*nocut` term)
    rank_dist: Box<[AtomicU64]>,      // laid out [pc * RANK_BUCKETS + r]
    /// `QUEENS_COLD=1`: selects `const MODE = M_COLD` (= M_ORD_W + a cold tap on the entry-probe
    /// hit/miss per pc — gates the memory-side prefetch/pre-warm lever family). Per-worker hit/miss
    /// arrays merged from each worker's [`COLD_ACC`] at drain; report printed post-solve, isolating
    /// the giant-root tail (top-probe worker) from the aggregate. Off = byte-identical M_ORD_W.
    cold: bool,
    /// Per-worker drained `M_COLD` results (one [`ColdWorker`] per worker that probed). Sorted by
    /// probe count in the report so worker rank 0 = the giant-root tail. Only populated when `cold`.
    cold_workers: Mutex<Vec<ColdWorker>>,
    /// `QUEENS_HITKEY=1`: selects `const MODE = M_HITKEY` (= M_ORD_W + capture each pc≥17 entry probe's
    /// canonical key + avail bits to a file for offline study of the 0.2% deep-tail hits). Off =
    /// byte-identical M_ORD_W (the capture DCEs).
    hitkey: bool,
    /// Output path for the `M_HITKEY` binary dump (`QUEENS_HITKEY_OUT`, default `/tmp/queens-hitkeys.bin`).
    hitkey_out: String,
    /// Shared `M_HITKEY` record collector, drained from each worker's [`HITKEY_ACC`] at solve end and
    /// written to [`hitkey_out`](Self::hitkey_out). Only populated when `hitkey`.
    hitkey_recs: Mutex<Vec<HitRec>>,
    nimber_k: u32,
    nimber_pc: u32,
    tiny8_direct: bool,
    /// `QUEENS_UNROLL=1`: route the production ≤7 band solve through the recursion-unwound
    /// iterative [`solve_local_iter`](Self::solve_local_iter) instead of the recursive
    /// [`solve_local`](Self::solve_local). Resolved once here; a single branch per band entry
    /// (never in the inner ≤7 loop). Off = byte-identical control.
    unroll: bool,
    /// `QUEENS_ITER=1`: route the production deep upper-tree solve through the fully
    /// recursion-unwound [`wins_inc_iter`](Self::wins_inc_iter) (an explicit frame stack)
    /// instead of the recursive [`wins_inc`](Self::wins_inc). Every miss *pushes* a frame
    /// instead of recursing; node completion is a `pop`; the unwind cascades in a loop — no
    /// call frames in the deep upper tree at all. Resolved once at the subtree handoff (a
    /// per-handoff branch, never per node). Off = byte-identical control (`wins_inc` untouched).
    iter_inc: bool,
    /// `QUEENS_ABDADA=1`: run the deep upper-tree solve through the explicit-stack
    /// [`wins_inc_iter`](Self::wins_inc_iter) with **ABDADA in-flight markers** — a worker that
    /// probes a child another worker is currently expanding *defers* it (tries siblings first,
    /// revisits once before falling back to expanding it itself) instead of re-expanding it
    /// concurrently. Implies the iterative path. Resolved once at the subtree handoff into a
    /// `const ABDADA` monomorphisation (never a per-node branch); off = byte-identical control.
    abdada: bool,
    /// `QUEENS_STEAL=1`: **frontier work-stealing** on the explicit-stack deep solve (implies the
    /// ABDADA machinery). At an *even-depth* (prove-loss, zero-speculation — every child is
    /// searched anyway) frame, when idle cores exist ([`deep_busy`](Self::deep_busy) `< n_threads`),
    /// the worker *publishes* its pending children as `rayon` scope tasks (marking each in-flight so
    /// it then defers them via ABDADA) instead of expanding them itself. An idle worker steals each
    /// published child, searches it, and writes the verdict to the shared TT — so the busy giant-root
    /// worker's deferral now resolves to a *hit* (a separate core did the work) rather than the
    /// fallback re-expansion that made plain ABDADA a wash. Targets the 51%-util giant-root tail.
    /// Resolved once at the handoff into a `const STEAL` monomorphisation; off = byte-identical.
    steal: bool,
    /// `QUEENS_STEAL_DELAY` seconds (default 60): work-stealing stays **off** until this much wall has
    /// elapsed, then a watchdog flips [`steal_armed`](Self::steal_armed). The early all-roots phase is
    /// already ~fully parallel, so publishing there only adds re-expansion + scheduling overhead (the
    /// measured 2.4× loss). By the delay the cheap roots have finished, so only the ~2 dominant roots
    /// (the 51%-util tail) are still running ⇒ stealing targets exactly them.
    steal_delay: u64,
    /// Set true by the steal watchdog after [`steal_delay`](Self::steal_delay) seconds. The hot loop
    /// samples it only every `STEAL_CHECK_EVERY` nodes (a cheap local counter, never a per-node atomic
    /// load), so an early-phase handoff that finishes quickly never even reads it. `Arc` so the
    /// watchdog thread holds a clone.
    steal_armed: Arc<AtomicBool>,
    /// Worker count of the `rayon` pool, read once (`rayon::current_num_threads`). The
    /// work-stealing publish gate compares [`deep_busy`](Self::deep_busy) against it.
    n_threads: usize,
    /// `QUEENS_STEAL_WIDTH` (default 2): max children a single frame publishes. The idle-core gate
    /// alone bursts up to ~`n_threads` wide at one frame and each stealer recurses the same ⇒ an
    /// exponential publish fan-out that re-does shared transpositions (the measured +15% nodes). A
    /// small per-frame cap spreads the helpers *deep* instead (deeper subtrees are more disjoint ⇒
    /// less re-expansion) while still saturating idle cores across plies.
    steal_width: u32,
    /// `QUEENS_STEAL_MIN_PC` (default 18): only publish a child whose available-popcount is at least
    /// this — i.e. a *substantial* subtree worth the spawn/scope/defer overhead. A small subtree is
    /// cheaper to expand locally than to ship to another core (the overhead dominates and stealing
    /// nets a loss). "Split the known-expensive nodes, not the tiny ones."
    steal_min_pc: u32,
    /// Number of workers currently inside a deep `wins_inc_iter` solve (incremented on entry,
    /// decremented on exit). `n_threads - deep_busy` is the idle-core proxy that gates publishing
    /// (publish only when cores are idle, so a stealer is there to take the work promptly — else
    /// over-publishing would queue work the busy worker re-expands as the PASS1 fallback).
    deep_busy: AtomicUsize,
    /// `QUEENS_STEAL_MAX` (default = `n_threads`): hard cap on **concurrent in-flight splits**. The
    /// idle-core proxy alone lets fast small splits churn (the measured 12M total spawns); an explicit
    /// concurrency cap means at most this many stolen subtrees are alive at once — paired with a high
    /// `min_pc` (big, long-running splits) it throttles the total spawn count to ~`steal_max`, replaced
    /// as each finishes, instead of an unbounded fan-out.
    steal_max: u32,
    /// Currently in-flight stolen subtrees (incremented at publish, decremented when the stealer's
    /// `wins_inc_iter` returns). Gated against [`steal_max`](Self::steal_max).
    active_splits: AtomicU64,
    /// Work-stealing diagnostics (cold; only touched on the rare gated publish / PASS1-fallback path).
    /// `steal_published` = subtrees split off to idle cores; `steal_pc_hist[pc]` = their available-
    /// popcount distribution (what we're splitting); `steal_fallback` = published-then-still-in-flight
    /// children the busy worker had to expand itself in PASS1 (a steal that failed to land ⇒ a
    /// re-expansion). Printed post-solve when stealing fired.
    steal_published: AtomicU64,
    steal_fallback: AtomicU64,
    steal_pc_hist: Box<[AtomicU64]>,
    oracle_attempts: AtomicU64,
    oracle_hits: AtomicU64,
    oracle_comp_hits: AtomicU64,
    oracle_comp_misses: AtomicU64,
    /// `QUEENS_SCHED=1`: capture the slow solo root (sq 0)'s 2nd-ply move *schedule* — per move, its
    /// in-situ [enter,exit] wall (µs), cumulative-node delta, child-move count, and whether it refuted
    /// (the `.any()` cut). sq-0's depth-1 loop is sequential on the elder-brother thread, so per-move
    /// attribution is exact (no sibling overlap). Cold; off ⇒ `IN_SCHED_ROOT` stays false ⇒ dead branch.
    sched: bool,
    sched_t0: std::sync::Mutex<Instant>,
    sched_recs: std::sync::Mutex<Vec<SchedRec>>,
    /// `QUEENS_PAR_ORD=1`: extend dynamic child-degree ordering to the parallel **upper tree**
    /// (`par_wins_inc`), which today uses the *static* `order8`. Orders each split node's children by
    /// current child-degree ascending (most-forcing first) so an odd (prove-win) node's `.any()` cutoff
    /// finds a refutation sooner — the 2nd-ply lever. Verdict-correct (reorder ⊥ value); changes node
    /// count at odd nodes by design (not part of the `--distinct` gate). Resolved once.
    par_ord: bool,
    /// `QUEENS_SPLIT=1`: speculatively parallelize the depth-1 (2nd-ply) `.any()` — normally sequential
    /// (odd/prove-win). The "split that root" lever; viable because 2nd-ply moves are near-independent.
    split: bool,
    /// `QUEENS_KILLER=<k>`: allow up to `k` cross-root killer-reply jumps (squares that already
    /// refuted another root, from [`KILLER_HITS`]) in each root's depth-1 2nd-ply `.any()` order;
    /// the rest keep the existing order. Base default 0 (iso-flat/iso-window control); the
    /// iso-dense constructor promotes the measured default 4. See [`KILLER_HITS`].
    killer_k: u32,
    /// `QUEENS_KILLER_DEEP=1` (default off): extend the killer jumps to the deeper odd (prove-win)
    /// plies of the parallel upper tree (depth 3, 5) with one shared table per ply band.
    killer_deep: bool,
}

impl IsoFlat {
    pub fn new(bits: u32) -> Self {
        Self::from_tt(QueensTt::new(bits))
    }

    pub fn new_window(bits: u32) -> Self {
        Self::from_tt_with_window(QueensTt::new(bits), true)
    }

    /// The `iso-dense` solver: `iso-window`'s kernel plus the dense low-popcount layer forced
    /// **on** by construction — every pc==9 node is resolved directly from the complete W0..W8
    /// tables (one BMI2-projected child sweep, no flat-TT probe and no subtree expansion),
    /// exactly as W8 already does pc==8. This is the next step in the iso-flat → iso-window →
    /// iso-dense lineage; `iso-window` stays the dense-off control. The dense ceiling K
    /// defaults to **16** — the `u128` labelled-code ceiling (16·15/2 = 120 bits). Once the
    /// pext-per-row code-build made the deep getK builders cheap, the W_K crossover moved all the
    /// way to the ceiling: each layer keeps cutting ~14–22% more nodes and the n=16 wall drops
    /// monotonically (deterministic n=14 nodes K=12 7.9M → K=16 4.0M = −50%; 16 GB single-run wall
    /// K=12 49.5s → K=14 42.1s → K=15 39.1s → K=16 34.4s = −30.6%). The node cut is **inherent**
    /// (TT-independent: 16 GB nodes ≈ 12 GB nodes), so it holds at production TT. W9..W11 keep the
    /// labelled code in a `u64`; W12..W16 (66..120-bit) run on the `u128` two-word `pext` path.
    /// `QUEENS_DENSE_K` (9..=16) overrides (lower it to trade getK cost for recurse expansion);
    /// `getK` resolves every `9 ≤ pc ≤ K` directly from the complete W0..W8 tables.
    pub fn new_dense(bits: u32) -> Self {
        // Overlap the CPU-bound dense-table build with the kernel-bound TT alloc + huge-page
        // collapse (different resources): warm the `DenseW8`/`small_canon` OnceLocks on a background
        // thread while `QueensTt::new` faults + `MADV_COLLAPSE`s the multi-GB table, so the build
        // calls inside `from_tt_with_window` below hit the warmed caches instantly. Saves ~2s of
        // pre-search startup at n=16 (the build is ~3.4s, the alloc ~2s; serial they sum to ~5.4s,
        // overlapped ~3.4s). Zero hot-path / correctness risk — both inits are idempotent OnceLocks.
        let warm = std::thread::spawn(|| {
            DenseW8::build();
            small_canon_table();
        });
        let tt = QueensTt::new(bits);
        warm.join().unwrap();
        let mut s = Self::from_tt_with_window(tt, true);
        s.name = "iso-dense";
        // Dense layer on by construction (not the iso-window env gate). The ceiling is read
        // once here, threaded as `dense_k`, and resolved to a `const DK` at the root dispatch.
        // Default K=16 (the u128 labelled-code ceiling, 16·15/2 = 120 bits): with the pext-per-row
        // code-build the deep getK layers are cheap enough that raising the ceiling pays the whole
        // way up — n=16 node count is TT-independent (the cut is inherent, not eviction-driven) and
        // wall drops monotonically K=12→16 (16 GB single-run: 49.5s→34.4s = −30.6%). `QUEENS_DENSE_K`
        // ★ DEFAULT K=17 (--18): the W17 dense layer (136-bit 3-word labelled code, above the u128
        // K=16 ceiling) resolves pc==17 nodes (~21% of the n=16 node set, M_HITKEY-measured) directly
        // as getK leaves instead of cold recurse-spine entry probes (pc==17 is 99.8% COLD). With the
        // degree-ordered getK sweep (`DenseW8::ord_getk`, default-on) this is **−13% wall** vs the old
        // K=16 default (4-round n=16 A/B). K=17 is the wall sweet spot — W18-20 cut nodes hugely but
        // work-conserve (cyc/node grows ~proportionally; HIK A/B K20 = +4% total cyc). `QUEENS_DENSE_K`
        // (9..=20) overrides the ceiling. **`QUEENS_FAST=0` reverts the whole stack to the old K=16 +
        // no-getK-ordering default** (the A/B control). Sweep knobs: `QUEENS_WK` pins the ceiling
        // (wins), `QUEENS_HIK`=0/1 picks K17/K20, `QUEENS_W17`=1 forces ≥17.
        s.dense_k = env_u32("QUEENS_DENSE_K", 17).clamp(9, 20);
        if matches!(std::env::var("QUEENS_W17").as_deref(), Ok("1")) {
            s.dense_k = s.dense_k.max(17);
        }
        if let Some(k) = std::env::var("QUEENS_WK")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
        {
            s.dense_k = k.clamp(9, 20);
        }
        // Clean 0/1 A/B toggle for the high-K ceiling decision (ordering always on): =1 → K=20,
        // =0 → K=17. Lets the canonical harness flip K17↔K20 on one binary.
        if let Ok(v) = std::env::var("QUEENS_HIK") {
            s.dense_k = if v == "1" { 20 } else { 17 };
        }
        // `QUEENS_FAST=0` reverts the ceiling to the old K=16 default (the A/B control); wins over
        // the default and the W17/HIK forces (the getK-ordering half is reverted in `DenseW8::build`).
        if matches!(std::env::var("QUEENS_FAST").as_deref(), Ok("0")) {
            s.dense_k = 16;
        }
        if s.dense_k >= 17 {
            // Pre-build the wide (3-word) induced-mask tables up to the ceiling (~ms..tens-of-ms,
            // 3..24 MiB) at startup, off the hot getK path.
            warm_wide(s.dense_k as usize);
        }
        // Warm-restart OFF by default for iso-dense (--15): the warm_secs(=2) parallel warm pass +
        // staggered restart trims the node count a touch but its ramp costs more wall than it saves
        // now that the counting sort sped the kernel — a 4-round n=16 A/B (12 GB) measured restart-ON
        // at +3.2% wall / −1.5% nodes vs OFF (roots hitting all cores immediately wins). The balance
        // flipped vs the M_WAVE era where it was wall-neutral ("levers compound"). `QUEENS_WARM_RESTART=1`
        // re-enables. iso-flat/iso-window keep it off (control intact).
        s.warm_restart = matches!(std::env::var("QUEENS_WARM_RESTART").as_deref(), Ok("1"));
        // M_WAVE (fused ETC + sorted-batch recurse-child cutoff, Phase-1b) on by default for iso-dense:
        // a measured net win at n=16 (−16% nodes / −4% total cycles / −2.7% wall, 6-round interleaved A/B;
        // gates green, TT-sweep graceful to a 2 GB TT, verdict SECOND). It changes the node count *by
        // design* (verdict-preserving, an earlier cutoff of the same value), so it is **not** part of the
        // exact `--distinct` gate. `QUEENS_WAVE=0` disables, `=1` forces. iso-flat/iso-window keep it off
        // (the byte-identical A/B control stays intact).
        s.wave = !matches!(std::env::var("QUEENS_WAVE").as_deref(), Ok("0"));
        // Dynamic move ordering + ETC (`M_ORD_W`) is the iso-dense DEFAULT — the SUB-60 record
        // (−38% vs the M_WAVE default): re-sort each node's moves by current available-block degree
        // (`child0.popcount()` ascending = most-forcing first; instant-win sorts first ⇒ earliest
        // cutoff), with the M_WAVE ETC cut on top. The dispatch checks `ord` before `wave`, so this
        // wins; `wave` stays on as the `QUEENS_ORD=0` fallback. Mirrors `wave`/`warm_restart`:
        //   (unset) → M_ORD_W ;  QUEENS_ORD=0 → M_WAVE ;  =1 → M_ORD (ordering, no ETC) ;  =2 → M_ORD_W.
        // `QUEENS_ORD_ETC=1` still forces ETC on (the A/B harness toggle: ord fixed, ETC 0/1).
        // Verdict-preserving (changes the node count *by design* — an earlier cutoff of the same
        // value — so it is **not** part of the exact `--distinct` gate). iso-flat/iso-window keep it
        // off (their `from_tt_with_window` defaults are unchanged ⇒ control + `--distinct` intact).
        // ★ DEFAULT-ON (--7, promoted): skip all TT work for pc==18 nodes (all roots, {18}) — the band
        // is ~100% cold and its children are all getK leaves ⇒ cascade-free; −3.6% total cyc / −2.5%
        // wall at n=16, verdict-preserving. Off via `QUEENS_SKIP18=0` (the A/B control) or the
        // whole-stack revert `QUEENS_FAST=0`. Empty `skip18_squares` ⇒ all roots (n-agnostic default).
        s.skip18 = !matches!(std::env::var("QUEENS_SKIP18").as_deref(), Ok("0"))
            && !matches!(std::env::var("QUEENS_FAST").as_deref(), Ok("0"));
        // ★ DEFAULT-ON (2026-07-01, promoted): cross-root killer replies at the 2nd ply — each root
        // publishes its refuting reply square; later roots jump to already-proven killers (the table
        // is re-read mid-loop, so late-published killers land). n=16 A/B: −37.6% nodes / −43.3% wall,
        // cyc/node flat; record 23.44s → 14.60s. Verdict-preserving (a pure reorder of the depth-1
        // `.any()` — changes the node count *by design*, so NOT part of the exact `--distinct` gate;
        // iso-flat/iso-window keep the base default 0 ⇒ control + `--distinct` intact).
        // `QUEENS_KILLER=<k>` overrides (0 disables); `QUEENS_FAST=0` reverts the whole stack.
        s.killer_k = std::env::var("QUEENS_KILLER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(
                if matches!(std::env::var("QUEENS_FAST").as_deref(), Ok("0")) {
                    0
                } else {
                    4
                },
            )
            .min(8);
        // ★ DEFAULT-ON (2026-07-01, promoted with the killer): killer jumps at the deeper odd
        // plies (depth 3/5, per-band tables) stack another −7.5% nodes / −4.5% wall on the depth-1
        // win. `QUEENS_KILLER_DEEP=0` (or killer off / `QUEENS_FAST=0`) reverts.
        s.killer_deep = !matches!(std::env::var("QUEENS_KILLER_DEEP").as_deref(), Ok("0"))
            && !matches!(std::env::var("QUEENS_FAST").as_deref(), Ok("0"));
        // ★ DEFAULT-ON (2026-07-01): the ETC pc-gate (batch-probe off below pc 29) re-tested
        // POSITIVE in the killer regime — the killer cut leaves the TT ~7.5% full, so the
        // eviction-protection value that originally killed the gate (--3: +2.0% nodes) is gone;
        // now cyc/node −1.2% (every pair), total cyc −1.8%. `QUEENS_ETC_GATE=0` or
        // `QUEENS_ETC_PC=<pc>` overrides; `QUEENS_FAST=0` reverts.
        if std::env::var("QUEENS_ETC_PC").is_err()
            && !matches!(std::env::var("QUEENS_ETC_GATE").as_deref(), Ok("0"))
            && !matches!(std::env::var("QUEENS_FAST").as_deref(), Ok("0"))
        {
            s.etc_pc_gate = 29;
        }
        s.skip18_squares = std::env::var("QUEENS_SKIP18_ROOTS")
            .ok()
            .map(|v| {
                v.split(',')
                    .filter_map(|t| t.trim().parse::<u8>().ok())
                    .collect()
            })
            .unwrap_or_default();
        s.skip18_pcs = std::env::var("QUEENS_SKIP18_PCS")
            .ok()
            .map(|v| {
                v.split(',')
                    .filter_map(|t| t.trim().parse::<u32>().ok())
                    .filter(|&pc| pc < 64)
                    .fold(0u64, |m, pc| m | (1u64 << pc))
            })
            .filter(|&m| m != 0)
            .unwrap_or(1u64 << 18);
        // QUEENS_SKIP18_FRAC (0/1) = the A/B toggle; QUEENS_SKIP18_FRAC_M (default 4) = the 1/M fraction.
        s.skip18_frac = if std::env::var("QUEENS_SKIP18_FRAC").as_deref() == Ok("1") {
            env_u32("QUEENS_SKIP18_FRAC_M", 4).max(2)
        } else {
            1 // off
        };
        s.skip18_frac_pcs = std::env::var("QUEENS_SKIP18_FRAC_PCS")
            .ok()
            .map(|v| {
                v.split(',')
                    .filter_map(|t| t.trim().parse::<u32>().ok())
                    .filter(|&pc| pc < 64)
                    .fold(0u64, |m, pc| m | (1u64 << pc))
            })
            .unwrap_or(0);
        s.ord = !matches!(std::env::var("QUEENS_ORD").as_deref(), Ok("0"));
        s.ord_etc = !matches!(std::env::var("QUEENS_ORD").as_deref(), Ok("0") | Ok("1"))
            || std::env::var("QUEENS_ORD_ETC").as_deref() == Ok("1");
        s
    }

    /// Snapshot the work-stealing counters into a [`StealReport`] (cold; post-solve). Shared by the
    /// TTY diagnostic line and the `--to-file` JSON so both carry identical data.
    fn build_steal_report(&self) -> StealReport {
        let published = self.steal_published.load(Ordering::Relaxed);
        let fallback = self.steal_fallback.load(Ordering::Relaxed);
        let pc_hist: Vec<(u32, u64)> = self
            .steal_pc_hist
            .iter()
            .enumerate()
            .map(|(pc, c)| (pc as u32, c.load(Ordering::Relaxed)))
            .filter(|&(_, c)| c > 0)
            .collect();
        let pc_lo = pc_hist.first().map(|&(pc, _)| pc).unwrap_or(0);
        let pc_hi = pc_hist.last().map(|&(pc, _)| pc).unwrap_or(0);
        let sum_pc: u64 = pc_hist.iter().map(|&(pc, c)| pc as u64 * c).sum();
        let pc_mean = if published > 0 {
            sum_pc as f64 / published as f64
        } else {
            0.0
        };
        StealReport {
            published,
            fallback,
            pc_lo,
            pc_hi,
            pc_mean,
            pc_hist,
            width: self.steal_width,
            min_pc: self.steal_min_pc,
            max: self.steal_max,
            delay: self.steal_delay,
        }
    }

    /// Force the ABDADA in-flight-deferral path on (the `const ABDADA == true` `wins_inc_iter`
    /// monomorphisation), independent of the `QUEENS_ABDADA` env gate. Used by the agreement
    /// test to exercise deferral under the real parallel solver without mutating process env.
    #[cfg(test)]
    pub(crate) fn with_abdada(mut self) -> Self {
        self.abdada = true;
        self
    }

    /// Force frontier work-stealing on (the `const STEAL == true` monomorphisation, which also
    /// engages the ABDADA markers), independent of `QUEENS_STEAL`. The agreement test uses it to
    /// exercise stealing under the real parallel solver without mutating process env.
    #[cfg(test)]
    pub(crate) fn with_steal(mut self) -> Self {
        self.steal = true;
        self.steal_delay = 0; // arm immediately so the small-board test actually publishes + steals
        self
    }

    /// As [`IsoFlat::new`], but counting the distinct (tagged iso/D4) keys visited
    /// (`--distinct`). The HyperLogLog folded in at each `get` is lock-free, so it works
    /// under root parallelism.
    pub fn new_counting(bits: u32, hll_p: u32) -> Self {
        Self::from_tt(QueensTt::new_counting(bits, hll_p, false))
    }

    fn from_tt(tt: QueensTt) -> Self {
        Self::from_tt_with_window(tt, false)
    }

    fn from_tt_with_window(tt: QueensTt, window: bool) -> Self {
        let counting = tt.is_counting();
        let segment = tt.is_segmented();
        let assoc = tt.is_assoc();
        let sidecar = std::env::var("QUEENS_SIDECAR").as_deref() == Ok("1");
        let kprobe_on = matches!(std::env::var("QUEENS_KPROBE").as_deref(), Ok("1") | Ok("2"));
        IsoFlat {
            name: if window { "iso-window" } else { "iso-flat" },
            tt,
            att: OnceLock::new(),
            order8: OnceLock::new(),
            order_rank: OnceLock::new(),
            tiny_tt: (0..TINY_TABLE_SLOTS).map(|_| AtomicU8::new(0)).collect(),
            dense8: window.then(DenseW8::build),
            // The dense low-popcount layer is exclusive to `iso-dense` (set by `new_dense`);
            // `iso-flat`/`iso-window` keep the ceiling at 8 so they run identically to before.
            dense_k: 8,
            tiny_canon: small_canon_table(),
            par_depth: par_depth(),
            par_min_avail: par_min_avail_override(),
            // Off here (iso-flat/iso-window control); `new_dense` turns it on for iso-dense.
            warm_restart: std::env::var("QUEENS_WARM_RESTART").as_deref() == Ok("1"),
            warm_secs: env_u32("QUEENS_WARM_SECS", 2) as u64,
            warm_stagger_ms: env_u32("QUEENS_WARM_STAGGER_MS", 500) as u64,
            warm_phase: AtomicBool::new(false),
            warm_deadline: Arc::new(AtomicBool::new(false)),
            root_timing: std::env::var("QUEENS_ROOT_TIMING").as_deref() == Ok("1"),
            iso_max_avail: iso_flat_key_max_avail(),
            // Default 8 = off (pc≤7 band + W8 unchanged); clamp ≤12 for the [i8;4096] L1 memo.
            block_k: env_u32("QUEENS_BLOCK_K", 8).clamp(8, 12),
            eff_min_avail: AtomicU32::new(u32::MAX),
            root_done: AtomicU64::new(0),
            root_total: AtomicU64::new(0),
            nimber_oracle: std::env::var("QUEENS_NIMBER_ORACLE").as_deref() == Ok("1"),
            counting,
            hist: std::env::var("QUEENS_PC_HIST").as_deref() == Ok("1"),
            // Mirror the table's segmentation (it resolved `QUEENS_TT_SEGMENT` at startup), so
            // the subtree-handoff dispatch can pick `MODE = M_SEG` once and monomorphise.
            segment,
            assoc,
            sidecar,
            sched: std::env::var("QUEENS_SCHED").as_deref() == Ok("1"),
            sched_t0: std::sync::Mutex::new(Instant::now()),
            sched_recs: std::sync::Mutex::new(Vec::new()),
            par_ord: std::env::var("QUEENS_PAR_ORD").as_deref() == Ok("1"),
            split: std::env::var("QUEENS_SPLIT").as_deref() == Ok("1"),
            killer_k: std::env::var("QUEENS_KILLER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
                .min(8),
            killer_deep: std::env::var("QUEENS_KILLER_DEEP").as_deref() == Ok("1"),
            // Gather-time recurse-child prefetch (cheap-first PREFETCH lever). Resolved once here;
            // gated per node at the gather (off ⇒ byte-identical to the current prefetch behaviour).
            pf_deep: std::env::var("QUEENS_PFDEEP").as_deref() == Ok("1"),
            // `QUEENS_ETC_PC=<pc>` is the explicit threshold (for sweeping); `QUEENS_ETC_GATE=1` is the
            // harness 0/1 toggle selecting the tap-chosen default crossover (29). Off ⇒ 0 (disabled).
            etc_pc_gate: std::env::var("QUEENS_ETC_PC")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| {
                    if std::env::var("QUEENS_ETC_GATE").as_deref() == Ok("1") {
                        29
                    } else {
                        0
                    }
                }),
            pc_hist: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            prof: std::env::var("QUEENS_PROF").as_deref() == Ok("1"),
            prof_data: (0..4 * MAXPC).map(|_| AtomicU64::new(0)).collect(),
            wave: std::env::var("QUEENS_WAVE").as_deref() == Ok("1"),
            size: matches!(std::env::var("QUEENS_SIZE").as_deref(), Ok("1") | Ok("2")),
            size_wave: std::env::var("QUEENS_SIZE").as_deref() == Ok("2"),
            size_w: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            size_hll: Hll::new(SIZE_HLL_P),
            size_sample: Mutex::new(Vec::new()),
            size_rc_bits: rc_bits(),
            size_rc: Mutex::new(Vec::new()),
            size_rc_pc: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            wave_b: std::env::var("QUEENS_WAVE_B").as_deref() == Ok("1"),
            l0: std::env::var("QUEENS_L0").as_deref() == Ok("1"),
            wave_c: std::env::var("QUEENS_WAVE_C").as_deref() == Ok("1"),
            ord: matches!(std::env::var("QUEENS_ORD").as_deref(), Ok("1") | Ok("2")),
            // Off by default (the iso-flat/iso-window control + the A/B's `=0` arm); the iso-dense
            // constructor overrides it from `QUEENS_SKIP18` after this base build.
            skip18: false,
            skip18_squares: Vec::new(),
            skip18_pcs: 1u64 << 18,
            skip18_frac: 1,
            skip18_frac_pcs: 0,
            // `QUEENS_ORD=2` OR `QUEENS_ORD_ETC=1` (the latter lets the A/B harness toggle ETC with
            // `QUEENS_ORD=1` fixed ⇒ a clean M_ORD vs M_ORD_W interleaved comparison).
            ord_etc: std::env::var("QUEENS_ORD").as_deref() == Ok("2")
                || std::env::var("QUEENS_ORD_ETC").as_deref() == Ok("1"),
            decprobe: std::env::var("QUEENS_DECPROBE").as_deref() == Ok("1"),
            dhist: std::env::var("QUEENS_DHIST").as_deref() == Ok("1"),
            kprobe: kprobe_on,
            kprobe_canon: std::env::var("QUEENS_KPROBE").as_deref() == Ok("2"),
            kprobe_hll: (0..KPROBE_BANDS).map(|_| Hll::new(KPROBE_HLL_P)).collect(),
            kprobe_hll_c: (0..KPROBE_BANDS).map(|_| Hll::new(KPROBE_HLL_P)).collect(),
            kprobe_entries: (0..KPROBE_BANDS).map(|_| AtomicU64::new(0)).collect(),
            kprobe_s_hits: (0..KPROBE_BANDS).map(|_| AtomicU64::new(0)).collect(),
            kprobe_l_hits: (0..KPROBE_BANDS).map(|_| AtomicU64::new(0)).collect(),
            // The sim tag tables are the only heavy allocation (8 MiB + 512 MiB) — probe runs only.
            kprobe_sim_s: (0..if kprobe_on {
                1usize << KPROBE_SIM_S_BITS
            } else {
                0
            })
                .map(|_| AtomicU64::new(0))
                .collect(),
            kprobe_sim_l: (0..if kprobe_on {
                1usize << KPROBE_SIM_L_BITS
            } else {
                0
            })
                .map(|_| AtomicU64::new(0))
                .collect(),
            dec_nodes: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            dec_ncomp_sum: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            dec_ge2: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            dec_all_le8: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            dec_all_le_km1: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            dec_msz: (0..MAXPC * 17).map(|_| AtomicU64::new(0)).collect(),
            rank: std::env::var("QUEENS_RANK").as_deref() == Ok("1"),
            rank_nodes: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            rank_etc: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            rank_etc_probes: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            rank_nocut: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            rank_nocut_deg: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            rank_dist: (0..MAXPC * RANK_BUCKETS)
                .map(|_| AtomicU64::new(0))
                .collect(),
            cold: std::env::var("QUEENS_COLD").as_deref() == Ok("1"),
            cold_workers: Mutex::new(Vec::new()),
            hitkey: std::env::var("QUEENS_HITKEY").as_deref() == Ok("1"),
            hitkey_out: std::env::var("QUEENS_HITKEY_OUT")
                .unwrap_or_else(|_| "/tmp/queens-hitkeys.bin".to_string()),
            hitkey_recs: Mutex::new(Vec::new()),
            nimber_k: env_u32("QUEENS_NIMBER_K", 7).min(7),
            nimber_pc: env_u32("QUEENS_NIMBER_PC", 28),
            tiny8_direct: std::env::var("QUEENS_TINY8").as_deref() == Ok("1"),
            unroll: std::env::var("QUEENS_UNROLL").as_deref() == Ok("1"),
            iter_inc: std::env::var("QUEENS_ITER").as_deref() == Ok("1"),
            abdada: std::env::var("QUEENS_ABDADA").as_deref() == Ok("1"),
            steal: std::env::var("QUEENS_STEAL").as_deref() == Ok("1"),
            steal_delay: env_u32("QUEENS_STEAL_DELAY", 50) as u64,
            steal_armed: Arc::new(AtomicBool::new(false)),
            steal_width: env_u32("QUEENS_STEAL_WIDTH", 2).max(1),
            steal_min_pc: env_u32("QUEENS_STEAL_MIN_PC", 18),
            n_threads: rayon::current_num_threads().max(1),
            steal_max: env_u32(
                "QUEENS_STEAL_MAX",
                rayon::current_num_threads().max(1) as u32,
            ),
            active_splits: AtomicU64::new(0),
            deep_busy: AtomicUsize::new(0),
            steal_published: AtomicU64::new(0),
            steal_fallback: AtomicU64::new(0),
            steal_pc_hist: (0..MAXPC).map(|_| AtomicU64::new(0)).collect(),
            oracle_attempts: AtomicU64::new(0),
            oracle_hits: AtomicU64::new(0),
            oracle_comp_hits: AtomicU64::new(0),
            oracle_comp_misses: AtomicU64::new(0),
        }
    }

    /// Resolve a `popcount == 8` node from the complete dense W8 table. The 8-vertex
    /// available graph's Node-Kayles value is relabelling-invariant, so build the raw 28-bit
    /// upper-triangular edge code **directly** from the 8 attack rows (one pass, no
    /// intermediate `adj`/`closed` arrays, no canonicalisation) and index the bitset. The
    /// table is complete, so this always returns a value — there is never a flat-TT probe or
    /// a subtree expansion here. Only reached from the `WINDOW` instantiation, where
    /// `dense8` is always `Some`.
    #[inline]
    fn w8_get(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the WINDOW const generic at the (only) call site guarantees `dense8` is
        // `Some` (set by `from_tt_with_window(.., true)`). Drops a None-branch + panic target
        // from the hot body — the i-cache shave of the banked PROVE_LOSS win.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), 8);
        let mut verts = [0u8; 8];
        verts_of(avail, &mut verts);
        let mut code = 0usize;
        let mut bit = 0u32;
        for i in 0..8 {
            let row = att08(att, verts[i]);
            for &vj in verts.iter().take(8).skip(i + 1) {
                code |= (row.get(vj as u32) as usize) << bit;
                bit += 1;
            }
        }
        dense8.get(8, code)
    }

    /// Resolve a 9-vertex graph directly from W0..W8. Building the labelled edge
    /// code is the only board-geometry work; [`DenseW8::get9`] uses BMI2 projection
    /// for every child and performs no TT access or W9 allocation.
    #[inline]
    fn w9_get(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the DK≥9 const generic at the (only) call site guarantees `dense8` is `Some`.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), 9);
        let mut verts = [0u8; 9];
        verts_of(avail, &mut verts);
        let mut code = 0u64;
        let mut bit = 0u32;
        for i in 0..9 {
            let row = att08(att, verts[i]);
            for &vj in verts.iter().take(9).skip(i + 1) {
                code |= (row.get(vj as u32) as u64) << bit;
                bit += 1;
            }
        }
        dense8.get9(code)
    }

    /// Resolve a 10-vertex graph directly from W0..W8 (the W10 layer). Twin of
    /// [`w9_get`](Self::w9_get): build the 45-bit labelled edge code from the 10 attack rows,
    /// then [`DenseW8::get10`] sweeps every child (nested one ply into W9, else a W≤8 lookup).
    #[inline]
    fn w10_get(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the DK≥10 const generic at the (only) call site guarantees `dense8` is `Some`.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), 10);
        let mut verts = [0u8; 10];
        verts_of(avail, &mut verts);
        // pext code-build: each vert's K-bit adjacency row in one 4-word pext (`adj_row_pext`),
        // packed into the 45-bit upper-triangular labelled code. `off`/`width` const-fold per the
        // unrolled `i` (same const-fold the scalar build relied on for `bit>>6`); K≤11 ⇒ off<55 ⇒
        // no 64-bit word straddle ⇒ a single `u64`. Byte-identical code bits to the scalar build.
        let a = &avail.0;
        let c0 = a[0].count_ones();
        let c1 = c0 + a[1].count_ones();
        let c2 = c1 + a[2].count_ones();
        let cpre = [c0, c1, c2];
        let mut code = 0u64;
        let mut off = 0u32;
        // Root-adj carry: `packed` IS vertex i's labelled adjacency row (att masks are
        // self-inclusive, so clearing bit i gives exactly the `extract_adj` row); `get10_adj`
        // skips the root re-extraction (one pext + gap-reinsert per row). Same in w11..w16.
        let mut adjr = [0u16; MAX_DENSE_K];
        let mut iso = 0u16;
        for i in 0..10u32 {
            let packed = adj_row_pext(att08(att, verts[i as usize]), a, cpre);
            let row = (packed as u16) & !(1u16 << i);
            adjr[i as usize] = row;
            iso |= u16::from(row == 0) << i;
            let width = 10 - 1 - i;
            code |= ((packed >> (i + 1)) & ((1u64 << width) - 1)) << off;
            off += width;
        }
        dense8.get10_adj(code, &adjr, iso)
    }

    /// Resolve an 11-vertex graph directly from W0..W8 (the W11 layer). Twin of
    /// [`w9_get`](Self::w9_get) over the 55-bit labelled edge code; [`DenseW8::get11`] nests
    /// one ply into W10/W9 (else a W≤8 lookup) per child.
    #[inline]
    fn w11_get(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the DK≥11 const generic at the (only) call site guarantees `dense8` is `Some`.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), 11);
        let mut verts = [0u8; 11];
        verts_of(avail, &mut verts);
        // pext code-build (see `w10_get`): 11 rows of one 4-word pext into the 55-bit code (K≤11 ⇒
        // off<55 ⇒ single `u64`, no word straddle). Byte-identical code bits to the scalar build.
        let a = &avail.0;
        let c0 = a[0].count_ones();
        let c1 = c0 + a[1].count_ones();
        let c2 = c1 + a[2].count_ones();
        let cpre = [c0, c1, c2];
        let mut code = 0u64;
        let mut off = 0u32;
        // Root-adj carry (see `w10_get`).
        let mut adjr = [0u16; MAX_DENSE_K];
        let mut iso = 0u16;
        for i in 0..11u32 {
            let packed = adj_row_pext(att08(att, verts[i as usize]), a, cpre);
            let row = (packed as u16) & !(1u16 << i);
            adjr[i as usize] = row;
            iso |= u16::from(row == 0) << i;
            let width = 11 - 1 - i;
            code |= ((packed >> (i + 1)) & ((1u64 << width) - 1)) << off;
            off += width;
        }
        dense8.get11_adj(code, &adjr, iso)
    }

    /// Resolve a 12-vertex graph directly from W0..W8 (the W12 layer, the first past the
    /// `u64` code ceiling). Twin of [`w9_get`](Self::w9_get) but the 66-bit labelled code is
    /// a `u128`; [`DenseW8::get12`] nests one ply into W11/W10/W9 (else a W≤8 lookup) per child.
    #[inline]
    fn w12_get(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the DK≥12 const generic at the (only) call site guarantees `dense8` is `Some`.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), 12);
        let mut verts = [0u8; 12];
        verts_of(avail, &mut verts);
        // pext code-build (see `w10_get`): 12 rows of one 4-word pext, packed into the 66-bit code
        // as two `u64` words (low 0..63, high 64..65). Each row's `width`-bit contribution can
        // straddle the 64-bit word boundary, so split it when `lo + width > 64`. `off`/`width`/`lo`
        // and the straddle predicate const-fold per the unrolled `i`, so the spill branch DCEs on
        // the rows that don't cross. Byte-identical code bits to the scalar build.
        let a = &avail.0;
        let c0 = a[0].count_ones();
        let c1 = c0 + a[1].count_ones();
        let c2 = c1 + a[2].count_ones();
        let cpre = [c0, c1, c2];
        let mut words = [0u64; 2];
        let mut off = 0u32;
        // Root-adj carry (see `w10_get`).
        let mut adjr = [0u16; MAX_DENSE_K];
        let mut iso = 0u16;
        for i in 0..12u32 {
            let packed = adj_row_pext(att08(att, verts[i as usize]), a, cpre);
            let row = (packed as u16) & !(1u16 << i);
            adjr[i as usize] = row;
            iso |= u16::from(row == 0) << i;
            let width = 12 - 1 - i;
            let contrib = (packed >> (i + 1)) & ((1u64 << width) - 1);
            let lo = off & 63;
            words[(off >> 6) as usize] |= contrib << lo;
            if lo + width > 64 {
                words[((off >> 6) + 1) as usize] |= contrib >> (64 - lo);
            }
            off += width;
        }
        let code = (words[0] as u128) | ((words[1] as u128) << 64);
        dense8.get12_adj(code, &adjr, iso)
    }

    /// Resolve a 13-vertex graph directly from W0..W8 (the W13 layer). Twin of
    /// [`w12_get`](Self::w12_get) over the 78-bit labelled edge code (`u128`); [`DenseW8::get13`]
    /// nests one ply into W12 (66-bit child) / W11..W9 / a W≤8 lookup per child.
    #[inline]
    fn w13_get(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the DK≥13 const generic at the (only) call site guarantees `dense8` is `Some`.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), 13);
        let mut verts = [0u8; 13];
        verts_of(avail, &mut verts);
        // pext code-build (see `w12_get`): 13 rows of one 4-word pext, packed into the 78-bit code
        // (two `u64` words, low 0..63, high 64..77, straddle split per row). Byte-identical bits.
        let a = &avail.0;
        let c0 = a[0].count_ones();
        let c1 = c0 + a[1].count_ones();
        let c2 = c1 + a[2].count_ones();
        let cpre = [c0, c1, c2];
        let mut words = [0u64; 2];
        let mut off = 0u32;
        // Root-adj carry (see `w10_get`).
        let mut adjr = [0u16; MAX_DENSE_K];
        let mut iso = 0u16;
        for i in 0..13u32 {
            let packed = adj_row_pext(att08(att, verts[i as usize]), a, cpre);
            let row = (packed as u16) & !(1u16 << i);
            adjr[i as usize] = row;
            iso |= u16::from(row == 0) << i;
            let width = 13 - 1 - i;
            let contrib = (packed >> (i + 1)) & ((1u64 << width) - 1);
            let lo = off & 63;
            words[(off >> 6) as usize] |= contrib << lo;
            if lo + width > 64 {
                words[((off >> 6) + 1) as usize] |= contrib >> (64 - lo);
            }
            off += width;
        }
        let code = (words[0] as u128) | ((words[1] as u128) << 64);
        dense8.get13_adj(code, &adjr, iso)
    }

    /// Resolve a 14-vertex graph directly from W0..W8 (the W14 layer, the `u128` code ceiling).
    /// Twin of [`w13_get`](Self::w13_get) over the 91-bit labelled edge code (`u128`);
    /// [`DenseW8::get14`] nests one ply into W13 (78-bit) / W12 (66-bit) / W11..W9 / a W≤8 lookup.
    #[inline]
    fn w14_get(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the DK≥14 const generic at the (only) call site guarantees `dense8` is `Some`.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), 14);
        let mut verts = [0u8; 14];
        verts_of(avail, &mut verts);
        // pext code-build (see `w12_get`): 14 rows of one 4-word pext, packed into the 91-bit code
        // (two `u64` words, low 0..63, high 64..90, straddle split per row). Byte-identical bits.
        let a = &avail.0;
        let c0 = a[0].count_ones();
        let c1 = c0 + a[1].count_ones();
        let c2 = c1 + a[2].count_ones();
        let cpre = [c0, c1, c2];
        let mut words = [0u64; 2];
        let mut off = 0u32;
        // Root-adj carry (see `w10_get`).
        let mut adjr = [0u16; MAX_DENSE_K];
        let mut iso = 0u16;
        for i in 0..14u32 {
            let packed = adj_row_pext(att08(att, verts[i as usize]), a, cpre);
            let row = (packed as u16) & !(1u16 << i);
            adjr[i as usize] = row;
            iso |= u16::from(row == 0) << i;
            let width = 14 - 1 - i;
            let contrib = (packed >> (i + 1)) & ((1u64 << width) - 1);
            let lo = off & 63;
            words[(off >> 6) as usize] |= contrib << lo;
            if lo + width > 64 {
                words[((off >> 6) + 1) as usize] |= contrib >> (64 - lo);
            }
            off += width;
        }
        let code = (words[0] as u128) | ((words[1] as u128) << 64);
        dense8.get14_adj(code, &adjr, iso)
    }

    /// Resolve a 15-vertex graph directly from W0..W8 (the W15 layer). Twin of
    /// [`w14_get`](Self::w14_get) over the 105-bit labelled edge code (`u128`); [`DenseW8::get15`]
    /// nests one ply into W14..W12 (`>64`-bit children) / W11..W9 / a W≤8 lookup.
    #[inline]
    fn w15_get(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the DK≥15 const generic at the (only) call site guarantees `dense8` is `Some`.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), 15);
        let mut verts = [0u8; 15];
        verts_of(avail, &mut verts);
        // pext code-build (see `w12_get`): 15 rows of one 4-word pext, packed into the 105-bit
        // code (two `u64` words, low 0..63, high 64..104, straddle split per row). Byte-identical.
        let a = &avail.0;
        let c0 = a[0].count_ones();
        let c1 = c0 + a[1].count_ones();
        let c2 = c1 + a[2].count_ones();
        let cpre = [c0, c1, c2];
        let mut words = [0u64; 2];
        let mut off = 0u32;
        // Root-adj carry (see `w10_get`).
        let mut adjr = [0u16; MAX_DENSE_K];
        let mut iso = 0u16;
        for i in 0..15u32 {
            let packed = adj_row_pext(att08(att, verts[i as usize]), a, cpre);
            let row = (packed as u16) & !(1u16 << i);
            adjr[i as usize] = row;
            iso |= u16::from(row == 0) << i;
            let width = 15 - 1 - i;
            let contrib = (packed >> (i + 1)) & ((1u64 << width) - 1);
            let lo = off & 63;
            words[(off >> 6) as usize] |= contrib << lo;
            if lo + width > 64 {
                words[((off >> 6) + 1) as usize] |= contrib >> (64 - lo);
            }
            off += width;
        }
        let code = (words[0] as u128) | ((words[1] as u128) << 64);
        dense8.get15_adj(code, &adjr, iso)
    }

    /// Resolve a 16-vertex graph directly from W0..W8 (the W16 layer, the `u128` code ceiling).
    /// Twin of [`w15_get`](Self::w15_get) over the 120-bit labelled edge code (`u128`);
    /// [`DenseW8::get16`] nests one ply into W15..W12 / W11..W9 / a W≤8 lookup.
    #[inline]
    fn w16_get(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the DK≥16 const generic at the (only) call site guarantees `dense8` is `Some`.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), 16);
        let mut verts = [0u8; 16];
        verts_of(avail, &mut verts);
        // pext code-build (see `w12_get`): 16 rows of one 4-word pext, packed into the 120-bit
        // code (two `u64` words, low 0..63, high 64..119, straddle split per row). Byte-identical.
        let a = &avail.0;
        let c0 = a[0].count_ones();
        let c1 = c0 + a[1].count_ones();
        let c2 = c1 + a[2].count_ones();
        let cpre = [c0, c1, c2];
        let mut words = [0u64; 2];
        let mut off = 0u32;
        // Root-adj carry (see `w10_get`).
        let mut adjr = [0u16; MAX_DENSE_K];
        let mut iso = 0u16;
        for i in 0..16u32 {
            let packed = adj_row_pext(att08(att, verts[i as usize]), a, cpre);
            let row = (packed as u16) & !(1u16 << i);
            adjr[i as usize] = row;
            iso |= u16::from(row == 0) << i;
            let width = 16 - 1 - i;
            let contrib = (packed >> (i + 1)) & ((1u64 << width) - 1);
            let lo = off & 63;
            words[(off >> 6) as usize] |= contrib << lo;
            if lo + width > 64 {
                words[((off >> 6) + 1) as usize] |= contrib >> (64 - lo);
            }
            off += width;
        }
        let code = (words[0] as u128) | ((words[1] as u128) << 64);
        dense8.get16_adj(code, &adjr, iso)
    }

    /// Resolve a `K`-vertex graph (K=17..20) directly from W0..W8 — the wide W_K layers, one or
    /// more above the `u128` K=16 ceiling (the `K·(K-1)/2` = 136..190-bit labelled code spans
    /// **three** `u64` words). Twin of [`w16_get`](Self::w16_get): one `adj_row_pext` per vertex
    /// packed into the 3-word code, then [`DenseW8::get_dyn_wide`] sweeps every child. Only reached
    /// at the matching `DK >= K` instantiation. The straddle only crosses the 0→1 and 1→2 word
    /// boundaries (max bit 189 < 192), so `wi+1 ≤ 2` always.
    #[inline]
    fn w_wide_get<const K: u32>(&self, att: &[[Bits; 8]], avail: Bits) -> bool {
        // SAFETY: the DK≥K const generic at the (only) call site guarantees `dense8` is `Some`.
        let dense8 = unsafe { self.dense8.as_ref().unwrap_unchecked() };
        debug_assert_eq!(avail.popcount(), K);
        let mut verts = [0u8; 20];
        verts_of(avail, &mut verts[..K as usize]);
        let a = &avail.0;
        let c0 = a[0].count_ones();
        let c1 = c0 + a[1].count_ones();
        let c2 = c1 + a[2].count_ones();
        let cpre = [c0, c1, c2];
        let mut words = [0u64; 3];
        let mut off = 0u32;
        for i in 0..K {
            let packed = adj_row_pext(att08(att, verts[i as usize]), a, cpre);
            let width = K - 1 - i;
            let contrib = (packed >> (i + 1)) & ((1u64 << width) - 1);
            let lo = off & 63;
            let wi = (off >> 6) as usize;
            words[wi] |= contrib << lo;
            if lo + width > 64 {
                words[wi + 1] |= contrib >> (64 - lo);
            }
            off += width;
        }
        dense8.get_dyn_wide(K as usize, &words)
    }

    #[inline]
    fn att(&self, q: &Queens) -> &[[Bits; 8]] {
        self.att.get_or_init(|| build_att(q))
    }

    #[inline]
    fn order8(&self, q: &Queens) -> &[u8] {
        self.order8.get_or_init(|| {
            q.order
                .iter()
                .map(|&sq| {
                    debug_assert!(sq < 256);
                    sq as u8
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
    }

    #[inline]
    fn order_rank(&self, q: &Queens) -> &[u8] {
        self.order_rank.get_or_init(|| {
            let mut rank = vec![0u8; (q.n * q.n) as usize].into_boxed_slice();
            for (r, &sq) in q.order.iter().enumerate() {
                rank[sq as usize] = r as u8;
            }
            rank
        })
    }

    #[inline]
    fn tt_get_key<const COUNT: bool>(&self, key: Bits) -> Option<u8> {
        if COUNT {
            self.tt.get(key)
        } else {
            let (route, fp) = QueensTt::hash128(key);
            self.tt.get_hashed(route, fp)
        }
    }

    #[inline]
    fn tt_put_key<const COUNT: bool>(&self, key: Bits, val: u8) {
        if COUNT {
            self.tt.put(key, val);
        } else {
            let (route, fp) = QueensTt::hash128(key);
            self.tt.put_hashed(route, fp, val);
        }
    }

    #[inline]
    fn tt_get_h<const COUNT: bool>(&self, key: Bits, route: u64, fp: u64) -> Option<u8> {
        if COUNT {
            self.tt.get_h(key, route, fp)
        } else {
            self.tt.get_hashed(route, fp)
        }
    }

    #[inline]
    fn tt_put_h<const COUNT: bool>(&self, key: Bits, route: u64, fp: u64, val: u8) {
        if COUNT {
            self.tt.put_h(key, route, fp, val);
        } else {
            self.tt.put_hashed(route, fp, val);
        }
    }

    /// [`wins_inc`](Self::wins_inc) flat-TT lookup, dispatched on the `const MODE`: `M_SEG`
    /// routes by per-popcount band ([`QueensTt::get_seg_hashed`], `pc` = the node's available
    /// popcount); `M_HIST`/`M_NORMAL` use the flat probe (identical to the control). The branch
    /// is on a `const`, so each instantiation compiles to exactly one path — no per-node test.
    #[inline]
    fn mtt_get<const COUNT: bool, const MODE: u8>(
        &self,
        key: Bits,
        route: u64,
        fp: u64,
        pc: u32,
    ) -> Option<u8> {
        // TT-layout axis (resolved-once runtime fields), orthogonal to the search-strategy `MODE`:
        // the winning M_ORD_W deep loop otherwise always takes the flat `else` branch, so these
        // expose seg/assoc to it. `assoc` ⇒ 8-way cache-line bucket; `segment` ⇒ 1-way band.
        if self.assoc {
            return self.tt.get_assoc_hashed(route, fp, pc);
        }
        if self.segment {
            return self.tt.get_seg_hashed(route, fp, pc);
        }
        if MODE == M_SEG {
            self.tt.get_seg_hashed(route, fp, pc)
        } else if MODE == M_L0 {
            // L0 dedup: serve a recurring key from the per-worker L0 cache; on a miss fall to the flat
            // TT and populate L0 with the solved value. DCEs to the plain `tt_get_h` for every other MODE.
            if let Some(v) = l0_get(route, fp) {
                Some(v)
            } else {
                let g = self.tt_get_h::<COUNT>(key, route, fp);
                if let Some(v) = g {
                    l0_put(route, fp, v);
                }
                g
            }
        } else {
            self.tt_get_h::<COUNT>(key, route, fp)
        }
    }

    /// [`wins_inc`](Self::wins_inc) flat-TT store, MODE-dispatched (twin of [`mtt_get`](Self::mtt_get)).
    #[inline]
    fn mtt_put<const COUNT: bool, const MODE: u8>(
        &self,
        key: Bits,
        route: u64,
        fp: u64,
        pc: u32,
        val: u8,
    ) {
        if self.assoc {
            return self.tt.put_assoc_hashed(route, fp, pc, val);
        }
        if self.segment {
            return self.tt.put_seg_hashed(route, fp, pc, val);
        }
        if MODE == M_SEG {
            self.tt.put_seg_hashed(route, fp, pc, val);
        } else if MODE == M_L0 {
            self.tt_put_h::<COUNT>(key, route, fp, val);
            l0_put(route, fp, val); // keep L0 hot with every just-solved value
        } else {
            self.tt_put_h::<COUNT>(key, route, fp, val);
        }
    }

    /// Prefetch the slot a child key will land in, MODE-dispatched: `M_SEG` prefetches its
    /// band slot (`pc` = the child's popcount); otherwise the flat slot.
    #[inline]
    fn mtt_prefetch<const MODE: u8>(&self, route: u64, pc: u32) {
        if self.assoc {
            return self.tt.prefetch_assoc_hashed(route, pc);
        }
        if self.segment {
            return self.tt.prefetch_seg_hashed(route, pc);
        }
        if MODE == M_SEG {
            self.tt.prefetch_seg_hashed(route, pc);
        } else {
            self.tt.prefetch_h(route);
        }
    }

    /// [`par_wins_inc`](Self::par_wins_inc) split-node lookup. Only the *few* shallow split
    /// nodes reach this (never the deep hot path), so the segmentation choice is a cheap
    /// resolved-once runtime branch on `self.segment` rather than another `const` to thread.
    #[inline]
    fn par_tt_get<const COUNT: bool>(&self, key: Bits, pc: u32) -> Option<u8> {
        if self.assoc {
            let (route, fp) = QueensTt::hash128(key);
            self.tt.get_assoc_hashed(route, fp, pc)
        } else if self.segment {
            let (route, fp) = QueensTt::hash128(key);
            self.tt.get_seg_hashed(route, fp, pc)
        } else {
            self.tt_get_key::<COUNT>(key)
        }
    }

    /// [`par_wins_inc`](Self::par_wins_inc) split-node store (twin of [`par_tt_get`](Self::par_tt_get)).
    #[inline]
    fn par_tt_put<const COUNT: bool>(&self, key: Bits, pc: u32, val: u8) {
        if self.assoc {
            let (route, fp) = QueensTt::hash128(key);
            self.tt.put_assoc_hashed(route, fp, pc, val);
        } else if self.segment {
            let (route, fp) = QueensTt::hash128(key);
            self.tt.put_seg_hashed(route, fp, pc, val);
        } else {
            self.tt_put_key::<COUNT>(key, val);
        }
    }

    /// The single canonical key for a node from its 8 orientations: the tiny-table graph-iso
    /// key (tagged) when the available graph is small enough to merge cheaply, else the
    /// incremental D4 key (tagged into a disjoint namespace). One key per node.
    #[inline]
    fn node_key(&self, q: &Queens, orient: &[Bits; 8]) -> Bits {
        let avail = orient[0];
        let pc = avail.popcount();
        if pc <= self.iso_max_avail {
            let h = if pc <= 7 {
                q.iso_key_tiny_table_pc(avail, pc, self.tiny_canon)
            } else if pc == 8 && self.tiny8_direct {
                q.iso_key8_direct(avail)
            } else {
                q.iso_key_fast(avail)
            };
            graph_bits(h)
        } else {
            d4_bits(lex_min8(orient))
        }
    }

    /// The iso-band key from `avail` alone (no orientations), given its already-computed
    /// popcount `pc ≤ iso_max_avail`: the cheap tiny-table iso key for `pc ≤ 7`, the WL fast
    /// key above. The [`wins_tiny`](Self::wins_tiny) tail never needs the 8 D4 orientations,
    /// so this skips `node_key`'s `lex_min8`/`child_orient` machinery entirely.
    #[inline]
    fn iso_node_key(&self, q: &Queens, avail: Bits, pc: u32) -> Bits {
        let h = if pc <= 7 {
            q.iso_key_tiny_table_pc(avail, pc, self.tiny_canon)
        } else if pc == 8 && self.tiny8_direct {
            q.iso_key8_direct(avail)
        } else {
            q.iso_key_fast(avail)
        };
        graph_bits(h)
    }

    /// Lever-B leaf oracle: if `avail` decomposes into connected components each
    /// `≤ nimber_k`, return its exact win/loss via the Sprague-Grundy nimber (XOR of the
    /// per-component nimbers, `win ⇔ ≠0`) with no recursion; else `None` (a big component
    /// remains → fall through to the normal search). Sound: impartial normal-play, the
    /// components are independent games (a queen in one component only removes squares of
    /// that component), so the position nimber is their nim-sum.
    #[inline]
    fn try_oracle_nimber(&self, q: &Queens, avail: Bits) -> Option<u8> {
        self.oracle_attempt();
        let mut x = 0u8;
        let mut rem = avail;
        while let Some(start) = rem.lowest() {
            let comp = q.component(start, avail);
            if comp.popcount() > self.nimber_k {
                return None;
            }
            rem = rem.and_not(comp);
            x ^= self.comp_nimber(q, comp);
        }
        self.oracle_hit();
        Some(x)
    }

    /// Nimber of a single **connected** component (`≤ nimber_k ≤ 7` ⇒ `iso_key_tiny_table`
    /// is the cheap *complete* iso key), memoised in the flat TT under a disjoint
    /// nimber-tagged namespace. Recurses through the component's children (each strictly
    /// smaller ⇒ stays in-band) via the nim-sum of *their* components.
    fn comp_nimber(&self, q: &Queens, comp: Bits) -> u8 {
        let key = comp_nimber_bits(q.iso_key_tiny_table_in(comp, self.tiny_canon));
        let (route, fp) = QueensTt::hash128(key);
        if let Some(v) = self.tt.get_hashed(route, fp) {
            self.oracle_comp_hit();
            return v;
        }
        self.oracle_comp_miss();
        let mut seen = 0u64; // bitset of child nimbers (all < n ≤ 16 < 64)
        let mut rem = comp;
        while let Some(sq) = rem.lowest() {
            rem = rem.and_not(single(sq));
            // place a queen on sq: remove sq + every square it attacks (q.attack[sq]
            // includes sq), leaving a (possibly disconnected) smaller sub-position.
            let child = comp.and_not(q.attack[sq as usize]);
            seen |= 1u64 << self.position_nimber(q, child);
        }
        let mex = (!seen).trailing_zeros() as u8;
        self.tt.put_hashed(route, fp, mex);
        mex
    }

    /// Nimber of a possibly-disconnected in-band sub-position = nim-sum of its components'
    /// nimbers. Only ever called on children of an in-band component (so every component
    /// is `≤ nimber_k`; no size re-check needed).
    fn position_nimber(&self, q: &Queens, mask: Bits) -> u8 {
        let mut x = 0u8;
        let mut rem = mask;
        while let Some(start) = rem.lowest() {
            let comp = q.component(start, mask);
            rem = rem.and_not(comp);
            x ^= self.comp_nimber(q, comp);
        }
        x
    }

    #[inline]
    fn oracle_attempt(&self) {
        ORACLE_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            a.attempts += 1;
            if a.attempts + a.hits + a.comp_hits + a.comp_misses >= ORACLE_FLUSH {
                self.flush_oracle_acc(&mut a);
            }
        });
    }

    #[inline]
    fn oracle_hit(&self) {
        ORACLE_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            a.hits += 1;
            if a.attempts + a.hits + a.comp_hits + a.comp_misses >= ORACLE_FLUSH {
                self.flush_oracle_acc(&mut a);
            }
        });
    }

    #[inline]
    fn oracle_comp_hit(&self) {
        ORACLE_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            a.comp_hits += 1;
            if a.attempts + a.hits + a.comp_hits + a.comp_misses >= ORACLE_FLUSH {
                self.flush_oracle_acc(&mut a);
            }
        });
    }

    #[inline]
    fn oracle_comp_miss(&self) {
        ORACLE_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            a.comp_misses += 1;
            if a.attempts + a.hits + a.comp_hits + a.comp_misses >= ORACLE_FLUSH {
                self.flush_oracle_acc(&mut a);
            }
        });
    }

    fn flush_oracle_acc(&self, a: &mut OracleAcc) {
        if a.attempts != 0 {
            self.oracle_attempts
                .fetch_add(a.attempts, Ordering::Relaxed);
            a.attempts = 0;
        }
        if a.hits != 0 {
            self.oracle_hits.fetch_add(a.hits, Ordering::Relaxed);
            a.hits = 0;
        }
        if a.comp_hits != 0 {
            self.oracle_comp_hits
                .fetch_add(a.comp_hits, Ordering::Relaxed);
            a.comp_hits = 0;
        }
        if a.comp_misses != 0 {
            self.oracle_comp_misses
                .fetch_add(a.comp_misses, Ordering::Relaxed);
            a.comp_misses = 0;
        }
    }

    fn drain_oracle_local(&self) {
        ORACLE_ACC.with(|cell| self.flush_oracle_acc(&mut cell.borrow_mut()));
    }

    fn drain_oracle_all(&self) {
        rayon::broadcast(|_| ORACLE_ACC.with(|cell| self.flush_oracle_acc(&mut cell.borrow_mut())));
        self.drain_oracle_local();
    }

    /// Tally one flat-TT put at available-popcount `pc` into this worker's thread-local
    /// histogram (`QUEENS_PC_HIST` measurement only — reached solely from the `HIST = true`
    /// monomorphisation of [`wins_inc`](Self::wins_inc), so the production path has no bump).
    #[inline]
    fn hist_bump(&self, pc: u32) {
        PC_HIST_ACC.with(|cell| cell.borrow_mut()[pc as usize] += 1);
    }

    /// Merge this worker's thread-local put histogram into the shared [`pc_hist`](Self::pc_hist)
    /// and clear it (so a later solve in this process starts fresh).
    fn drain_hist_local(&self) {
        PC_HIST_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            for (i, v) in a.iter_mut().enumerate() {
                if *v != 0 {
                    self.pc_hist[i].fetch_add(*v, Ordering::Relaxed);
                    *v = 0;
                }
            }
        });
    }

    /// Merge every rayon worker's put histogram into the shared total (the parallel twin of
    /// [`drain_hist_local`](Self::drain_hist_local)).
    fn drain_hist_all(&self) {
        rayon::broadcast(|_| self.drain_hist_local());
        self.drain_hist_local();
    }

    /// Merge this worker's thread-local profile ([`PROF_ACC`]) into the shared [`prof`](Self::prof_data)
    /// totals and clear it. Layout: `prof_data[metric * MAXPC + pc]`, metric ∈ {get-cyc, get-n,
    /// put-cyc, nodes}.
    fn drain_prof_local(&self) {
        PROF_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            for pc in 0..MAXPC {
                if a.get_n[pc] == 0 && a.nodes[pc] == 0 {
                    continue;
                }
                self.prof_data[pc].fetch_add(a.get_cyc[pc], Ordering::Relaxed);
                self.prof_data[MAXPC + pc].fetch_add(a.get_n[pc], Ordering::Relaxed);
                self.prof_data[2 * MAXPC + pc].fetch_add(a.put_cyc[pc], Ordering::Relaxed);
                self.prof_data[3 * MAXPC + pc].fetch_add(a.nodes[pc], Ordering::Relaxed);
                a.get_cyc[pc] = 0;
                a.get_n[pc] = 0;
                a.put_cyc[pc] = 0;
                a.nodes[pc] = 0;
            }
        });
    }

    /// Merge every rayon worker's profile into the shared total (parallel twin of
    /// [`drain_prof_local`](Self::drain_prof_local)).
    fn drain_prof_all(&self) {
        rayon::broadcast(|_| self.drain_prof_local());
        self.drain_prof_local();
    }

    /// Merge this worker's `M_SIZE` accumulator ([`SIZE_ACC`]) into the shared sizing state and
    /// clear it: per-pc width into [`size_w`](Self::size_w), the HLL register slice into
    /// [`size_hll`](Self::size_hll) (register-wise max), and the route sample into
    /// [`size_sample`](Self::size_sample). Cold — called once per worker at drain.
    fn drain_size_local(&self) {
        SIZE_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            for pc in 0..MAXPC {
                if a.w[pc] != 0 {
                    self.size_w[pc].fetch_add(a.w[pc], Ordering::Relaxed);
                    a.w[pc] = 0;
                }
            }
            self.size_hll.merge_from(&a.hll);
            a.hll.iter_mut().for_each(|r| *r = 0);
            if !a.sample.is_empty() {
                let mut s = self.size_sample.lock().unwrap();
                // Cap the shared sample so the cold sort stays bounded even across workers.
                let room = SIZE_SAMPLE_CAP.saturating_sub(s.len());
                let take = room.min(a.sample.len());
                s.extend_from_slice(&a.sample[..take]);
                a.sample.clear();
            }
            // Recency-cache: record this worker's (probes, hits) so the report can show the
            // top-probe workers (the 2 slow roots / giant-root tail) separately from the global mix.
            if a.rc_probes != 0 {
                self.size_rc.lock().unwrap().push(RcWorker {
                    probes: a.rc_probes,
                    hits: a.rc_hits,
                    win_h: a.rc_win_h,
                    win_p: a.rc_win_p,
                });
                for pc in 0..MAXPC {
                    if a.rc_per_pc_hits[pc] != 0 {
                        self.size_rc_pc[pc].fetch_add(a.rc_per_pc_hits[pc], Ordering::Relaxed);
                        a.rc_per_pc_hits[pc] = 0;
                    }
                }
                a.rc_probes = 0;
                a.rc_hits = 0;
                a.rc = Vec::new();
                a.rc_win_h = [0; RC_WINDOWS];
                a.rc_win_p = [0; RC_WINDOWS];
            }
        });
    }

    /// Merge every rayon worker's `M_SIZE` accumulator into the shared sizing state (parallel
    /// twin of [`drain_size_local`](Self::drain_size_local)).
    fn drain_size_all(&self) {
        rayon::broadcast(|_| self.drain_size_local());
        self.drain_size_local();
    }

    /// Merge this worker's `M_DECPROBE` accumulator ([`DEC_ACC`]) into the shared `dec_*` totals.
    fn drain_dec_local(&self) {
        DEC_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            for pc in 0..MAXPC {
                if a.nodes[pc] == 0 {
                    continue;
                }
                self.dec_nodes[pc].fetch_add(a.nodes[pc], Ordering::Relaxed);
                self.dec_ncomp_sum[pc].fetch_add(a.ncomp_sum[pc], Ordering::Relaxed);
                self.dec_ge2[pc].fetch_add(a.ge2[pc], Ordering::Relaxed);
                self.dec_all_le8[pc].fetch_add(a.all_le8[pc], Ordering::Relaxed);
                self.dec_all_le_km1[pc].fetch_add(a.all_le_km1[pc], Ordering::Relaxed);
                for s in 0..17 {
                    if a.msz_dist[pc][s] != 0 {
                        self.dec_msz[pc * 17 + s].fetch_add(a.msz_dist[pc][s], Ordering::Relaxed);
                        a.msz_dist[pc][s] = 0;
                    }
                }
                a.nodes[pc] = 0;
                a.ncomp_sum[pc] = 0;
                a.ge2[pc] = 0;
                a.all_le8[pc] = 0;
                a.all_le_km1[pc] = 0;
            }
        });
    }

    fn drain_dec_all(&self) {
        rayon::broadcast(|_| self.drain_dec_local());
        self.drain_dec_local();
    }

    /// Merge this worker's `M_KPROBE` accumulator ([`KPROBE_ACC`]) into the shared `kprobe_*` totals.
    fn drain_kprobe_local(&self) {
        KPROBE_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            for b in 0..KPROBE_BANDS {
                if a.entries[b] != 0 {
                    self.kprobe_entries[b].fetch_add(a.entries[b], Ordering::Relaxed);
                    self.kprobe_s_hits[b].fetch_add(a.sim_s_hits[b], Ordering::Relaxed);
                    self.kprobe_l_hits[b].fetch_add(a.sim_l_hits[b], Ordering::Relaxed);
                    a.entries[b] = 0;
                    a.sim_s_hits[b] = 0;
                    a.sim_l_hits[b] = 0;
                    self.kprobe_hll[b].merge_from(&a.hll[b]);
                    if self.kprobe_canon {
                        self.kprobe_hll_c[b].merge_from(&a.hll_c[b]);
                    }
                }
            }
        });
    }

    fn drain_kprobe_all(&self) {
        rayon::broadcast(|_| self.drain_kprobe_local());
        self.drain_kprobe_local();
    }

    /// Print the `M_KPROBE` (`QUEENS_KPROBE=1`) getK-entry repeat-rate report. Gates the code-keyed
    /// getK memo lever: `repeat× = entries/distinct` is the infinite-memo hit ceiling per band, and
    /// the two simulated direct-mapped tables give the realizable hit rate at an L3-resident size
    /// (2^20 slots) and a DRAM size (2^26). A memo pays only where hit% × getK-cost saved exceeds
    /// the probe's load cost — the deep bands (K≥13, the expensive sweeps) are the ones to read.
    fn print_kprobe_report(&self) {
        let ld = |b: &[AtomicU64], i: usize| b[i].load(Ordering::Relaxed);
        let total: u64 = (0..KPROBE_BANDS).map(|b| ld(&self.kprobe_entries, b)).sum();
        println!(
            "\n  getK entry repeat-rate (QUEENS_KPROBE) — {} getK entries; memo sim {} KiB / {} MiB",
            commas(total),
            (1u64 << KPROBE_SIM_S_BITS) * 8 / 1024,
            (1u64 << KPROBE_SIM_L_BITS) * 8 / (1024 * 1024),
        );
        println!(
            "    {:>3} {:>15} {:>15} {:>8} {:>9} {:>9}{}",
            "pc",
            "entries",
            "distinct",
            "repeat×",
            "hit%-s",
            "hit%-l",
            if self.kprobe_canon {
                format!(" {:>15} {:>9}", "canon-dist", "c-rep×")
            } else {
                String::new()
            },
        );
        let mut tot_distinct = 0.0f64;
        let mut tot_cdist = 0.0f64;
        let (mut tot_s, mut tot_l) = (0u64, 0u64);
        for b in 0..KPROBE_BANDS {
            let n = ld(&self.kprobe_entries, b);
            if n == 0 {
                continue;
            }
            let distinct = self.kprobe_hll[b].estimate();
            tot_distinct += distinct;
            let s = ld(&self.kprobe_s_hits, b);
            let l = ld(&self.kprobe_l_hits, b);
            tot_s += s;
            tot_l += l;
            let ctail = if self.kprobe_canon {
                let cdist = self.kprobe_hll_c[b].estimate();
                tot_cdist += cdist;
                format!(" {:>15} {:>9.1}", commas(cdist as u64), n as f64 / cdist)
            } else {
                String::new()
            };
            println!(
                "    {:>3} {:>15} {:>15} {:>8.2} {:>8.1}% {:>8.1}%{ctail}",
                b + 9,
                commas(n),
                commas(distinct as u64),
                n as f64 / distinct,
                100.0 * s as f64 / n as f64,
                100.0 * l as f64 / n as f64,
            );
        }
        if total > 0 {
            let ctail = if self.kprobe_canon {
                format!(
                    " {:>15} {:>9.1}",
                    commas(tot_cdist as u64),
                    total as f64 / tot_cdist
                )
            } else {
                String::new()
            };
            println!(
                "    all {:>15} {:>15} {:>8.2} {:>8.1}% {:>8.1}%{ctail}",
                commas(total),
                commas(tot_distinct as u64),
                total as f64 / tot_distinct,
                100.0 * tot_s as f64 / total as f64,
                100.0 * tot_l as f64 / total as f64,
            );
        }
    }

    /// Merge this worker's `M_RANK` accumulator ([`RANK_ACC`]) into the shared `rank_*` totals.
    fn drain_rank_local(&self) {
        RANK_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            for pc in 0..MAXPC {
                if a.nodes[pc] == 0 {
                    continue;
                }
                self.rank_nodes[pc].fetch_add(a.nodes[pc], Ordering::Relaxed);
                self.rank_etc[pc].fetch_add(a.etc_cut[pc], Ordering::Relaxed);
                self.rank_etc_probes[pc].fetch_add(a.etc_probes[pc], Ordering::Relaxed);
                self.rank_nocut[pc].fetch_add(a.no_cut[pc], Ordering::Relaxed);
                self.rank_nocut_deg[pc].fetch_add(a.no_cut_deg[pc], Ordering::Relaxed);
                for r in 0..RANK_BUCKETS {
                    if a.rank_dist[pc][r] != 0 {
                        self.rank_dist[pc * RANK_BUCKETS + r]
                            .fetch_add(a.rank_dist[pc][r], Ordering::Relaxed);
                        a.rank_dist[pc][r] = 0;
                    }
                }
                a.nodes[pc] = 0;
                a.etc_cut[pc] = 0;
                a.etc_probes[pc] = 0;
                a.no_cut[pc] = 0;
                a.no_cut_deg[pc] = 0;
            }
        });
    }

    fn drain_rank_all(&self) {
        rayon::broadcast(|_| self.drain_rank_local());
        self.drain_rank_local();
    }

    /// Merge this worker's `M_COLD` accumulator ([`COLD_ACC`]) into the shared per-worker list
    /// ([`cold_workers`](Self::cold_workers)) — kept per-worker (not folded into one aggregate) so
    /// the report can rank workers by probe count and print the giant-root tail's miss% separately.
    /// Cold — called once per worker at drain.
    fn drain_cold_local(&self) {
        COLD_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            let probes: u64 = a.hits.iter().chain(a.misses.iter()).sum();
            if probes == 0 {
                return;
            }
            self.cold_workers.lock().unwrap().push(ColdWorker {
                probes,
                hits: Box::new(a.hits),
                misses: Box::new(a.misses),
            });
            a.hits = [0; MAXPC];
            a.misses = [0; MAXPC];
        });
    }

    fn drain_cold_all(&self) {
        rayon::broadcast(|_| self.drain_cold_local());
        self.drain_cold_local();
    }

    /// Move this worker's captured `M_HITKEY` records into the shared [`hitkey_recs`](Self::hitkey_recs).
    /// Cold — called once per worker at drain.
    fn drain_hitkey_local(&self) {
        HITKEY_ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            if a.recs.is_empty() {
                return;
            }
            let recs = std::mem::take(&mut a.recs);
            a.miss_seen = 0;
            self.hitkey_recs.lock().unwrap().extend(recs);
        });
    }

    fn drain_hitkey_all(&self) {
        rayon::broadcast(|_| self.drain_hitkey_local());
        self.drain_hitkey_local();
    }

    /// Write the drained `M_HITKEY` records to [`hitkey_out`](Self::hitkey_out) as a flat little-endian
    /// binary stream for the offline study. Header: magic `b"QHK1"`, `n` (u32), record count (u64).
    /// Each record: `key` (4×u64) · `avail` (4×u64) · `pc` (u16) · `hit` (u8) · pad (u8) = 68 bytes.
    /// (Avail is a board-square bitset: bit `r*n+c`; the reader rebuilds the conflict graph from it.)
    fn write_hitkey_file(&self, n: u32) {
        use std::io::Write;
        let recs = self.hitkey_recs.lock().unwrap();
        let mut buf: Vec<u8> = Vec::with_capacity(16 + recs.len() * 74);
        buf.extend_from_slice(b"QHK1");
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&(recs.len() as u64).to_le_bytes());
        for r in recs.iter() {
            for w in r.key.0 {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            for w in r.avail.0 {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            buf.extend_from_slice(&r.pc.to_le_bytes());
            buf.push(r.hit as u8);
            buf.push(0); // pad to even
        }
        let hits = recs.iter().filter(|r| r.hit).count();
        match std::fs::File::create(&self.hitkey_out).and_then(|mut f| f.write_all(&buf)) {
            Ok(()) => eprintln!(
                "(M_HITKEY: wrote {} records [{} hits / {} sampled misses] to {})",
                recs.len(),
                hits,
                recs.len() - hits,
                self.hitkey_out,
            ),
            Err(e) => eprintln!("(M_HITKEY: failed to write {}: {e})", self.hitkey_out),
        }
    }

    /// Print the `M_DECPROBE` (`QUEENS_DECPROBE=1`) per-pc connected-component decomposability report.
    /// Gates the nimber-decomposition node-count lever: a getK node whose conflict graph splits into
    /// components ≤8 (or ≤k−1) could be resolved (or shrunk a layer) by a Sprague-Grundy XOR of the
    /// component values instead of the whole-graph getK sweep.
    fn print_dec_report(&self) {
        let ld = |b: &[AtomicU64], i: usize| b[i].load(Ordering::Relaxed);
        let total: u64 = (9..=16).map(|pc| ld(&self.dec_nodes, pc)).sum();
        println!(
            "\n  getK connected-component decomposition (QUEENS_DECPROBE) — {} getK nodes, pc 9..16",
            commas(total)
        );
        println!(
            "    {:>3} {:>15} {:>7} {:>9} {:>11} {:>13}   max-comp-size dist (size:%)",
            "pc", "nodes", "mean#c", "≥2 comp%", "all≤8 %", "all≤(k-1)%"
        );
        for pc in 9..=16usize {
            let n = ld(&self.dec_nodes, pc);
            if n == 0 {
                continue;
            }
            let nf = n as f64;
            let mean_c = ld(&self.dec_ncomp_sum, pc) as f64 / nf;
            let ge2 = 100.0 * ld(&self.dec_ge2, pc) as f64 / nf;
            let le8 = 100.0 * ld(&self.dec_all_le8, pc) as f64 / nf;
            let lekm1 = 100.0 * ld(&self.dec_all_le_km1, pc) as f64 / nf;
            // Top max-component-size buckets (largest first), only the meaningful ones.
            let mut dist: Vec<(usize, u64)> = (1..=16)
                .map(|s| (s, ld(&self.dec_msz, pc * 17 + s)))
                .filter(|&(_, c)| c != 0)
                .collect();
            dist.sort_by(|a, b| b.1.cmp(&a.1));
            let dist_s: String = dist
                .iter()
                .take(5)
                .map(|&(s, c)| format!("{s}:{:.0}%", 100.0 * c as f64 / nf))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "    {pc:>3} {:>15} {mean_c:>7.3} {ge2:>8.1}% {le8:>10.2}% {lekm1:>12.2}%   {dist_s}",
                commas(n)
            );
        }
    }

    /// Print the `M_RANK` (`QUEENS_RANK=1`) per-pc first-losing-child cutoff-rank report. Gates the
    /// move-ordering lever: if nearly every cutoff lands at ETC / rank 0 / rank 1, the ordering is
    /// near-exhausted; a fat rank≥2 tail means a better order would still cut earlier.
    fn print_rank_report(&self) {
        let ld = |b: &[AtomicU64], i: usize| b[i].load(Ordering::Relaxed);
        let total: u64 = (0..MAXPC).map(|pc| ld(&self.rank_nodes, pc)).sum();
        println!(
            "\n  first-losing-child cutoff rank (QUEENS_RANK) — {} expanded OR-nodes",
            commas(total)
        );
        // E = mean children examined / node (the actual ordering cost). `loss` = the *avoidable* part:
        // E − E_perfect, where a perfect order cuts every winning node at its first child (rank 0) and a
        // LOSS (nocut) node still full-scans. The unavoidable ETC/nocut terms cancel, so loss collapses
        // to the mean 0-based cutoff rank (0·r0 + 1·r1 + 2·r2 + Σ_{r≥3} r·n_r). loss/node averages over
        // every node; loss/cut over just the descent-cut nodes (extra children before the losing one);
        // loss_mass = nodes·loss/node = the band's total avoidable child-exams — the prioritization metric.
        println!(
            "    {:>3} {:>15} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>9} {:>10} {:>9} {:>16} {:>15} {:>8}",
            "pc",
            "nodes",
            "ETC%",
            "r0%",
            "r1%",
            "r2%",
            "r≥3%",
            "nocut%",
            "E/node",
            "loss/node",
            "loss/cut",
            "loss_mass",
            "etc_pr",
            "pr/cut",
        );
        // Grand accumulators. The outcome counts (etc/r0/r1/r2/rge3/nocut) partition the nodes —
        // each expanded OR-node lands in exactly one. `e_total`/`examined_ge3`/`nocut_deg` feed E:
        //   E = 0·ETC + 1·r0 + 2·r1 + 3·r2 + avg(r≥3)·r≥3 + avg(degree)·nocut   (children examined)
        // A descent cut at 0-based rank r means children 0..=r were examined ⇒ r+1 children; a LOSS
        // (no-cut) node examines its full degree; an ETC pre-pass cut examines none in the descent.
        let mut g_nodes = 0u64;
        let mut g_etc = 0u64;
        let mut g_r0 = 0u64;
        let mut g_r1 = 0u64;
        let mut g_r2 = 0u64;
        let mut g_rge3 = 0u64;
        let mut g_nocut = 0u64;
        let mut g_nocut_deg = 0u64; // Σ degree over no-cut nodes (the degree·nocut term)
        let mut g_examined_ge3 = 0u64; // Σ (r+1)·count over r≥3 (children examined by r≥3 nodes)
        let mut g_e_total = 0u64; // grand total children examined (the E numerator)
        let mut g_descent_cuts = 0u64; // nodes cut during the descent (r0+r1+r2+…) — the orderable population
        let mut g_loss = 0u64; // Σ 0-based cutoff rank = total avoidable child-exams (the loss mass)
        let mut g_etc_pr = 0u64; // Σ ETC probes issued (Tier-A)
        for pc in 0..MAXPC {
            let n = ld(&self.rank_nodes, pc);
            if n == 0 {
                continue;
            }
            let nf = n as f64;
            let etc = ld(&self.rank_etc, pc);
            let etc_pr = ld(&self.rank_etc_probes, pc);
            let nocut = ld(&self.rank_nocut, pc);
            let nocut_deg = ld(&self.rank_nocut_deg, pc);
            let r0 = ld(&self.rank_dist, pc * RANK_BUCKETS);
            let r1 = ld(&self.rank_dist, pc * RANK_BUCKETS + 1);
            let r2 = ld(&self.rank_dist, pc * RANK_BUCKETS + 2);
            let rge3: u64 = (3..RANK_BUCKETS)
                .map(|r| ld(&self.rank_dist, pc * RANK_BUCKETS + r))
                .sum();
            // Children examined in the descent: Σ (r+1)·count[r] over every rank bucket. (The last
            // bucket caps at RANK_BUCKETS-1, so a rank ≥ that tail slightly under-counts — negligible.)
            let descent_examined: u64 = (0..RANK_BUCKETS)
                .map(|r| (r as u64 + 1) * ld(&self.rank_dist, pc * RANK_BUCKETS + r))
                .sum();
            let examined_ge3: u64 = (3..RANK_BUCKETS)
                .map(|r| (r as u64 + 1) * ld(&self.rank_dist, pc * RANK_BUCKETS + r))
                .sum();
            let e_total = descent_examined + nocut_deg; // ETC contributes 0
                                                        // Descent cuts = the winning nodes a perfect order could shortcut to rank 0; the avoidable
                                                        // loss is every child examined past that first one = descent_examined − descent_cuts = Σ r·n_r.
            let descent_cuts = r0 + r1 + r2 + rge3;
            let loss = descent_examined - descent_cuts;
            g_nodes += n;
            g_etc += etc;
            g_r0 += r0;
            g_r1 += r1;
            g_r2 += r2;
            g_rge3 += rge3;
            g_nocut += nocut;
            g_nocut_deg += nocut_deg;
            g_examined_ge3 += examined_ge3;
            g_e_total += e_total;
            g_descent_cuts += descent_cuts;
            g_loss += loss;
            g_etc_pr += etc_pr;
            let loss_per_cut = if descent_cuts > 0 {
                loss as f64 / descent_cuts as f64
            } else {
                0.0
            };
            // probes-per-cut: ETC probes issued ÷ ETC cuts. High in cold bands = wasted ETC work
            // (the Tier-A gate candidate); near 1 = the ETC earns its probes.
            let pr_per_cut = if etc > 0 {
                etc_pr as f64 / etc as f64
            } else {
                f64::INFINITY
            };
            println!(
                "    {pc:>3} {:>15} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>7.1}% {:>7.1}% {:>9.2} {:>10.3} {:>9.2} {:>16} {:>15} {:>8.1}",
                commas(n),
                100.0 * etc as f64 / nf,
                100.0 * r0 as f64 / nf,
                100.0 * r1 as f64 / nf,
                100.0 * r2 as f64 / nf,
                100.0 * rge3 as f64 / nf,
                100.0 * nocut as f64 / nf,
                e_total as f64 / nf,
                loss as f64 / nf,
                loss_per_cut,
                commas(loss),
                commas(etc_pr),
                pr_per_cut,
            );
        }
        if g_nodes > 0 {
            let gtot = g_nodes as f64;
            let cut_le1 = g_etc + g_r0 + g_r1; // ETC + r0 + r1
            let g_loss_per_cut = if g_descent_cuts > 0 {
                g_loss as f64 / g_descent_cuts as f64
            } else {
                0.0
            };
            let g_pr_per_cut = if g_etc > 0 {
                g_etc_pr as f64 / g_etc as f64
            } else {
                f64::INFINITY
            };
            println!(
                "    ALL {:>15} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>7.1}% {:>7.1}% {:>9.2} {:>10.3} {:>9.2} {:>16} {:>15} {:>8.1}",
                commas(g_nodes),
                100.0 * g_etc as f64 / gtot,
                100.0 * g_r0 as f64 / gtot,
                100.0 * g_r1 as f64 / gtot,
                100.0 * g_r2 as f64 / gtot,
                100.0 * g_rge3 as f64 / gtot,
                100.0 * g_nocut as f64 / gtot,
                g_e_total as f64 / gtot,
                g_loss as f64 / gtot,
                g_loss_per_cut,
                commas(g_loss),
                commas(g_etc_pr),
                g_pr_per_cut,
            );
            println!(
                "    ETC+r0+r1 = {:.1}% of all",
                100.0 * cut_le1 as f64 / gtot
            );
            // E breakdown: the children-examined budget the move ordering directly controls. Lower
            // E/node ⇒ the first losing child is found sooner. avg(r≥3) and avg(degree) are the mean
            // children examined by the r≥3 and no-cut buckets respectively.
            let avg_ge3 = if g_rge3 > 0 {
                g_examined_ge3 as f64 / g_rge3 as f64
            } else {
                0.0
            };
            let avg_deg = if g_nocut > 0 {
                g_nocut_deg as f64 / g_nocut as f64
            } else {
                0.0
            };
            println!(
                "    E = mean children examined / expanded OR-node = {:.3}  (lower = better ordering)",
                g_e_total as f64 / gtot,
            );
            println!(
                "      = [ 0·ETC + 1·r0 + 2·r1 + 3·r2 + {:.2}·(r≥3) + {:.2}·nocut ] / nodes  ⇒  {} children / {} nodes",
                avg_ge3,
                avg_deg,
                commas(g_e_total),
                commas(g_nodes),
            );
            // The avoidable-ordering ceiling: even a perfect first-losing-child oracle cannot remove more
            // than this many child examinations (the nocut full-scans + the ETC/rank-0 cuts are fixed).
            let pct = if g_e_total > 0 {
                100.0 * g_loss as f64 / g_e_total as f64
            } else {
                0.0
            };
            println!(
                "    ordering_loss = {} avoidable child-exams = {:.1}% of all {}  ⇒  max ordering win (loss/node {:.3}, loss/cut {:.2})",
                commas(g_loss),
                pct,
                commas(g_e_total),
                g_loss as f64 / gtot,
                g_loss_per_cut,
            );
            // Rule-A "explored nodes" = every position the search evaluates and does NOT α-β-prune.
            // Each non-root explored node is exactly one resolved child of its parent, so
            //   explored = roots + Σ_parents(children resolved) = roots + g_e_total + g_etc.
            // g_e_total = descent-examined + no-cut degree (children resolved by descent-cut / loss
            // nodes); g_etc = pre-pass (ETC / empty-child) cuts, each resolving ≈1 winning child (a
            // slight under-count: the ETC's pre-cut Some(1) probes are not added — a ≤few-% band).
            // getK / W_K / band / block / tiny children are resolved as single descent iterations
            // (counted once); their internal recursion is a tablebase-style probe and is NOT
            // re-counted — the standard game-solving / EGTB convention. The root fan (≤ a few dozen
            // first moves) is the only un-counted layer; negligible vs the total. Contrast
            // `tt.nodes()` = TT-miss expansions only (leaf evaluations excluded).
            let explored = g_e_total + g_etc;
            let expansions = self.tt.nodes();
            println!(
                "    explored (rule A, leaves incl) = {}   vs expansions (tt.nodes) = {}   ratio = {:.3}",
                commas(explored),
                commas(expansions),
                explored as f64 / expansions.max(1) as f64,
            );
        }
    }

    /// Print the `M_COLD` (`QUEENS_COLD=1`) per-pc entry-probe hit/miss (cold-compute) report. Gates
    /// the memory-side prefetch/pre-warm lever family: a node's entry get either HITs the flat TT (a
    /// transposition served warm — nothing to pre-warm) or MISSes (the node expands = cold compute).
    /// The miss% is the cold fraction the prefetch lever could target. The giant-root tail (the slowest
    /// worker = the most entry probes = ~the whole wall) is isolated from the aggregate, because its
    /// miss% — not the fast-roots-diluted global mix — decides the lever. Verdict: the node-weighted
    /// miss% over the deep recurse-spine bands (pc≥17) vs the ~25–30% kill threshold (below ⇒ the tail
    /// is transposition-saturated/warm ⇒ the memory family is DEAD; above ⇒ there is cold work to warm).
    fn print_cold_report(&self) {
        // Print a per-pc hit/miss table for one (hits, misses) pair; returns (deep-band misses,
        // deep-band probes) over pc≥17 for the verdict. `DEEP_LO` = the recurse-spine floor.
        const DEEP_LO: usize = 17;
        let print_table = |label: &str, hits: &[u64; MAXPC], misses: &[u64; MAXPC]| -> (u64, u64) {
            let total: u64 = hits.iter().chain(misses.iter()).sum();
            println!("  {label} — {} entry probes", commas(total));
            println!(
                "    {:>3} {:>15} {:>8} {:>9}",
                "pc", "probes", "hit%", "miss%(cold)"
            );
            let (mut deep_m, mut deep_p) = (0u64, 0u64);
            for pc in 0..MAXPC {
                let h = hits[pc];
                let m = misses[pc];
                let p = h + m;
                if p == 0 {
                    continue;
                }
                if pc >= DEEP_LO {
                    deep_m += m;
                    deep_p += p;
                }
                println!(
                    "    {pc:>3} {:>15} {:>7.1}% {:>8.1}%",
                    commas(p),
                    100.0 * h as f64 / p as f64,
                    100.0 * m as f64 / p as f64,
                );
            }
            (deep_m, deep_p)
        };

        let mut workers = self.cold_workers.lock().unwrap();
        if workers.is_empty() {
            println!("\n  entry-probe cold fraction (QUEENS_COLD): no probes captured");
            return;
        }
        // Rank workers by probe count desc — rank 0 = the slowest worker = the giant-root tail.
        workers.sort_unstable_by(|a, b| b.probes.cmp(&a.probes));

        // Aggregate across every worker.
        let mut agg_h = [0u64; MAXPC];
        let mut agg_m = [0u64; MAXPC];
        for w in workers.iter() {
            for pc in 0..MAXPC {
                agg_h[pc] += w.hits[pc];
                agg_m[pc] += w.misses[pc];
            }
        }

        println!(
            "\n  entry-probe cold fraction (QUEENS_COLD) — {} workers, top-probe worker = giant-root tail",
            workers.len()
        );

        // The giant-root tail: rank 0 (the slowest worker). Its miss% is the number that gates the lever.
        let tail = &workers[0];
        let (tail_dm, tail_dp) = print_table(
            "TAIL worker (rank 0, by probes) ← giant-root tail",
            &tail.hits,
            &tail.misses,
        );

        // The aggregate (all workers) — diluted by the fast roots, kept for reference.
        let (agg_dm, agg_dp) = print_table("AGGREGATE (all workers)", &agg_h, &agg_m);

        // Verdict: node-weighted deep-band (pc≥DEEP_LO, the recurse spine) miss% vs the kill threshold.
        let tail_pct = if tail_dp > 0 {
            100.0 * tail_dm as f64 / tail_dp as f64
        } else {
            0.0
        };
        let agg_pct = if agg_dp > 0 {
            100.0 * agg_dm as f64 / agg_dp as f64
        } else {
            0.0
        };
        let verdict = if tail_pct < 25.0 {
            "< 25% ⇒ tail is transposition-saturated/WARM — memory prefetch/pre-warm family DEAD"
        } else if tail_pct < 30.0 {
            "25–30% ⇒ borderline — marginal cold work to pre-warm"
        } else {
            "≥ 30% ⇒ tail is substantially COLD — memory prefetch/pre-warm family is LIVE"
        };
        println!(
            "  VERDICT (deep bands pc≥{DEEP_LO}, the recurse spine): tail miss% = {tail_pct:.1}% (aggregate {agg_pct:.1}%) — {verdict}"
        );
    }

    /// Print the A'' Phase-2a offload-sizing report (`QUEENS_SIZE=1`; cold, post-solve). Sizes the
    /// idle-core producer/consumer offload (Approach B) from the recurse-arm probe stream:
    /// (1) per-pc **frontier width** (how many flat-TT probes the consumer issues per band);
    /// (2) the **dedup ceiling** — `1 − distinct/probes` over the whole stream (HLL), the max
    /// duplicate fraction a producer's sort+dedup could remove; (3) the **slot-sorted locality** of
    /// a probe sample — after sorting by TT slot, the fraction of consecutive probes landing in the
    /// same cache line / DRAM-row window (the realized-sorted-stream proxy that the `mlp_bench` 5.7×
    /// assumes). All three feed the gate "a θ exists where sort+SPSC < the sorted-stream saving."
    fn print_size_report(&self) {
        let w: Vec<u64> = self
            .size_w
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect();
        let total: u64 = w.iter().sum();
        if total == 0 {
            eprintln!("(size: no recurse-arm probes captured)");
            return;
        }
        let distinct = self.size_hll.estimate();
        let dedup = 1.0 - (distinct / total as f64).min(1.0);
        eprintln!(
            "(A'' Phase-2a offload sizing — recurse-arm probe stream, {})",
            if self.size_wave {
                "WAVE-on = post-ETC-cut residual (what B offloads on top of the default)"
            } else {
                "WAVE-off = pre-cut upper bound on offloadable work"
            },
        );
        eprintln!(
            "  per-pc frontier width (flat-TT probes the consumer streams), by available-pc:"
        );
        let mut cum = 0u64;
        for (pc, &c) in w.iter().enumerate() {
            if c == 0 {
                continue;
            }
            cum += c;
            eprintln!(
                "    pc={pc:>3}: {c:>15}  ({:6.2}%, cum {:6.2}%)",
                c as f64 / total as f64 * 100.0,
                cum as f64 / total as f64 * 100.0,
            );
        }
        eprintln!(
            "  total probes {total} · distinct (HLL p={SIZE_HLL_P}) {} · dedup ceiling {:.1}% (duplicate probes a sort could collapse)",
            distinct as u64,
            dedup * 100.0,
        );
        // Recency-cache sidecar viability: a per-worker direct-mapped 2^bits cache served what
        // fraction of probes from cache (no DRAM)? The top-probe workers ARE the 2 slow roots
        // (the giant-root tail = ~94% of wall), so their hit rate is the number that decides a
        // sidecar — the global mix is diluted by the fast roots. ≤ the global dedup ceiling.
        {
            let mut rc = self.size_rc.lock().unwrap();
            if !rc.is_empty() {
                rc.sort_unstable_by(|a, b| b.probes.cmp(&a.probes)); // probes desc — slowest roots first
                let bits = self.size_rc_bits;
                let entries = 1u64 << bits;
                let bytes = entries * 4;
                eprintln!(
                    "  recency-cache sidecar sim: 2^{bits} = {entries} u32 tags/worker = {:.1} MB/worker (direct-mapped, per-worker):",
                    bytes as f64 / 1e6,
                );
                let (mut tp, mut th) = (0u64, 0u64);
                for (i, rw) in rc.iter().enumerate() {
                    tp += rw.probes;
                    th += rw.hits;
                    if i < 4 {
                        eprintln!(
                            "    worker rank {i} (by probes): {:>13} probes · {:.1}% hit (DRAM-cut if sidecar'd){}",
                            rw.probes,
                            rw.hits as f64 / rw.probes as f64 * 100.0,
                            if i == 0 { "  ← slowest root / giant-root tail" } else { "" },
                        );
                    }
                }
                // Temporal bands of the slowest root (rank 0): hit rate per RC_WINDOW-probe window
                // — does reuse climb past cold-start into a higher steady state in the late tail?
                if let Some(top) = rc.first() {
                    let curve: String = (0..RC_WINDOWS)
                        .filter(|&wi| top.win_p[wi] != 0)
                        .map(|wi| {
                            format!("{:.0}", top.win_h[wi] as f64 / top.win_p[wi] as f64 * 100.0)
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!(
                        "    slowest-root hit% by {}M-probe window: {curve}",
                        RC_WINDOW / 1_000_000
                    );
                }
                eprintln!(
                    "    ALL workers: {tp} probes · {:.1}% recency-hit (the realistic per-worker DRAM-cut at this cache size)",
                    th as f64 / tp as f64 * 100.0,
                );
                // Where the recency hits land by band (a sidecar would target these).
                let pch: Vec<u64> = self
                    .size_rc_pc
                    .iter()
                    .map(|a| a.load(Ordering::Relaxed))
                    .collect();
                let tot_h: u64 = pch.iter().sum();
                if tot_h != 0 {
                    let mut top: Vec<(usize, u64)> = pch
                        .iter()
                        .enumerate()
                        .filter(|(_, &c)| c != 0)
                        .map(|(pc, &c)| (pc, c))
                        .collect();
                    top.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                    let bands: String = top
                        .iter()
                        .take(8)
                        .map(|(pc, c)| format!("pc{pc}:{:.0}%", *c as f64 / tot_h as f64 * 100.0))
                        .collect::<Vec<_>>()
                        .join(" ");
                    eprintln!("    recency hits by band (top): {bands}");
                }
            }
        }
        // Slot-sorted locality: sort the route sample by TT slot, then measure how many consecutive
        // probes land in the same cache line / DRAM-row window — the row-buffer-hit rate the sorted
        // stream realizes (`mlp_bench`'s 3–5.7× regime) vs the random scatter today (~0% same-row).
        // NOTE this is a *floor*: the sample is capped at `SIZE_SAMPLE_CAP`, so the sorted slots are
        // spread `nslots/sample` apart; a real producer sorts a far larger chunk of the frontier, so
        // its slots pack denser and its same-line/same-row rates only rise toward fully sequential.
        let mut sample = self.size_sample.lock().unwrap();
        if sample.len() < 2 {
            return;
        }
        let nslots = self.tt.capacity().0 as u128;
        let slot = |r: u64| -> u64 { ((r as u128).wrapping_mul(nslots) >> 64) as u64 };
        let mut slots: Vec<u64> = sample.iter().map(|&r| slot(r)).collect();
        sample.clear();
        slots.sort_unstable();
        let n = slots.len();
        let mut same_line = 0u64; // consecutive within 8 slots (a 64-byte cache line)
        let mut same_row = 0u64; // consecutive within 512 slots (~a 4 KB DRAM-row window)
        let mut distinct_slots = 1u64;
        for i in 1..n {
            let d = slots[i] - slots[i - 1];
            if d != 0 {
                distinct_slots += 1;
            }
            if d < 8 {
                same_line += 1;
            }
            if d < 512 {
                same_row += 1;
            }
        }
        let pairs = (n - 1) as f64;
        eprintln!(
            "  slot-sorted sample: {n} probes → {distinct_slots} distinct slots ({:.1}% slot-dedup) · sample spread {:.0} slots/probe over {} TT slots",
            (1.0 - distinct_slots as f64 / n as f64) * 100.0,
            nslots as f64 / n as f64,
            nslots,
        );
        eprintln!(
            "  after sort (sample floor): {:.1}% consecutive same-cache-line (<8 slots) · {:.1}% same-row (<512) — vs ~0% for the random scatter today",
            same_line as f64 / pairs * 100.0,
            same_row as f64 / pairs * 100.0,
        );
    }

    /// A'' Phase-2b-0 (`M_WAVE_B`): reorder `dst` (a copy of a node's filtered moves) so the descent
    /// visits children in **TT-slot order** — the single-thread sorted-frontier wave. Empty / cheap
    /// (`pc ≤ recurse_min`, resolved by dense/band/iso arms with no flat-TT probe) children key to 0 so
    /// the **stable** sort keeps them first in move order (preserving the cheap-cut ordering); recurse
    /// children (`pc > recurse_min`, the flat-TT probes) key to `1 + slot` and follow in slot order.
    /// Verdict-preserving (reorder never changes the OR/AND value). Cold measurement path — the per-move
    /// key build (`child_orient`/`lex_min8`/`d4_bits`/`hash128`) is the M_WAVE gather cost, irrelevant to
    /// the node-count delta this isolates.
    fn sort_moves_by_slot<const DK: u32>(
        &self,
        dst: &mut [u8],
        avail: Bits,
        att: &[[Bits; 8]],
        orient: &[Bits; 8],
    ) {
        let recurse_min = DK.max(self.block_k).max(self.iso_max_avail);
        let nslots = self.tt.capacity().0 as u128;
        let mut keyed = [(0u64, 0u8); MAXV];
        for (k, &sq) in dst.iter().enumerate() {
            let a = att_for8(att, sq);
            let child0 = avail.and_not(a[0]);
            let key = if child0 == Bits::ZERO || child0.popcount() <= recurse_min {
                0
            } else {
                let ckey = d4_bits(lex_min8(&child_orient(orient, a, child0)));
                let (cr, _) = QueensTt::hash128(ckey);
                1 + (((cr as u128).wrapping_mul(nslots) >> 64) as u64)
            };
            keyed[k] = (key, sq);
        }
        let n = dst.len();
        keyed[..n].sort_by(|a, b| a.0.cmp(&b.0)); // stable ⇒ equal-key (cheap) children keep move order
        for (slot, &(_, sq)) in keyed[..n].iter().enumerate() {
            dst[slot] = sq;
        }
    }

    /// Dynamic move ordering (`M_ORD`): reorder `dst` (a copy of a node's filtered moves) by the
    /// **current available-block degree** — `child0.popcount()` ascending, i.e. the move that removes
    /// the most currently-available squares first (the most "forcing" move; a `child0 == 0` move is an
    /// instant win ⇒ sorts first ⇒ earliest α-β cutoff). The production `q.order` is a *static* proxy
    /// (descending **empty-board** attack degree, fixed at build); this recomputes the degree against
    /// the *live* `avail`, so it sharpens deep in the tree where the board has filled. Stable sort ⇒
    /// equal-degree ties keep their `q.order` (the static order is the tiebreak). Verdict-preserving
    /// (reorder never changes the OR/AND value); the node-count delta vs static is the ordering gain.
    fn sort_moves_by_degree<const HIST_TB: bool>(
        &self,
        dst: &mut [u8],
        deg: &mut [u16],
        avail: Bits,
        att: &[[Bits; 8]],
    ) {
        // `n ≤ MAXV` (and ≤ both slice lengths) is an existing invariant — the moves are a subset of
        // the ≤ MAXV board squares — but the optimizer can't see it, so the `.min` makes it explicit.
        // With it, every fixed-array index below (`draw[k]`/`src[k]`, `k < n ≤ MAXV`) and the masked
        // degree (`d & (MAXV-1)`, a no-op since `d = popcount < n ≤ MAXV`, mirroring getK's `& 0x3ff`)
        // carry NO bounds check; the stable-scatter writes use `get_unchecked` under the counting-sort
        // invariant. These checks (`cmp $0x100`/`cmp $0xff`/`cmp %rax,%rbx` in the annotate) were a big
        // slice of this 7.5%-of-run function AND extra branches the frontend (22.6% stalled) must fetch.
        debug_assert_eq!(dst.len(), deg.len());
        let n = dst.len().min(deg.len()).min(MAXV);
        if HIST_TB {
            // M_DHIST: composite key = degree·4 + (3 − history bucket) — ascending degree exactly
            // as the base path, but equal-degree ties break by DESCENDING deep-cutoff history
            // ([`DEEP_HIST`]), bucketed 0..=3 against this node's own max tally (node-local
            // normalization: no global total counter to contend on, and the bucketing adapts as
            // tallies grow). hmax == 0 (no history yet) puts every move in bucket 0 ⇒ the key is a
            // uniform offset of the base key ⇒ identical order. Same branchless stable counting
            // sort over the 4n-key domain; `deg` still receives the REAL child degree (the fused
            // descent reuses it as the child pc).
            let mut hist = [0u32; MAXV];
            let mut hmax = 0u32;
            for k in 0..n {
                let v = DEEP_HIST[dst[k] as usize].load(Ordering::Relaxed);
                hist[k] = v;
                hmax = hmax.max(v);
            }
            let (t1, t2, t3) = (hmax >> 2, hmax >> 1, hmax - (hmax >> 2));
            let mut draw = [0u16; MAXV];
            let mut dpc = [0u16; MAXV];
            for k in 0..n {
                let pc = avail.and_not(att_for8(att, dst[k])[0]).popcount() as u16;
                let b = u16::from(hist[k] > t1) + u16::from(hist[k] > t2) + u16::from(hist[k] > t3);
                dpc[k] = pc;
                // `pc < n ≤ MAXV` ⇒ key < 4n ≤ 4·MAXV; the `& (4·MAXV − 1)` masks below elide the
                // bounds checks exactly like the base path's `& (MAXV − 1)`.
                draw[k] = pc * 4 + (3 - b);
            }
            let mut count = [0u16; MAXV * 4];
            for &d in &draw[..n] {
                count[(d as usize) & (MAXV * 4 - 1)] += 1;
            }
            let mut acc = 0u16;
            for c in count[..4 * n].iter_mut() {
                let cur = *c;
                *c = acc;
                acc += cur;
            }
            let mut src = [0u8; MAXV];
            src[..n].copy_from_slice(&dst[..n]);
            for k in 0..n {
                let d = (draw[k] as usize) & (MAXV * 4 - 1);
                let p = count[d] as usize;
                count[d] += 1;
                // SAFETY: stable counting sort ⇒ `p ∈ [0, n)` (same invariant as the base path),
                // and `n ≤ dst.len()`/`deg.len()`.
                unsafe {
                    *dst.get_unchecked_mut(p) = src[k];
                    *deg.get_unchecked_mut(p) = dpc[k];
                }
            }
            return;
        }
        // Degree key per move (current available-block popcount) in move order. A child drops the
        // placed square plus its attacked squares, so `0 ≤ degree < n` — the key fits the counting
        // sort's `[0, n)` index range. pc ≤ 256 fits u16.
        let mut draw = [0u16; MAXV];
        for k in 0..n {
            draw[k] = avail.and_not(att_for8(att, dst[k])[0]).popcount() as u16;
        }
        // STABLE counting sort by ascending degree. The insertion sort this replaces had a
        // data-dependent `while deg[j-1] > dk` comparison that was the **#1 branch-mispredict site**
        // in the n=16 profile (27.9% of all branch-misses; the search is ~16% of cycles lost to
        // mispredicts). Counting sort has no data-dependent comparison branch — only fixed-trip loops
        // over `[0, n)` — so it trades the mispredict storm for a few predictable passes. Stable
        // (moves scattered in original order at `count[d]++`) ⇒ equal-degree ties keep their q.order,
        // so the searched node set is byte-identical. The descent/gather reuse the sorted `deg`.
        let mut count = [0u16; MAXV];
        for &d in &draw[..n] {
            count[(d as usize) & (MAXV - 1)] += 1;
        }
        // Prefix-sum the per-degree counts into start positions (degrees live in `[0, n)`).
        let mut acc = 0u16;
        for c in count[..n].iter_mut() {
            let cur = *c;
            *c = acc;
            acc += cur;
        }
        // Stable scatter: read moves in original order, place each at its degree's running slot.
        let mut src = [0u8; MAXV];
        src[..n].copy_from_slice(&dst[..n]);
        for k in 0..n {
            let dk = draw[k];
            let d = (dk as usize) & (MAXV - 1);
            let p = count[d] as usize;
            count[d] += 1;
            // SAFETY: stable counting sort ⇒ `p = count[d] ∈ [prefix[d], prefix[d]+cnt[d]) ⊆ [0, n)`,
            // and `n ≤ dst.len()` and `n ≤ deg.len()`, so both scatter writes are in bounds.
            unsafe {
                *dst.get_unchecked_mut(p) = src[k];
                *deg.get_unchecked_mut(p) = dk;
            }
        }
    }

    /// `QUEENS_SKIP18` gate: true iff `pc==18` AND this worker is inside a configured slow deep root
    /// (its whole run). Short-circuits on the run-constant `self.skip18` so the control pays only a
    /// single field-bool test per node; the thread-local is read only on the slow-root pc==18 path.
    #[inline(always)]
    fn skip18_pc18(&self, pc: u32, avail: Bits) -> bool {
        if !self.skip18 || pc >= 64 || !IN_SKIP18_ROOT.with(|f| f.get()) {
            return false;
        }
        if (self.skip18_pcs >> pc) & 1 == 1 {
            return true; // full-skip band (cascade-free {18} by default)
        }
        if self.skip18_frac > 1 && (self.skip18_frac_pcs >> pc) & 1 == 1 {
            // Fractional band: skip 1/frac of nodes by a pre-key hash of the raw avail; keep the rest
            // memoised as cascade/re-probe anchors (pre-key because the canon key is what we skip).
            let h = (avail.0[0]
                ^ avail.0[1].rotate_left(17)
                ^ avail.0[2].rotate_left(31)
                ^ avail.0[3].rotate_left(47))
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            return (h >> 33).is_multiple_of(self.skip18_frac as u64);
        }
        false
    }

    /// Sequential cutoff search (the [`Fused::wins_inc`](super::Fused) twin over the flat TT).
    /// `(route, fp)` are `key`'s precomputed hash halves (hash-carry): each child key is
    /// hashed once at creation and the halves are reused for its prefetch, lookup, and store.
    // Flat args (key + carried hash + move list) are deliberate on this hot recursive path —
    // bundling them into a context struct would add a per-node pointer-chase.
    #[allow(clippy::too_many_arguments)]
    // NB: the former `PROVE_LOSS` const generic was removed — it selected between two
    // behaviourally identical arms (pure OR-search, break on first child-loss; it was
    // vestigial YBWC even/odd-parity bookkeeping that does not affect the sequential node
    // set). Carrying it monomorphised the deep hot recursion into *two* full copies of this
    // body, doubling its L1i footprint in the measured frontend-bound region. One body now.
    fn wins_inc<
        const ORACLE: bool,
        const COUNT: bool,
        const WINDOW: bool,
        const DK: u32,
        const MODE: u8,
    >(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        orient: &[Bits; 8],
        key: Bits,
        route: u64,
        fp: u64,
        pmoves: &[u8],
        nodes: &mut u64,
    ) -> bool {
        let avail = orient[0];
        // The node's own available-popcount, needed *before* the probe in `M_SEG` (it picks
        // the band) and for the `M_HIST` tally. `M_NORMAL` never reads it (popcount compiled
        // out). Same value drives the entry-get, the exit-put, and the histogram bump.
        let node_pc = if MODE == M_NORMAL {
            0
        } else {
            avail.popcount()
        };
        // QUEENS_SKIP18: this node is a pc==18 node whose TT work (key already skipped by the parent,
        // entry probe, exit put) is dropped — pc==18's children are all getK leaves, so re-expansion is
        // a bounded sweep, never a cascade. Gated to the **2 slow deep roots only** (≤2 roots left to
        // finish — the giant-root tail that is ~all of wall); the early fully-parallel all-roots phase
        // is untouched. `node_pc` is 0 on the M_NORMAL control ⇒ skip_tt is false there.
        let skip_tt = self.skip18_pc18(node_pc, avail);
        // `M_DECPROBE`/`M_RANK`/`M_COLD` are measurement twins of `M_ORD_W`: they take the exact same
        // dynamic-ordering + fused-ETC path (so the observed cutoff rank / getK population / entry-probe
        // hit-rate is the production one), adding only a cold tally. These predicates fold the OR away
        // (`MODE` is a const generic ⇒ each is a compile-time constant, DCE'd per monomorphisation; a
        // nested `const` can't capture the outer generic, so a `let` carries it — same codegen).
        let ord_w: bool = MODE == M_ORD_W
            || MODE == M_DECPROBE
            || MODE == M_RANK
            || MODE == M_COLD
            || MODE == M_HITKEY
            || MODE == M_DHIST
            || MODE == M_KPROBE;
        let ord_sort: bool = MODE == M_ORD || MODE == M_RANK_O || ord_w;
        // A'' Phase-2a offload sizing (`M_SIZE`/`M_SIZE_WAVE` only; DCEs to nothing on every other
        // `MODE`, so the control path is byte-identical). Every `wins_inc` entry performs exactly one
        // flat-TT get below, so tapping here records the recurse-arm probe stream the idle-core producer
        // (Approach B) would sort/dedup: per-pc width (frontier size by band), the probed canonical
        // key folded into a global HLL (distinct ⇒ the dedup ceiling), and a bounded route sample for
        // the post-sort row-buffer-locality check. Cold, non-atomic per-worker accumulation. `M_SIZE`
        // measures the WAVE-off stream (pre-ETC-cut upper bound); `M_SIZE_WAVE` also runs the M_WAVE
        // ETC cut below, so it taps the post-cut **residual** stream B actually offloads on top of the
        // default. (The ETC batch probes are not tapped — conservative: those add volume, not less.)
        if MODE == M_SIZE || MODE == M_SIZE_WAVE {
            SIZE_ACC.with(|c| {
                let mut a = c.borrow_mut();
                a.w[node_pc as usize] += 1;
                self.size_hll.add_local(key, &mut a.hll);
                if a.sample.len() < SIZE_SAMPLE_CAP {
                    a.sample.push(route);
                }
                // Recency-cache sidecar-viability sim: a direct-mapped 2^RC_BITS cache of u32 tags.
                // A hit = this key recurred within the worker's recency window ⇒ a cache-resident
                // sidecar of this size serves it without a DRAM probe. Tag from the high route bits
                // (the index uses the low bits, so the tag is ~independent); tag 0 means "empty".
                let bits = self.size_rc_bits as usize;
                if a.rc.is_empty() {
                    a.rc = vec![0u32; 1usize << bits];
                }
                let idx = (route as usize) & ((1usize << bits) - 1);
                let tag = ((route >> 32) as u32) | 1; // never 0 (0 = empty slot)
                let win = ((a.rc_probes / RC_WINDOW) as usize).min(RC_WINDOWS - 1);
                a.rc_probes += 1;
                a.rc_win_p[win] += 1;
                if a.rc[idx] == tag {
                    a.rc_hits += 1;
                    a.rc_per_pc_hits[node_pc as usize] += 1;
                    a.rc_win_h[win] += 1;
                } else {
                    a.rc[idx] = tag;
                }
            });
        }
        // Raw-pointer L0 sidecar base, fetched once for this node (entry probe + the put below).
        let sc_base = if self.sidecar {
            raw_l0_ptr()
        } else {
            std::ptr::null_mut()
        };
        let got = if skip_tt {
            None // QUEENS_SKIP18: never probe a pc==18 node (hit rate ~0.3% ⇒ near-pure waste)
        } else if self.sidecar {
            // SAFETY: `sc_base` is a valid `L0_SIZE` buffer from `raw_l0_ptr` whenever `self.sidecar`.
            match unsafe { raw_l0_get(sc_base, route, fp) } {
                Some(v) => Some(v), // sidecar hit — no TT DRAM probe
                None => {
                    let g = self.mtt_get::<COUNT, MODE>(key, route, fp, node_pc);
                    if let Some(v) = g {
                        unsafe { raw_l0_put(sc_base, route, fp, v) };
                    }
                    g
                }
            }
        } else if MODE == M_PROF {
            // rdtsc-time the flat-TT get and bin the cycles by this node's popcount — the get
            // *is* the random DRAM probe, so its latency by pc is the memory cost distribution.
            let t = rdtsc();
            let g = self.mtt_get::<COUNT, MODE>(key, route, fp, node_pc);
            let dt = rdtsc().wrapping_sub(t);
            PROF_ACC.with(|c| {
                let mut a = c.borrow_mut();
                a.get_cyc[node_pc as usize] += dt;
                a.get_n[node_pc as usize] += 1;
            });
            g
        } else {
            self.mtt_get::<COUNT, MODE>(key, route, fp, node_pc)
        };
        // M_COLD (`QUEENS_COLD=1` only; DCEs to nothing on every other `MODE` ⇒ M_ORD_W byte-identical).
        // Tally this node's entry probe: a HIT (`got.is_some()`) = a transposition/re-probe served warm
        // from the flat TT; a MISS (`got.is_none()`) = the node falls through and EXPANDS = cold compute.
        // The miss% per pc is the cold-compute fraction the prefetch/pre-warm levers target. Binned by
        // `node_pc`; per-worker (non-atomic) so the report can isolate the giant-root tail.
        if MODE == M_COLD {
            COLD_ACC.with(|c| {
                let mut a = c.borrow_mut();
                if got.is_some() {
                    a.hits[node_pc as usize] += 1;
                } else {
                    a.misses[node_pc as usize] += 1;
                }
            });
        }
        // M_HITKEY (`QUEENS_HITKEY=1` only; DCEs to nothing on every other `MODE` ⇒ M_ORD_W
        // byte-identical). Capture the deep-tail (pc≥17) entry probes for offline structural study:
        // every HIT (the rare 0.2% — high-value transpositions) in full, plus a 1/64 sample of MISSes
        // for contrast. `key` is the canonical (D4-merged) identity; `avail` is this node's own
        // square-set, from which the offline study rebuilds the exact conflict graph + features.
        if MODE == M_HITKEY && node_pc >= 17 {
            HITKEY_ACC.with(|c| {
                let mut a = c.borrow_mut();
                let hit = got.is_some();
                let keep = if hit {
                    true
                } else {
                    a.miss_seen += 1;
                    a.miss_seen % HITKEY_MISS_SAMPLE == 0
                };
                if keep {
                    a.recs.push(HitRec {
                        key,
                        avail,
                        pc: node_pc as u16,
                        hit,
                    });
                }
            });
        }
        if let Some(w) = got {
            return w != 0;
        }
        if ORACLE && avail.popcount() <= self.nimber_pc {
            if let Some(nim) = self.try_oracle_nimber(q, avail) {
                let w = nim != 0;
                self.tt_put_h::<COUNT>(key, route, fp, w as u8);
                return w;
            }
        }
        self.tt.bump_local(nodes);
        // Segmented-TT sizing measurement: every node reaching here does exactly one flat-TT
        // put below, so tallying its popcount is the per-pc put histogram (gated to the
        // `M_HIST` monomorphisation — production never executes this).
        if MODE == M_HIST {
            self.hist_bump(node_pc);
        }
        let mut result = false;
        // Compact the available moves once (branchless), then iterate with no per-square
        // availability branch. Children inherit `moves` (the
        // availability-filtered `q.order` subsequence): a child re-filters by *its* avail,
        // and child-avail ⊆ avail, so filtering `moves` vs `pmoves` yields the identical
        // child move list ⇒ byte-identical node set, and the child scans a shorter list.
        let mut buf = [MaybeUninit::<u8>::uninit(); MAXV];
        let moves = filter_moves(&mut buf, pmoves, avail);
        // A'' Phase-2b-0 de-risk (`M_WAVE_B` only; DCEs to nothing on every other `MODE`). Descend the
        // children in **TT-slot order** instead of move order — the single-thread sorted-frontier wave.
        // `sorted` is a reordered copy of `moves`; the standard descent below runs over it. Empty/cheap
        // (no flat-TT probe) children keep their move order first (preserving the cheap-cut ordering),
        // recurse children follow sorted by slot. Verdict-preserving (order never changes the value);
        // the node-count delta vs the move-order baseline is the move-ordering tax the wave pays.
        let mut sorted = [0u8; MAXV];
        // `degbuf[i]` = the available-block degree (child0 popcount) of the i-th *sorted* move, filled
        // by `sort_moves_by_degree` and reused by the M_ORD_W gather + descent below (the sort-fuse:
        // they read the degree instead of recomputing `child0.popcount()`). Only touched on the
        // M_ORD/M_ORD_W sort path; DCEs on every other MODE.
        let mut degbuf = [0u16; MAXV];
        let moves: &[u8] = if MODE == M_WAVE_B {
            let n = moves.len();
            sorted[..n].copy_from_slice(moves);
            self.sort_moves_by_slot::<DK>(&mut sorted[..n], avail, att, orient);
            &sorted[..n]
        } else if ord_sort {
            // Dynamic move ordering: re-sort by current available-block degree (most-forcing first).
            // `M_ORD_W` then runs the M_WAVE ETC body over the degree-sorted moves (ETC + ordering).
            let n = moves.len();
            sorted[..n].copy_from_slice(moves);
            // `MODE` is const ⇒ one arm per monomorphisation (the dead arm DCEs; no per-node branch).
            if MODE == M_DHIST {
                self.sort_moves_by_degree::<true>(&mut sorted[..n], &mut degbuf[..n], avail, att);
            } else {
                self.sort_moves_by_degree::<false>(&mut sorted[..n], &mut degbuf[..n], avail, att);
            }
            &sorted[..n]
        } else {
            moves
        };
        let degs = &degbuf;
        // A'' Phase-1 — ETC (enhanced transposition cutoff) + sorted-batch wave (`M_WAVE` only;
        // DCEs to nothing on every other `MODE`, so the control path is byte-identical). Every node
        // here is an OR node (a *losing* child = a winning move ⇒ this node wins). Before the
        // descent, gather this node's recurse-arm children — the ones the `else` arm below would
        // flat-TT-probe and possibly *expand* (`pc > recurse_min`) — sort them by target slot
        // (monotone in the route hash, so sorting by `cr` = slot order), issue **all** prefetches
        // up front, then probe the batch (full MLP overlap, the sorted wave). A known-LOSS child —
        // or an empty child (the mover-to-be has no reply) — wins this node outright: put + cut,
        // skipping every child expansion the move-ordered descent would have done first. Read-only
        // and verdict-preserving (the cut is only ever an *earlier* return of the same verdict); it
        // changes only which children expand ⇒ gate-safe on iso-dense (no `--distinct`). If no cut,
        // fall through to the normal descent unchanged (it re-resolves every child, these included).
        // A'' Phase-1 — "proper Approach A": a FUSED ETC + sorted-batch descent. The earlier
        // M_WAVE ran a separate pre-pass (rebuild every recurse child's key, sort, probe) and then
        // fell through to the unchanged descent — which *re-derived* `child0` and *rebuilt the same
        // key* and *re-probed* every recurse child a second time at recursion entry. On a no-cut node
        // that double-key-build + double-cold-probe taxed all nodes (the measured +27% cyc/node). This
        // fused body computes each recurse child's key ONCE: the gather stores the full descriptor
        // (`ckey`/`cr`/`cf`) in a stack SoA; the ETC probes the sorted batch (cut on a proven-loss /
        // empty child); and on no cut the descent below **reuses** the stored descriptors instead of
        // rebuilding them. Verdict-preserving (the ETC cut is only ever an earlier return of the same
        // OR-node verdict). DCEs to nothing on every other `MODE` (control byte-identical).
        // `M_SIZE_WAVE` runs this same body (so its tapped stream is the post-cut residual B offloads);
        // `M_L0` runs it with the L0 probe cache layered into `mtt_get`/`mtt_put` (production identical);
        // `M_WAVE_C` runs it with the recurse arm hoisted to the front of the fused-descent cascade;
        // `M_ORD_W` runs it over degree-sorted moves (dynamic ordering + ETC).
        if MODE == M_WAVE
            || MODE == M_SIZE_WAVE
            || MODE == M_L0
            || MODE == M_WAVE_C
            || MODE == M_RANK_WV
            || ord_w
        {
            let recurse_min = DK.max(self.block_k).max(self.iso_max_avail);
            // M_RANK: count this expanded OR-node once at block entry; the resolution site (an ETC
            // pre-descent cut, a descent-rank cut, or the no-cut loop end) records exactly one outcome.
            if mode_rank(MODE) {
                RANK_ACC.with(|c| c.borrow_mut().nodes[node_pc as usize] += 1);
            }
            // SoA descriptor store, recurse children in move order (consumed in order by the descent).
            // `wk` keeps the full child key so the `COUNT=true` HLL path is exact (production
            // `COUNT=false` ignores it — `tt_get_h`/`tt_put_h` use only route/fp).
            let mut wk = [Bits::ZERO; WAVE_CAP];
            let mut wr = [0u64; WAVE_CAP];
            let mut wf = [0u64; WAVE_CAP];
            // ETC probe result per recurse child (QUEENS_ETC_REUSE): 1 = the ETC proved this child a
            // win (⇒ the move fails; the descent skips recursing + re-probing it), 2 = miss/unknown
            // (must recurse — a sibling may have solved it since). 0 (loss) cuts in the ETC loop, so
            // it never reaches the descent. Stays 2 when the ETC didn't run (nw<2) ⇒ recurse, no opt.
            let mut wv = [2u8; WAVE_CAP];
            let mut nw = 0usize;
            for (i, &sq) in moves.iter().enumerate() {
                let a = att_for8(att, sq);
                // Sort-fuse (M_ORD_W): the degree was already computed by `sort_moves_by_degree`, so
                // read it instead of recomputing `child0.popcount()`, and skip `child0` (an `and_not`)
                // entirely for the cheap children the gather doesn't store. Other fused modes (M_WAVE/
                // M_L0/M_WAVE_C/M_SIZE_WAVE) have no `degs`, so they recompute — const-folds per MODE.
                let child0;
                if ord_w {
                    // `i < moves.len() ≤ MAXV`, so `& (MAXV-1)` is a no-op that elides `degs`' bounds
                    // check (degs is `[u16; MAXV]`) — a hot per-move branch the frontend would fetch.
                    let pc = degs[i & (MAXV - 1)] as u32;
                    if pc == 0 {
                        // M_RANK: an empty child found during the gather is a pre-descent (ETC-side) cut.
                        if mode_rank(MODE) {
                            RANK_ACC.with(|c| c.borrow_mut().etc_cut[node_pc as usize] += 1);
                        }
                        if !skip_tt {
                            self.mtt_put::<COUNT, MODE>(key, route, fp, node_pc, 1);
                            // empty child ⇒ node wins
                        }
                        return true;
                    }
                    if !(pc > recurse_min && nw < WAVE_CAP) {
                        continue;
                    }
                    child0 = avail.and_not(a[0]);
                    // QUEENS_SKIP18: a skipped recurse child (full or fractional band) in a slow root is
                    // resolved key-free in the descent (no canon key, no entry probe, no put) — don't
                    // gather it (so no ETC key build / ETC probe for it either). The descent's skip arm
                    // handles it directly. (After child0 so the fractional avail-hash sees it.)
                    if self.skip18_pc18(pc, child0) {
                        continue;
                    }
                } else {
                    child0 = avail.and_not(a[0]);
                    if child0 == Bits::ZERO {
                        if !skip_tt {
                            self.mtt_put::<COUNT, MODE>(key, route, fp, node_pc, 1);
                            // empty child ⇒ node wins
                        }
                        return true;
                    }
                    if !(child0.popcount() > recurse_min && nw < WAVE_CAP) {
                        continue;
                    }
                }
                let child = child_orient(orient, a, child0);
                let ckey = d4_bits(lex_min8(&child));
                let (cr, cf) = QueensTt::hash128(ckey);
                wk[nw] = ckey;
                wr[nw] = cr;
                wf[nw] = cf;
                nw += 1;
            }
            // Gather-time recurse-child prefetch (cheap-first PREFETCH lever, `QUEENS_PFDEEP=1`).
            // Recurse children (pc > recurse_min) sort LAST in the degree-ordered move list, so the
            // descent first scans the cheap getK/band children (real cycles, no TT probe) before
            // reaching the recurse arm. Issuing every recurse child's prefetch HERE — at gather time —
            // overlaps that scan with the ~165-cyc cold DRAM entry probe, far more prefetch-to-use
            // distance than the descent's one-ahead `prefetch_h` (~30 cyc). The win is at `nw == 1`,
            // the deep-tail majority (pc≥17 nodes have too few recurse children for the `nw >= 2` ETC
            // batch below, which is the only prior gather-time prefetch). The first recurse child
            // reached gets the full overlap; later siblings are evicted by the first's subtree and
            // re-prefetched in the descent (free if still warm). Pure cache hint ⇒ byte-identical node
            // set. `pf_deep` off ⇒ this runs only inside the `nw >= 2` ETC batch exactly as before.
            if nw >= 2 {
                for &r in wr.iter().take(nw) {
                    self.tt.prefetch_h(r); // T0: probed immediately by the ETC loop below
                }
            } else if self.pf_deep {
                // Long-distance gather-time prefetch (nw < 2): the lone recurse child sorts LAST in
                // the descent, after the cheap-getK children stream the W8 arena — an L1 (T0) line
                // would be evicted before the recurse arm, so warm it to L2 (T1), which survives the
                // few-hundred-line cheap scan. Turns its ~165-cyc cold DRAM probe into an ~L2 hit.
                for &r in wr.iter().take(nw) {
                    self.tt.prefetch_h_t1(r);
                }
            }
            // ETC pays only with ≥2 recurse children (a single recurse child the descent would probe
            // at entry anyway — no sibling expansions to skip). Skip the probe batch when nw<2 but
            // STILL reuse the one stored descriptor in the descent (kills the rebuild). Probe in GATHER
            // order (no sort): with the fused descent the ETC is the *only* batch probe, the batch is a
            // handful, and the sort cost on the critical path was Fermi'd below its small-batch payoff.
            // The descent below re-prefetches each miss as it expands.
            // QUEENS_ETC_PC: gate the ETC probe batch off below the pc threshold (default 0 ⇒ the
            // `>= 0` test is always true ⇒ byte-identical). The prefetch block above is KEPT, so a
            // gated node's descent still gets warm entry probes — just without the redundant eager probe.
            if nw >= 2 && node_pc >= self.etc_pc_gate {
                // M_RANK (Tier-A tap): tally the ETC probes this batch issues (one per recurse child).
                if mode_rank(MODE) {
                    RANK_ACC.with(|c| c.borrow_mut().etc_probes[node_pc as usize] += nw as u64);
                }
                for j in 0..nw {
                    let v = self.mtt_get::<COUNT, MODE>(wk[j], wr[j], wf[j], 0);
                    if v == Some(0) {
                        // M_RANK: a proven-loss child found by the ETC pre-pass = a pre-descent cut.
                        if mode_rank(MODE) {
                            RANK_ACC.with(|c| c.borrow_mut().etc_cut[node_pc as usize] += 1);
                        }
                        if !skip_tt {
                            self.mtt_put::<COUNT, MODE>(key, route, fp, node_pc, 1);
                            // a losing child ⇒ node wins
                        }
                        return true;
                    }
                    // Remember a proven-win child so the descent can skip recursing + re-probing it.
                    if v == Some(1) {
                        wv[j] = 1;
                    }
                }
            }
            // Fused descent: reuse the stored recurse-child descriptors (no key rebuild). Cheap
            // children (dense/band/block/iso) resolve via their own arms exactly as the control
            // descent. `wi` walks the descriptor store in move order alongside the recurse predicate.
            let mut wi = 0usize;
            // M_RANK: reaching the descent means no ETC pre-pass cut fired (those `return true` above).
            // The descent rank is the move-order index `i` of the first `lost==true` child (`if lost`).
            for (i, &sq) in moves.iter().enumerate() {
                let a = att_for8(att, sq);
                let child0 = avail.and_not(a[0]);
                // Sort-fuse (M_ORD_W): reuse the sorted degree instead of recomputing the popcount.
                let pc = if ord_w {
                    degs[i & (MAXV - 1)] as u32 // mask elides the bounds check (i < moves.len() ≤ MAXV)
                } else {
                    child0.popcount()
                };
                // M_DECPROBE: tap every pc 9..16 getK node's connected-component decomposition. `child0`
                // IS the pc==pc getK node the arm below resolves; build its conflict graph + components.
                if MODE == M_DECPROBE && (9..=16).contains(&pc) {
                    let (ncomp, msz, all_le8, all_le_km1) =
                        decompose_node(child0, att, pc as usize);
                    DEC_ACC.with(|c| {
                        let mut a = c.borrow_mut();
                        let b = pc as usize;
                        a.nodes[b] += 1;
                        a.ncomp_sum[b] += ncomp as u64;
                        a.ge2[b] += (ncomp >= 2) as u64;
                        a.all_le8[b] += all_le8 as u64;
                        a.all_le_km1[b] += all_le_km1 as u64;
                        a.msz_dist[b][(msz as usize).min(16)] += 1;
                    });
                }
                // M_KPROBE: tap every getK entry (a pc 9..=DK child is resolved by the cheap arms
                // below, never recursed). Rebuild its labelled code — the key a code-keyed getK memo
                // would use — fold into the band HLL, and probe the two shared simulated memo tables.
                if MODE == M_KPROBE && pc >= 9 && pc <= DK {
                    let ck = kprobe_code(att, child0, pc);
                    let (route, fp) = QueensTt::hash128(ck);
                    let tag = fp | 1;
                    let sim = |t: &[AtomicU64]| {
                        let i = (route as usize) & (t.len() - 1);
                        let hit = t[i].load(Ordering::Relaxed) == tag;
                        if !hit {
                            t[i].store(tag, Ordering::Relaxed);
                        }
                        hit
                    };
                    let hs = sim(&self.kprobe_sim_s);
                    let hl = sim(&self.kprobe_sim_l);
                    // Level 2 (Tier-C1 go/no-go): the CANONICAL key — decompose the child graph and
                    // combine each component's measurement-exact WL/IR certificate order-independently
                    // (splitmix of (size, canon), wrapping-summed; the tail is 97–100% single-component
                    // per DECPROBE, so the combine rarely fires). Runtime branch is fine: cold tap only.
                    let canon_key = if self.kprobe_canon {
                        let mut acc = 0u64;
                        q.each_comp_canon(child0, |sz, key| {
                            let mut h = key ^ ((sz as u64) << 56);
                            h ^= h >> 30;
                            h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                            h ^= h >> 27;
                            h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
                            h ^= h >> 31;
                            acc = acc.wrapping_add(h);
                        });
                        Some(Bits([acc, 0, 0, pc as u64]))
                    } else {
                        None
                    };
                    KPROBE_ACC.with(|c| {
                        let mut a = c.borrow_mut();
                        let b = (pc - 9) as usize;
                        a.entries[b] += 1;
                        a.sim_s_hits[b] += hs as u64;
                        a.sim_l_hits[b] += hl as u64;
                        self.kprobe_hll[b].add_local(ck, &mut a.hll[b]);
                        if let Some(ckey) = canon_key {
                            self.kprobe_hll_c[b].add_local(ckey, &mut a.hll_c[b]);
                        }
                    });
                }
                if pc == 0 {
                    // M_RANK: an empty child reached in the descent is the first cut at this rank `i`.
                    if mode_rank(MODE) {
                        let r = i.min(RANK_BUCKETS - 1);
                        RANK_ACC.with(|c| c.borrow_mut().rank_dist[node_pc as usize][r] += 1);
                    }
                    result = true;
                    break;
                }
                let lost = if MODE == M_WAVE_C && pc > recurse_min {
                    // Cascade-reorder micro-opt (`M_WAVE_C` only; DCEs for every other MODE ⇒ M_WAVE
                    // byte-identical). A deep-tail recurse child (pc > recurse_min) is the 88%-majority
                    // case; hoisting it to the front skips the ~8 failed `pc==k`/range comparisons of
                    // the cheap-arm cascade below (frontend/L1i pressure on the bottleneck). The body
                    // is identical to the `else` recurse arm — and `recurse_min` IS the cheap/recurse
                    // boundary, so this is behaviour- and node-count-identical, only branch order shifts.
                    let child = child_orient(orient, a, child0);
                    let (ckey, cr, cf) = if wi < nw {
                        let d = (wk[wi], wr[wi], wf[wi]);
                        wi += 1;
                        d
                    } else {
                        let ckey = d4_bits(lex_min8(&child));
                        let (cr, cf) = QueensTt::hash128(ckey);
                        (ckey, cr, cf)
                    };
                    self.tt.prefetch_h(cr);
                    !self.wins_inc::<ORACLE, COUNT, WINDOW, DK, MODE>(
                        q, att, &child, ckey, cr, cf, moves, nodes,
                    )
                } else if DK >= 20 && pc == 20 {
                    !self.w_wide_get::<20>(att, child0)
                } else if DK >= 19 && pc == 19 {
                    !self.w_wide_get::<19>(att, child0)
                } else if DK >= 18 && pc == 18 {
                    !self.w_wide_get::<18>(att, child0)
                } else if DK >= 17 && pc == 17 {
                    !self.w_wide_get::<17>(att, child0)
                } else if DK >= 16 && pc == 16 {
                    !self.w16_get(att, child0)
                } else if DK >= 15 && pc == 15 {
                    !self.w15_get(att, child0)
                } else if DK >= 14 && pc == 14 {
                    !self.w14_get(att, child0)
                } else if DK >= 13 && pc == 13 {
                    !self.w13_get(att, child0)
                } else if DK >= 12 && pc == 12 {
                    !self.w12_get(att, child0)
                } else if DK >= 11 && pc == 11 {
                    !self.w11_get(att, child0)
                } else if DK >= 10 && pc == 10 {
                    !self.w10_get(att, child0)
                } else if DK >= 9 && pc == 9 {
                    !self.w9_get(att, child0)
                } else if WINDOW && !ORACLE && !COUNT && pc == 8 {
                    !self.w8_get(att, child0)
                } else if !ORACLE && pc <= 7 {
                    !self.band_entry::<COUNT>(q, att, child0, pc, nodes)
                } else if !ORACLE && !COUNT && (9..=self.block_k).contains(&pc) {
                    let child = child_orient(orient, a, child0);
                    let ckey = d4_bits(lex_min8(&child));
                    let (cr, cf) = QueensTt::hash128(ckey);
                    self.tt.prefetch_h(cr);
                    !self.block_entry(q, att, child0, ckey, cr, cf, nodes)
                } else if pc <= self.iso_max_avail {
                    let ckey = self.iso_node_key(q, child0, pc);
                    let (cr, cf) = QueensTt::hash128(ckey);
                    self.tt.prefetch_h(cr);
                    !self.wins_tiny::<ORACLE, COUNT, true>(
                        q, att, child0, ckey, cr, cf, moves, nodes,
                    )
                } else if self.skip18_pc18(pc, child0) {
                    // QUEENS_SKIP18: a pc==18 child in a slow root — recurse KEY-FREE. `child_orient`
                    // is still needed (the node expands its getK-leaf children), but no `lex_min8`/
                    // `d4_bits`/`hash128` canon key, no entry probe, no put (the child's `wins_inc`
                    // skips those via `skip_tt`). It was not gathered, so it does not consume `wi`.
                    let child = child_orient(orient, a, child0);
                    !self.wins_inc::<ORACLE, COUNT, WINDOW, DK, MODE>(
                        q,
                        att,
                        &child,
                        Bits::ZERO,
                        0,
                        0,
                        moves,
                        nodes,
                    )
                } else if wi < nw && wv[wi] == 1 {
                    // The ETC pre-pass already proved this recurse child a WIN (`Some(1)` ⇒ opponent
                    // wins ⇒ this move fails ⇒ `lost = false`). Skip the recurse + entry re-probe
                    // entirely — it would re-read the same fixed value (or, if the slot was evicted
                    // since the ETC, re-EXPAND the child). Node-count ≤ baseline (never expands a child
                    // the baseline wouldn't) and correct (a position's win/loss is fixed). The cut grows
                    // with eviction pressure (smaller TT / n=18), so it is always-on. Advance `wi`.
                    wi += 1;
                    false
                } else {
                    // Recurse child: reuse the descriptor built in the gather (no `lex_min8`/
                    // `d4_bits`/`hash128` rebuild). `child_orient` is cheap (7 `and_not`) and the
                    // `[Bits; 8]` is too large to cache per child without a stack blowup, so it is
                    // the only thing recomputed. The recursion's own entry-probe re-reads the slot
                    // (now warm: just probed/prefetched) — that catches any sibling fill mid-node, so
                    // no extra re-expansion vs control. Beyond `WAVE_CAP` recurse children the gather
                    // stopped storing, so rebuild the descriptor (rare: deep-tail fan-out ≤ a handful).
                    // When the descriptor is already in the gather's SoA (the common case), `cr` is
                    // known *before* `child_orient`, so issue the slot prefetch first — it buys
                    // `child_orient`'s ~30 cyc of extra prefetch-to-use distance against the ~165 cyc
                    // DRAM latency of the child's entry probe (the largest single stall in the profile).
                    let (child, ckey, cr, cf) = if wi < nw {
                        let (ck, cr, cf) = (wk[wi], wr[wi], wf[wi]);
                        wi += 1;
                        self.tt.prefetch_h(cr);
                        (child_orient(orient, a, child0), ck, cr, cf)
                    } else {
                        let child = child_orient(orient, a, child0);
                        let ckey = d4_bits(lex_min8(&child));
                        let (cr, cf) = QueensTt::hash128(ckey);
                        self.tt.prefetch_h(cr);
                        (child, ckey, cr, cf)
                    };
                    !self.wins_inc::<ORACLE, COUNT, WINDOW, DK, MODE>(
                        q, att, &child, ckey, cr, cf, moves, nodes,
                    )
                };
                if lost {
                    // M_RANK: first proven-loss child in move order ⇒ this OR-node wins. `i` is its
                    // 0-based descent rank. The last bucket absorbs the rank ≥ RANK_BUCKETS-1 tail.
                    if mode_rank(MODE) {
                        let r = i.min(RANK_BUCKETS - 1);
                        RANK_ACC.with(|c| c.borrow_mut().rank_dist[node_pc as usize][r] += 1);
                    }
                    // M_DHIST: publish the cutoff square to the deep-history tally (the sort's
                    // equal-degree tiebreak). One relaxed add per winning node, spread over the
                    // table's 16 lines.
                    if MODE == M_DHIST {
                        DEEP_HIST[sq as usize].fetch_add(1, Ordering::Relaxed);
                    }
                    result = true;
                    break;
                }
            }
            // M_RANK: the descent completed with no cut ⇒ a LOSS node (full scan, no winning move).
            // A LOSS node examines *every* child, so it contributes its full degree (`moves.len()`,
            // the available-move count) to `E`'s `degree*nocut` term.
            if mode_rank(MODE) && !result {
                RANK_ACC.with(|c| {
                    let mut a = c.borrow_mut();
                    a.no_cut[node_pc as usize] += 1;
                    a.no_cut_deg[node_pc as usize] += moves.len() as u64;
                });
            }
            // The fused loop already ran the descent; skip the shared descent + put below.
            // QUEENS_SKIP18: a pc==18 node in a slow root writes nothing (no put, no sidecar) — its
            // value is recomputed (a bounded getK sweep) on the rare re-visit. Hit rate ~0.3%.
            if !skip_tt {
                if self.sidecar {
                    unsafe { raw_l0_put(sc_base, route, fp, result as u8) };
                }
                self.mtt_put::<COUNT, MODE>(key, route, fp, node_pc, result as u8);
            }
            return result;
        }
        // M_RANK_O / M_RANK_N: the unfused twins descend this plain loop, so the rank tally
        // lives here too. `ri` is bumped only inside `mode_rank` blocks ⇒ the counter and every
        // tally DCE off the non-rank instantiations (byte-identical production loop).
        if mode_rank(MODE) {
            RANK_ACC.with(|c| c.borrow_mut().nodes[node_pc as usize] += 1);
        }
        let mut ri: usize = 0;
        for &sq in moves {
            let a = att_for8(att, sq);
            let child0 = avail.and_not(a[0]);
            if child0 == Bits::ZERO {
                // M_RANK family: an empty child is the first cut, at 0-based descent rank `ri`.
                if mode_rank(MODE) {
                    let r = ri.min(RANK_BUCKETS - 1);
                    RANK_ACC.with(|c| c.borrow_mut().rank_dist[node_pc as usize][r] += 1);
                }
                result = true;
                break;
            }
            // available-popcount is monotone non-increasing down the tree, so once a child
            // enters the iso band it stays there: route it to the orientation-free `wins_tiny`
            // (no `child_orient`, no `lex_min8`) — the deepest, highest-node-count region.
            // Children inherit this node's `moves` as their parent list (a `q.order` subseq).
            let pc = child0.popcount();
            // Dense W_K ceiling: every `9 ≤ pc ≤ DK` child is resolved directly from W0..W8
            // (no flat-TT probe, no subtree expansion). `DK` is `const`, so each arm const-folds
            // away for the instantiations below it — `DK == 8` (iso-flat/iso-window) compiles all
            // three out, identical to before.
            let lost = if DK >= 20 && pc == 20 {
                !self.w_wide_get::<20>(att, child0)
            } else if DK >= 19 && pc == 19 {
                !self.w_wide_get::<19>(att, child0)
            } else if DK >= 18 && pc == 18 {
                !self.w_wide_get::<18>(att, child0)
            } else if DK >= 17 && pc == 17 {
                !self.w_wide_get::<17>(att, child0)
            } else if DK >= 16 && pc == 16 {
                !self.w16_get(att, child0)
            } else if DK >= 15 && pc == 15 {
                !self.w15_get(att, child0)
            } else if DK >= 14 && pc == 14 {
                !self.w14_get(att, child0)
            } else if DK >= 13 && pc == 13 {
                !self.w13_get(att, child0)
            } else if DK >= 12 && pc == 12 {
                !self.w12_get(att, child0)
            } else if DK >= 11 && pc == 11 {
                !self.w11_get(att, child0)
            } else if DK >= 10 && pc == 10 {
                !self.w10_get(att, child0)
            } else if DK >= 9 && pc == 9 {
                !self.w9_get(att, child0)
            } else if WINDOW && !ORACLE && !COUNT && pc == 8 {
                !self.w8_get(att, child0)
            } else if !ORACLE && pc <= 7 {
                !self.band_entry::<COUNT>(q, att, child0, pc, nodes)
            } else if !ORACLE && !COUNT && (9..=self.block_k).contains(&pc) {
                // Dense-block boundary (QUEENS_BLOCK_K > 8): pc in 9..=block_k. Same D4 key as the
                // non-block arm below (boundary-entry merging identical), then a local L1 subtree
                // solve — no per-descendant flat-TT probe. pc==8 stays W8 (iso-window) / D4
                // (iso-flat); default block_k == 8 ⇒ the range is empty ⇒ never taken.
                let child = child_orient(orient, a, child0);
                let ckey = d4_bits(lex_min8(&child));
                let (cr, cf) = QueensTt::hash128(ckey);
                self.tt.prefetch_h(cr);
                !self.block_entry(q, att, child0, ckey, cr, cf, nodes)
            } else if pc <= self.iso_max_avail {
                let ckey = self.iso_node_key(q, child0, pc);
                let (cr, cf) = QueensTt::hash128(ckey);
                self.tt.prefetch_h(cr);
                !self.wins_tiny::<ORACLE, COUNT, true>(q, att, child0, ckey, cr, cf, moves, nodes)
            } else {
                let child = child_orient(orient, a, child0);
                let ckey = d4_bits(lex_min8(&child));
                let (cr, cf) = QueensTt::hash128(ckey);
                self.mtt_prefetch::<MODE>(cr, pc);
                !self.wins_inc::<ORACLE, COUNT, WINDOW, DK, MODE>(
                    q, att, &child, ckey, cr, cf, moves, nodes,
                )
            };
            if lost {
                // M_RANK family: first proven-loss child at 0-based descent rank `ri`.
                if mode_rank(MODE) {
                    let r = ri.min(RANK_BUCKETS - 1);
                    RANK_ACC.with(|c| c.borrow_mut().rank_dist[node_pc as usize][r] += 1);
                }
                result = true;
                break;
            }
            if mode_rank(MODE) {
                ri += 1;
            }
        }
        // M_RANK family: descent completed with no cut ⇒ a LOSS node; it examined every child,
        // so it contributes its full degree to `E`'s `degree·nocut` term.
        if mode_rank(MODE) && !result {
            RANK_ACC.with(|c| {
                let mut a = c.borrow_mut();
                a.no_cut[node_pc as usize] += 1;
                a.no_cut_deg[node_pc as usize] += moves.len() as u64;
            });
        }
        if MODE == M_PROF {
            let t = rdtsc();
            if self.sidecar {
                unsafe { raw_l0_put(sc_base, route, fp, result as u8) };
            }
            self.mtt_put::<COUNT, MODE>(key, route, fp, node_pc, result as u8);
            let dt = rdtsc().wrapping_sub(t);
            PROF_ACC.with(|c| {
                let mut a = c.borrow_mut();
                a.put_cyc[node_pc as usize] += dt;
                a.nodes[node_pc as usize] += 1;
            });
        } else {
            if self.sidecar {
                unsafe { raw_l0_put(sc_base, route, fp, result as u8) };
            }
            self.mtt_put::<COUNT, MODE>(key, route, fp, node_pc, result as u8);
        }
        result
    }

    /// Recursion-unwound twin of [`wins_inc`](Self::wins_inc)'s `M_NORMAL`, non-oracle path
    /// (`QUEENS_ITER=1`). The deep upper-tree OR-search is a strictly-shrinking DFS (each ply drops
    /// ≥1 vertex), so the call stack is replaced by an explicit [`IncFrame`] stack in a reused
    /// thread-local arena. The recurse arm probes each child **inline**: a TT hit resolves with no
    /// frame; a miss *pushes* a frame (one expanded node) and descends. Node completion is a `pop`,
    /// and the verdict cascades up in a loop — a losing child wins its parent (keep unwinding), a
    /// winning child resumes it. Node set, per-expanded-node `bump_local` count, and TT puts are
    /// byte-identical to `wins_inc`. Restricted to `!ORACLE` (the dispatch DCEs it otherwise), so
    /// the oracle/`wins_tiny` arms are absent.
    #[allow(clippy::too_many_arguments)]
    fn wins_inc_iter<
        's,
        const COUNT: bool,
        const WINDOW: bool,
        const DK: u32,
        const ABDADA: bool,
        const STEAL: bool,
    >(
        &'s self,
        scope: &rayon::Scope<'s>,
        q: &'s Queens,
        att: &'s [[Bits; 8]],
        orient: &[Bits; 8],
        key: Bits,
        route: u64,
        fp: u64,
        pmoves: &[u8],
        depth: u32,
        nodes: &mut u64,
    ) -> bool {
        // Entry probe — the only top-of-node probe; deeper nodes are probed inline at the recurse
        // arm. A hit resolves the whole subtree handoff with no expansion, exactly as `wins_inc`.
        // Under ABDADA the slot may carry another worker's in-flight marker (`0xFF`), which
        // `tt_get_h` would misread as a win — so probe tri-state and treat a marker (we own this
        // handoff) like a miss: fall through and expand.
        if ABDADA && !COUNT {
            if let Probe3::Hit(w) = self.tt.get_inflight_hashed(route, fp) {
                return w != 0;
            }
        } else if let Some(w) = self.tt_get_h::<COUNT>(key, route, fp) {
            return w != 0;
        }
        // Count this worker as busy in a deep solve so the work-stealing publish gate
        // (`deep_busy < n_threads`) can read the idle-core count. STEAL-only — a per-handoff atomic,
        // never per node.
        if STEAL {
            self.deep_busy.fetch_add(1, Ordering::Relaxed);
        }
        let won = INC_STACK.with(|cell| {
            let arena = &mut *cell.borrow_mut();
            arena.frames.clear();
            arena.moves.clear();
            // Push the root frame: filter its move list once into the shared move arena.
            {
                let mut scratch = [MaybeUninit::<u8>::uninit(); MAXV];
                let filt = filter_moves(&mut scratch, pmoves, orient[0]);
                let nmoves = filt.len() as u32;
                arena.moves.extend_from_slice(filt);
                arena.frames.push(IncFrame {
                    orient: *orient,
                    key,
                    route,
                    fp,
                    moves_start: 0,
                    nmoves,
                    mi: 0,
                    pass: PASS0,
                    depth,
                    published: 0,
                });
            }
            if ABDADA && !COUNT {
                // Claim this subtree root so other workers probing it defer instead of
                // re-expanding; the completing put (below) overwrites the marker with the verdict.
                self.tt.mark_inflight_hashed(route, fp);
            }
            self.tt.bump_local(nodes); // root expanded (one node, as in `wins_inc` after a miss)
                                       // Work-stealing regime check, windowed: sample the (atomic) `steal_armed` flag into a
                                       // local only every `STEAL_CHECK_EVERY` nodes, so the hot loop never pays a per-node atomic
                                       // load. A short early-phase handoff finishes before the first sample ⇒ `armed` stays false
                                       // ⇒ no publish, no overhead. STEAL-const-gated, so the counter DCEs when stealing is off.
            let mut armed = false;
            let mut since_check: u32 = 0;
            'search: loop {
                if STEAL {
                    since_check += 1;
                    if since_check >= STEAL_CHECK_EVERY {
                        since_check = 0;
                        armed = self.steal_armed.load(Ordering::Relaxed);
                    }
                }
                // Drive the current (top) node's child loop. Leaf children resolve inline; a memo
                // miss on a recurse child pushes a frame and `continue 'search`es to work the top.
                //
                // Hoist the top frame's hot fields into locals for the whole child loop: the
                // ~320 B `IncFrame` (the `orient: [Bits;8]` alone is 256 B) lives in the arena, so
                // re-reading `arena.frames[top]` per child forced the compiler to spill/reload
                // `avail` (= `orient[0]`) and the `mi`/`nmoves`/`moves_start` scalars on every
                // iteration — the `vpandn` operand and the store-forwarded `child0` roundtrip were
                // the perf hot spots. We hold them in registers and only sync back when we suspend
                // (push a child, `continue 'search`) or when the node completes (the cascade reads
                // the frame for its key/route/fp/start, which are unchanged). The full `orient` is
                // copied out once per node-entry (read by `child_orient` in the rare pc≥8 arms).
                // SAFETY (all `get_unchecked` in this loop): `top` indexes the live top frame
                // (`frames` is non-empty here — the root was pushed, and a node is only driven
                // while its frame is on the stack); `moves_start + mi < moves.len()` by
                // construction (the slice `moves[moves_start..moves_start+nmoves]` was appended for
                // this frame and `mi < nmoves` is checked before each read).
                let top = arena.frames.len() - 1;
                let cur = unsafe { *arena.frames.get_unchecked(top) };
                let orient = cur.orient;
                let avail = orient[0];
                let moves_start = cur.moves_start as usize;
                let nmoves = cur.nmoves;
                let mut mi = cur.mi;
                // ABDADA deferral state, restored with the frame (dead `PASS0` when ABDADA is off).
                let mut pass = cur.pass;
                // Work-stealing: even depth ⇒ prove-loss ⇒ children publishable (zero speculation);
                // `published` is how many this frame has already handed to idle cores. Dead unless STEAL.
                let frame_even = cur.depth % 2 == 0;
                let mut published = cur.published;
                let node_won = 'node: loop {
                    if mi >= nmoves {
                        // Pass-0 scan exhausted. With deferrals outstanding, re-scan in PASS1 — the
                        // deferred in-flight children are now resolved hits (their owners finished
                        // while we worked our other children), or we expand any stragglers
                        // ourselves (the progress guarantee). Otherwise every child won ⇒ node LOSES.
                        if ABDADA && pass == PASS0_DEF {
                            pass = PASS1;
                            mi = 0;
                            continue 'node;
                        }
                        break 'node false; // every child won → node LOSES
                    }
                    let sq = unsafe { *arena.moves.get_unchecked(moves_start + mi as usize) };
                    mi += 1; // advance before resolving/descending
                    let a = att_for8(att, sq);
                    let child0 = avail.and_not(a[0]);
                    if child0 == Bits::ZERO {
                        break 'node true; // empty child wins outright
                    }
                    let pc = child0.popcount();
                    let lost = if DK >= 20 && pc == 20 {
                        !self.w_wide_get::<20>(att, child0)
                    } else if DK >= 19 && pc == 19 {
                        !self.w_wide_get::<19>(att, child0)
                    } else if DK >= 18 && pc == 18 {
                        !self.w_wide_get::<18>(att, child0)
                    } else if DK >= 17 && pc == 17 {
                        !self.w_wide_get::<17>(att, child0)
                    } else if DK >= 16 && pc == 16 {
                        !self.w16_get(att, child0)
                    } else if DK >= 15 && pc == 15 {
                        !self.w15_get(att, child0)
                    } else if DK >= 14 && pc == 14 {
                        !self.w14_get(att, child0)
                    } else if DK >= 13 && pc == 13 {
                        !self.w13_get(att, child0)
                    } else if DK >= 12 && pc == 12 {
                        !self.w12_get(att, child0)
                    } else if DK >= 11 && pc == 11 {
                        !self.w11_get(att, child0)
                    } else if DK >= 10 && pc == 10 {
                        !self.w10_get(att, child0)
                    } else if DK >= 9 && pc == 9 {
                        !self.w9_get(att, child0)
                    } else if WINDOW && !COUNT && pc == 8 {
                        !self.w8_get(att, child0)
                    } else if pc <= 7 {
                        !self.band_entry::<COUNT>(q, att, child0, pc, nodes)
                    } else if !COUNT && (9..=self.block_k).contains(&pc) {
                        let child = child_orient(&orient, a, child0);
                        let ckey = d4_bits(lex_min8(&child));
                        let (cr, cf) = QueensTt::hash128(ckey);
                        self.tt.prefetch_h(cr);
                        !self.block_entry(q, att, child0, ckey, cr, cf, nodes)
                    } else {
                        // Recurse arm: build the child key, probe its slot inline. A hit resolves
                        // with no frame; a miss pushes a frame (one expanded node) and descends.
                        // Under ABDADA the probe is tri-state: an in-flight child (another worker
                        // is expanding it) is *deferred* in pass 0 — skipped now, revisited in PASS1
                        // — so this worker stays on its other children instead of racing into a
                        // duplicate (re-)expansion. The deferred child keeps its slot in the move
                        // list (`mi` already advanced past it), so the PASS1 re-scan re-reaches it.
                        let child = child_orient(&orient, a, child0);
                        let ckey = d4_bits(lex_min8(&child));
                        let (cr, cf) = QueensTt::hash128(ckey);
                        self.tt.prefetch_h(cr);
                        let hit = if ABDADA && !COUNT {
                            match self.tt.get_inflight_hashed(cr, cf) {
                                Probe3::Hit(w) => Some(w == 0),
                                // Pass 0 / PASS0_DEF: defer it and keep working other children.
                                Probe3::InFlight if pass != PASS1 => {
                                    pass = PASS0_DEF;
                                    continue 'node;
                                }
                                // PASS1 in-flight: stop waiting — expand it ourselves. If stealing,
                                // this is a published child a stealer hasn't landed yet ⇒ a fallback
                                // re-expansion (the steal that didn't pay off); tally it.
                                Probe3::InFlight => {
                                    if STEAL {
                                        self.steal_fallback.fetch_add(1, Ordering::Relaxed);
                                    }
                                    None
                                }
                                Probe3::Miss => {
                                    // Work-stealing: at an even (prove-loss) frame, while idle cores
                                    // exist, *publish* this child as a `rayon` scope task instead of
                                    // expanding it here — an idle worker steals it, searches it, and
                                    // writes the verdict to the shared TT. Mark it in-flight and defer
                                    // (PASS1 then resolves it as the stealer's hit, not a re-expansion).
                                    if STEAL
                                        && armed
                                        && frame_even
                                        && pass != PASS1
                                        && pc >= self.steal_min_pc
                                        && published < self.steal_width
                                        && (self.active_splits.load(Ordering::Relaxed) as u32)
                                            < self.steal_max
                                        && (published as usize)
                                            < self.n_threads.saturating_sub(
                                                self.deep_busy.load(Ordering::Relaxed),
                                            )
                                    {
                                        self.tt.mark_inflight_hashed(cr, cf);
                                        let cdepth = cur.depth + 1;
                                        self.active_splits.fetch_add(1, Ordering::Relaxed);
                                        scope.spawn(move |s| {
                                            let mut n = 0u64;
                                            let pm = self.order8(q);
                                            // Verdict lands in the shared TT via the completing put.
                                            let _ = self
                                                .wins_inc_iter::<COUNT, WINDOW, DK, ABDADA, STEAL>(
                                                    s, q, att, &child, ckey, cr, cf, pm, cdepth,
                                                    &mut n,
                                                );
                                            self.tt.flush_local_nodes(&mut n);
                                            self.active_splits.fetch_sub(1, Ordering::Relaxed);
                                        });
                                        published += 1;
                                        self.steal_published.fetch_add(1, Ordering::Relaxed);
                                        self.steal_pc_hist[pc as usize]
                                            .fetch_add(1, Ordering::Relaxed);
                                        pass = PASS0_DEF;
                                        continue 'node;
                                    }
                                    None
                                }
                            }
                        } else {
                            // a losing child (value 0) wins this node; `None` ⇒ expand it.
                            self.tt_get_h::<COUNT>(ckey, cr, cf).map(|w| w == 0)
                        };
                        match hit {
                            Some(lost) => lost,
                            None => {
                                if ABDADA && !COUNT {
                                    // Claim before descending so concurrent probers defer this child.
                                    self.tt.mark_inflight_hashed(cr, cf);
                                }
                                self.tt.bump_local(nodes);
                                // Suspend: sync `mi`/`pass` back to the frame, then push the child.
                                // The parent's `moves` slice is its `pmoves`; filter into a scratch
                                // (so the borrow of `arena.moves` ends) then append it.
                                // SAFETY: `top` is still the live top frame (no push/pop since the
                                // hoist above).
                                unsafe {
                                    let f = arena.frames.get_unchecked_mut(top);
                                    f.mi = mi;
                                    f.pass = pass;
                                    f.published = published;
                                }
                                let mut scratch = [MaybeUninit::<u8>::uninit(); MAXV];
                                let filt = filter_moves(
                                    &mut scratch,
                                    // SAFETY: `moves_start..moves_start+nmoves` is this frame's
                                    // appended slice, in bounds of `moves`.
                                    unsafe {
                                        arena.moves.get_unchecked(
                                            moves_start..moves_start + nmoves as usize,
                                        )
                                    },
                                    child[0],
                                );
                                let cstart = arena.moves.len() as u32;
                                let cn = filt.len() as u32;
                                arena.moves.extend_from_slice(filt);
                                arena.frames.push(IncFrame {
                                    orient: child,
                                    key: ckey,
                                    route: cr,
                                    fp: cf,
                                    moves_start: cstart,
                                    nmoves: cn,
                                    mi: 0,
                                    pass: PASS0,
                                    depth: cur.depth + 1,
                                    published: 0,
                                });
                                continue 'search;
                            }
                        }
                    };
                    if lost {
                        break 'node true; // a losing child wins this node
                    }
                    // else: child won → keep trying this node's remaining children
                };
                // The top node resolved as `node_won`. Put it, pop it, cascade the verdict: a LOSS
                // makes the parent WIN (keep unwinding), a WIN resumes the parent's own child loop.
                let mut won = node_won;
                loop {
                    // SAFETY: `frames` is non-empty — we only enter the cascade after driving a
                    // node whose frame is on the stack, and we re-check emptiness after each pop.
                    let top = arena.frames.len() - 1;
                    let f = unsafe { arena.frames.get_unchecked(top) };
                    let fkey = f.key;
                    let froute = f.route;
                    let ffp = f.fp;
                    let fstart = f.moves_start as usize;
                    self.tt_put_h::<COUNT>(fkey, froute, ffp, won as u8);
                    arena.moves.truncate(fstart);
                    arena.frames.pop();
                    if arena.frames.is_empty() {
                        return won; // root resolved
                    }
                    if !won {
                        won = true; // child LOST → parent WINS; record + pop the parent too
                        continue;
                    }
                    continue 'search; // child WON → parent resumes its own child loop
                }
            }
        });
        if STEAL {
            self.deep_busy.fetch_sub(1, Ordering::Relaxed);
        }
        won
    }

    /// Orientation-free tail of [`wins_inc`](Self::wins_inc) for the iso band
    /// (`avail.popcount() ≤ iso_max_avail`). Available-popcount only shrinks down the tree,
    /// so every descendant is in-band too: carry just the `avail` mask (one `and_not` per
    /// move via `att[sq][0]`, no `child_orient`/`lex_min8`) and key by the iso key. Same
    /// keys, same search order, same TT as `wins_inc` ⇒ byte-identical node set; it only
    /// drops the dead 8-orientation bookkeeping in the highest-node-count region.
    #[allow(clippy::too_many_arguments)]
    fn wins_tiny<const ORACLE: bool, const COUNT: bool, const PROVE_LOSS: bool>(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        avail: Bits,
        key: Bits,
        route: u64,
        fp: u64,
        pmoves: &[u8],
        nodes: &mut u64,
    ) -> bool {
        if let Some(w) = self.tt_get_h::<COUNT>(key, route, fp) {
            return w != 0;
        }
        if ORACLE && avail.popcount() <= self.nimber_pc {
            if let Some(nim) = self.try_oracle_nimber(q, avail) {
                let w = nim != 0;
                self.tt_put_h::<COUNT>(key, route, fp, w as u8);
                return w;
            }
        }
        self.tt.bump_local(nodes);
        let mut result = false;
        if PROVE_LOSS {
            let mut buf = [MaybeUninit::<u8>::uninit(); MAXV];
            let moves = filter_moves(&mut buf, pmoves, avail);
            for &sq in moves {
                let child0 = avail.and_not(att08(att, sq));
                if child0 == Bits::ZERO {
                    result = true;
                    break;
                }
                let ckey = self.iso_node_key(q, child0, child0.popcount());
                let (cr, cf) = QueensTt::hash128(ckey);
                self.tt.prefetch_h(cr);
                if !self
                    .wins_tiny::<ORACLE, COUNT, false>(q, att, child0, ckey, cr, cf, moves, nodes)
                {
                    result = true;
                    break;
                }
            }
            self.tt_put_h::<COUNT>(key, route, fp, result as u8);
            return result;
        }
        for &sq in pmoves {
            if !avail_has8(avail, sq) {
                continue;
            }
            let child0 = avail.and_not(att08(att, sq));
            if child0 == Bits::ZERO {
                result = true;
                break;
            }
            let ckey = self.iso_node_key(q, child0, child0.popcount());
            let (cr, cf) = QueensTt::hash128(ckey);
            self.tt.prefetch_h(cr);
            if !self.wins_tiny::<ORACLE, COUNT, true>(q, att, child0, ckey, cr, cf, pmoves, nodes) {
                result = true;
                break;
            }
        }
        self.tt_put_h::<COUNT>(key, route, fp, result as u8);
        result
    }

    /// The carried-adjacency key of an in-band child (`alive` over a [`TinyGraph`]) plus
    /// its precomputed `(route, fp)`. Byte-identical to `iso_node_key`'s tiny-table key
    /// (see [`tiny_key_from_adj`]) but with no board scan / attack-row load.
    #[inline]
    fn graph_key(&self, g: &TinyGraph, alive: u8) -> (Bits, u64, u64) {
        let key = graph_bits(tiny_key_from_adj(&g.adj, alive, self.tiny_canon));
        let (route, fp) = QueensTt::hash128(key);
        (key, route, fp)
    }

    /// Resolve a ≤7 band child `child0` (popcount `pc`): production (`!COUNT`) keys it by the
    /// canon-free labelled index — **no `iso_node_key`, no 16 MB canon-table probe**; the
    /// `--distinct` (`COUNT`) build keeps the canonical flat-TT key so the HLL still sees it.
    #[inline]
    fn band_entry<const COUNT: bool>(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        child0: Bits,
        pc: u32,
        nodes: &mut u64,
    ) -> bool {
        if COUNT {
            let ckey = self.iso_node_key(q, child0, pc);
            let (cr, cf) = QueensTt::hash128(ckey);
            self.tt.prefetch_h(cr);
            self.enter_graph::<true>(q, att, child0, pc, ckey, cr, cf, nodes)
        } else {
            self.enter_graph::<false>(q, att, child0, pc, Bits::ZERO, 0, 0, nodes)
        }
    }

    /// Band entry: a node `child0` has just dropped to `popcount ≤ 7`. Build its
    /// [`TinyGraph`] once — the only place the board is read in the whole iso tail — then
    /// hand the subtree to the orientation-free graph game. Vertices are relabelled
    /// `0..k0` in **q.order** (extracted from `child0`, sorted by
    /// [`order_rank`](Self::order_rank)) so the move order — and the searched node set —
    /// match the old `wins_tiny` tail byte-for-byte. `key/route/fp` are the entry node's
    /// already-computed tiny key (reused so the entry probe isn't recomputed).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn enter_graph<const COUNT: bool>(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        child0: Bits,
        pc: u32,
        key: Bits,
        route: u64,
        fp: u64,
        nodes: &mut u64,
    ) -> bool {
        // Production: the band-entry win/loss lives in the complete ≤7 table keyed by the
        // *labelled* dense index — one direct byte load, no canon-table lookup, no DRAM.
        let tidx = if COUNT {
            0
        } else {
            q.tiny_table_index(child0, pc)
        };
        if COUNT {
            // `--distinct`: keep the band in the flat TT so the HLL counts every position.
            if let Some(w) = self.tt_get_h::<COUNT>(key, route, fp) {
                return w != 0;
            }
        } else if let Some(w) = self.tiny_get(tidx) {
            return w;
        }
        let rank = self.order_rank(q);
        let mut verts = [0u8; MAXV_TINY];
        let mut k0 = 0usize;
        child0.each(|v| {
            let v = v as u8;
            let r = rank[v as usize];
            let mut j = k0;
            while j > 0 && rank[verts[j - 1] as usize] > r {
                verts[j] = verts[j - 1];
                j -= 1;
            }
            verts[j] = v;
            k0 += 1;
        });
        // closed[i] = the local vertices in attack[verts[i]] (self included — the mask is
        // self-blocking); adj[i] drops the self bit for the edge code.
        let mut g = TinyGraph {
            adj: [0; MAXV_TINY],
            closed: [0; MAXV_TINY],
        };
        for i in 0..k0 {
            let row = att08(att, verts[i]);
            let mut c = 0u8;
            for (j, &vj) in verts.iter().enumerate().take(k0) {
                c |= (row.get(vj as u32) as u8) << j;
            }
            g.closed[i] = c;
            g.adj[i] = c & !(1u8 << i);
        }
        let alive = ((1u16 << k0) - 1) as u8;
        if COUNT {
            // `--distinct`: keep descendants in the flat TT so the HLL counts them.
            self.tt.bump_local(nodes);
            return self.expand_graph::<COUNT>(&g, alive, key, route, fp, nodes);
        }
        // Production: solve the whole ≤7 subtree in a thread-private 128-byte stack memo
        // (indexed by the alive bitmask) — pure L1, no flat-TT probe, no DRAM, no
        // cross-CCX coherence — then store the band-entry value in the complete ≤7 table.
        // Descendant transpositions across *different* entries are recomputed (cheap, L1)
        // rather than shared through DRAM.
        let mut memo = [-1i8; 128];
        let won = if self.unroll {
            self.solve_local_iter(&g, alive, &mut memo, nodes)
        } else {
            self.solve_local(&g, alive, &mut memo, nodes)
        };
        self.tiny_put(tidx, won);
        won
    }

    /// Probe the complete ≤7 [`tiny_tt`](Self::tiny_tt) at the labelled dense `idx`
    /// ([`Queens::tiny_table_index`]) — one direct indexed byte load (no canon, no fp).
    #[inline]
    fn tiny_get(&self, idx: usize) -> Option<bool> {
        // SAFETY: `idx` comes from `tiny_table_index`, which returns `< TINY_TABLE_SLOTS`.
        match unsafe { self.tiny_tt.get_unchecked(idx) }.load(Ordering::Relaxed) {
            0 => None,
            v => Some(v == 2),
        }
    }

    /// Store `won` at the labelled dense `idx` in the ≤7 [`tiny_tt`](Self::tiny_tt).
    #[inline]
    fn tiny_put(&self, idx: usize, won: bool) {
        // SAFETY: as `tiny_get` — `idx < TINY_TABLE_SLOTS == tiny_tt.len()`.
        unsafe { self.tiny_tt.get_unchecked(idx) }.store(1 + won as u8, Ordering::Relaxed);
    }

    /// Solve an in-band node against a **local** `memo` (indexed by the alive bitmask over
    /// the entry [`TinyGraph`]). Pure L1/register work: playing vertex `i` leaves
    /// `alive & !closed[i]`; an empty child wins outright, otherwise recurse and cut on the
    /// first child that loses. No board op, no key, no TT — the iso band off the memory path.
    fn solve_local(&self, g: &TinyGraph, alive: u8, memo: &mut [i8; 128], nodes: &mut u64) -> bool {
        let m = memo[alive as usize];
        if m >= 0 {
            return m != 0;
        }
        self.tt.bump_local(nodes);
        let mut result = false;
        let mut rem = alive;
        while rem != 0 {
            let i = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let child = alive & !g.closed[i];
            if child == 0 || !self.solve_local(g, child, memo, nodes) {
                result = true;
                break;
            }
        }
        memo[alive as usize] = result as i8;
        result
    }

    /// Recursion-unwound twin of [`solve_local`] (`QUEENS_UNROLL=1`). The ≤7 OR-search is a
    /// strictly-shrinking DFS (each ply drops ≥1 vertex, so `alive` decreases and depth ≤ 7), so
    /// the call stack is replaced by an explicit `[(alive, rem)]` array. The point of the unwind:
    /// a **memo hit on a child is resolved inline — it never enters a frame**, whereas the
    /// recursive form pays a full `solve_local` call (prologue/epilogue) just to read `memo[child]`
    /// and return. Node completion is a `pop` (a loop iteration), not a function return. The memo
    /// writes and the per-expanded-node `bump_local` count are byte-identical to [`solve_local`].
    fn solve_local_iter(
        &self,
        g: &TinyGraph,
        root: u8,
        memo: &mut [i8; 128],
        nodes: &mut u64,
    ) -> bool {
        // Root memo hit: resolve without expanding (matches `solve_local`'s entry check — no bump).
        // SAFETY: `root ≤ 0x7F` (≤7 vertices) ⇒ `root < 128 == memo.len()`; holds for every `child`
        // below too, since `child = alive & !closed[i] ⊆ alive ⊆ root`.
        let rm = *unsafe { memo.get_unchecked(root as usize) };
        if rm >= 0 {
            return rm != 0;
        }
        // `alive` strictly shrinks down every path ⇒ depth ≤ popcount(root) ≤ 7; an 8-deep stack
        // never overflows. The parent stack holds only *suspended* nodes (those waiting on a
        // memo-miss child); the node currently being worked lives in the `alive`/`rem` registers,
        // never re-loaded from `frames` mid-descent. `rem` = this node's vertices not yet tried.
        let mut frames: [(u8, u8); 8] = [(0, 0); 8];
        let closed = &g.closed;
        self.tt.bump_local(nodes);
        // Current node in registers; the explicit stack `frames[0..sp]` holds its suspended ancestors.
        let mut alive = root;
        let mut rem = root;
        let mut sp = 0usize;
        // The verdict of the node that wins this function (the root). The inner loop falls out of the
        // `'node` labelled block whenever a node resolves, carrying its verdict in `node_won`, then a
        // pop loop hands it to the parent — no per-iteration state-machine flag re-checked on entry.
        'search: loop {
            // Drive this node's child loop to completion in registers. Memo-hit children are resolved
            // inline (a WIN advances the loop, a LOSS wins the node); a memo MISS suspends and descends.
            let node_won = 'node: loop {
                if rem == 0 {
                    break 'node false; // every child won → node LOSES
                }
                let i = rem.trailing_zeros() as usize;
                rem &= rem - 1; // advance before descending/resolving
                                // SAFETY: `i < 8` — `rem ⊆ alive ⊆ root`, `root` has ≤7 vertices over indices 0..7.
                let child = alive & !*unsafe { closed.get_unchecked(i) };
                if child == 0 {
                    break 'node true; // empty child wins outright
                }
                // SAFETY: `child < 128` (see entry note).
                let cm = *unsafe { memo.get_unchecked(child as usize) };
                if cm > 0 {
                    continue 'node; // memo HIT, child WON → keep trying this node's children
                }
                if cm == 0 {
                    break 'node true; // memo HIT, child LOST → node WINS
                }
                // memo MISS — suspend this node, descend into `child` (one expanded node).
                // SAFETY: `sp < 7` — depth ≤ popcount(root)−1 ≤ 6 suspended ancestors fit `frames[8]`.
                *unsafe { frames.get_unchecked_mut(sp) } = (alive, rem);
                sp += 1;
                self.tt.bump_local(nodes);
                alive = child;
                rem = child;
                continue 'search;
            };
            // The node resolved. Record it, then unwind: a child LOSS makes the parent WIN (which is
            // itself a LOSS-from-above only at the function boundary — for the OR parent it's a winning
            // child, so the parent resumes); a child WIN makes the parent resume its own loop. We pop
            // while the just-resolved verdict forces the parent's verdict immediately.
            let mut won = node_won;
            loop {
                // SAFETY: `alive < 128` (see entry note).
                *unsafe { memo.get_unchecked_mut(alive as usize) } = won as i8;
                if sp == 0 {
                    return won; // root resolved
                }
                sp -= 1;
                // SAFETY: `sp < 8` after the decrement.
                let (p_alive, p_rem) = *unsafe { frames.get_unchecked(sp) };
                alive = p_alive;
                rem = p_rem;
                if !won {
                    // child LOST → parent WINS; keep unwinding (the grandparent saw a winning child).
                    won = true;
                    continue;
                }
                // child WON → parent resumes trying its remaining children.
                continue 'search;
            }
        }
    }

    /// Widened [`solve_local`] for a dense block at `8 < pc ≤ block_k`: the win/loss DP over the
    /// `u16` alive mask of a ≤12-vertex graph in a thread-private memo (no flat-TT probe below the
    /// boundary). Same recurrence as `solve_local`, just `u16`. Every visited state bumps the node
    /// counter so the block's descendant recompute is visible in the re-expansion measurement.
    ///
    /// **W8-base** (when `dense8` is `Some` — iso-window): any descendant with `popcount ≤ 8` is
    /// resolved by a single lookup into the *complete, shared* dense W0..W8 tables instead of being
    /// recomputed locally. Since the ≤8 subtree is the bulk of the nodes and the dense tables merge
    /// it across all boundaries (zero recompute), this collapses the block's re-expansion to just the
    /// thin pc 9..block_k shell — the variant that avoids the cross-boundary re-expansion.
    fn solve_block_wide(
        &self,
        closed: &[u16; 13],
        alive: u16,
        memo: &mut [i8],
        nodes: &mut u64,
    ) -> bool {
        let m = memo[alive as usize];
        if m >= 0 {
            return m != 0;
        }
        // W8-base: ≤8 vertices ⇒ one complete-table lookup (shared, no recompute, not a search node).
        let k = alive.count_ones() as usize;
        if k <= 8 {
            if let Some(d8) = &self.dense8 {
                let won = d8.get(k, dense_block_code(closed, alive, k));
                memo[alive as usize] = won as i8;
                return won;
            }
        }
        self.tt.bump_local(nodes);
        let mut result = false;
        let mut rem = alive;
        while rem != 0 {
            let i = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let child = alive & !closed[i];
            if child == 0 || !self.solve_block_wide(closed, child, memo, nodes) {
                result = true;
                break;
            }
        }
        memo[alive as usize] = result as i8;
        result
    }

    /// Dense-block boundary entry (`QUEENS_BLOCK_K > 8`, measurement prototype). A node has just
    /// dropped to `8 < pc ≤ block_k`. Merge the boundary value once via the flat TT under the
    /// **same D4 `key`** the non-block path would use (so boundary-entry merging is identical and
    /// the re-expansion delta is *purely* the descendant recompute), then on a miss solve the whole
    /// subtree in a thread-private `[i8;4096]` L1 memo over the ≤12-vertex graph from `child0` —
    /// no further flat-TT probe below the boundary, exactly as the pc≤7 band's `solve_local` does.
    #[allow(clippy::too_many_arguments)]
    fn block_entry(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        child0: Bits,
        key: Bits,
        route: u64,
        fp: u64,
        nodes: &mut u64,
    ) -> bool {
        if let Some(w) = self.tt_get_h::<false>(key, route, fp) {
            return w != 0;
        }
        // Extract the ≤12 live vertices in q.order rank order (matches the band/non-block move
        // order, so the node set is identical modulo the local-vs-global memo merge).
        let rank = self.order_rank(q);
        let mut verts = [0u8; 13];
        let mut k0 = 0usize;
        child0.each(|v| {
            let v = v as u8;
            let r = rank[v as usize];
            let mut j = k0;
            while j > 0 && rank[verts[j - 1] as usize] > r {
                verts[j] = verts[j - 1];
                j -= 1;
            }
            verts[j] = v;
            k0 += 1;
        });
        // closed[i] = live vertices in attack[verts[i]] (self-blocking), over u16 — as enter_graph.
        let mut closed = [0u16; 13];
        for i in 0..k0 {
            let row = att08(att, verts[i]);
            let mut c = 0u16;
            for (j, &vj) in verts.iter().enumerate().take(k0) {
                c |= (row.get(vj as u32) as u16) << j;
            }
            closed[i] = c;
        }
        let alive = ((1u32 << k0) - 1) as u16;
        let mut memo = [-1i8; 4096];
        let won = self.solve_block_wide(&closed, alive, &mut memo, nodes);
        self.tt_put_h::<false>(key, route, fp, won as u8);
        won
    }

    /// In-band recursion: probe the flat TT, else expand. The descendant twin of
    /// [`enter_graph`](Self::enter_graph) — same graph `g`, only `alive` shrinks.
    #[inline]
    fn wins_graph<const COUNT: bool>(
        &self,
        g: &TinyGraph,
        alive: u8,
        key: Bits,
        route: u64,
        fp: u64,
        nodes: &mut u64,
    ) -> bool {
        if let Some(w) = self.tt_get_h::<COUNT>(key, route, fp) {
            return w != 0;
        }
        self.tt.bump_local(nodes);
        self.expand_graph::<COUNT>(g, alive, key, route, fp, nodes)
    }

    /// Expand an in-band node (TT miss already counted): play each alive vertex in q.order
    /// label order; playing `i` leaves `alive & !closed[i]`. An empty child wins outright
    /// (opponent has no move), otherwise recurse and cut on the first child that loses.
    /// One unified path replaces `wins_tiny`'s prove-win/prove-loss split — the `alive`
    /// bitmask *is* the compacted move list, so there is nothing left to filter.
    #[inline]
    fn expand_graph<const COUNT: bool>(
        &self,
        g: &TinyGraph,
        alive: u8,
        key: Bits,
        route: u64,
        fp: u64,
        nodes: &mut u64,
    ) -> bool {
        // Gather all children and their keys first, issuing every TT prefetch up front, so
        // the probes in the resolve loop below overlap (memory-level parallelism) instead
        // of each stalling on DRAM in turn — the search is TT-latency-bound, and a node's
        // children are independent until the first cutoff. `child == 0` is an immediate win
        // (opponent left no move); it carries a zero key and is resolved in q.order so the
        // cutoff — and the searched node set — stay byte-identical.
        let mut kids: [(u8, Bits, u64, u64); MAXV_TINY] = [(0, Bits::ZERO, 0, 0); MAXV_TINY];
        let mut nk = 0usize;
        let mut rem = alive;
        while rem != 0 {
            let i = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let child = alive & !g.closed[i];
            if child != 0 {
                let (ckey, cr, cf) = self.graph_key(g, child);
                self.tt.prefetch_h(cr);
                kids[nk] = (child, ckey, cr, cf);
            }
            nk += 1;
        }
        let mut result = false;
        for &(child, ckey, cr, cf) in &kids[..nk] {
            let lost = child == 0 || !self.wins_graph::<COUNT>(g, child, ckey, cr, cf, nodes);
            if lost {
                result = true;
                break;
            }
        }
        self.tt_put_h::<COUNT>(key, route, fp, result as u8);
        result
    }

    /// Recursive parity-aware parallel cutoff search (the [`Fused::par_wins_inc`] twin). Even
    /// (prove-a-loss) plies fan all children across rayon (no α-β cutoff to lose ⇒ zero
    /// speculation); odd (prove-a-win) plies stay sequential. Below `par_depth` a node still
    /// splits while large (`> min_avail`, the #20 tail fix), else drops to [`wins_inc`].
    fn par_wins_inc<const ORACLE: bool, const COUNT: bool, const WINDOW: bool, const DK: u32>(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        orient: &[Bits; 8],
        key: Bits,
        depth: u32,
        min_avail: u32,
    ) -> bool {
        let avail = orient[0];
        // Warm-restart phase-1 cooperative deadline (`QUEENS_WARM_RESTART`). `panic = "abort"` rules
        // out unwinding, so instead of aborting we *cooperatively* wind down: once the watchdog sets
        // the flag, every split node returns a sentinel and writes nothing (the `warming` puts below
        // are guarded). In-flight sequential `wins_inc` handoffs still finish — warming the TT with
        // their correct values — but no new parallel work is issued and no incomplete value is ever
        // written, so the warmed entries stay sound. `warming` is false by default (one bool load).
        let warming = self.warm_restart && self.warm_phase.load(Ordering::Relaxed);
        if warming && self.warm_deadline.load(Ordering::Relaxed) {
            return false; // sentinel; phase-1's result is discarded
        }
        // Split nodes are few and shallow (never the deep hot path), so segmentation here is a
        // resolved-once `self.segment` runtime branch in `par_tt_get`/`par_tt_put` rather than
        // another `const` threaded through the recursion. `pc` is this node's popcount (only
        // read in the segmented branch — computed always here, but this is not the hot path).
        let pc = avail.popcount();
        if let Some(w) = self.par_tt_get::<COUNT>(key, pc) {
            return w != 0;
        }
        if depth >= self.par_depth && avail.popcount() <= min_avail {
            let (route, fp) = QueensTt::hash128(key);
            // Hand the sequential subtree the full move order as its parent list; `wins_inc`
            // filters it to `avail` once, then the list shrinks incrementally below.
            let mut nodes = 0;
            // The production-window measurement modes (`QUEENS_PC_HIST` / `QUEENS_TT_SEGMENT`)
            // pick their `MODE` monomorphisation **here, once per subtree handoff** (never per
            // node), so the deep `wins_inc` recursion is fully monomorphised. The `WINDOW &&
            // !ORACLE && !COUNT` guard is const, so `M_HIST`/`M_SEG` are only instantiated for
            // the production-window combo (the guard const-folds to `M_NORMAL` elsewhere, and
            // DCE drops the dead arms — no instantiation blow-up).
            let mode = if WINDOW && !ORACLE && !COUNT {
                // `size` is an explicit cold measurement (like `prof`); it wins over the production
                // `wave` default. `QUEENS_SIZE=1` (`M_SIZE`) measures the WAVE-off probe stream — the
                // full frontier before the ETC cut, the upper bound on what Approach B can offload;
                // `QUEENS_SIZE=2` (`M_SIZE_WAVE`) taps on top of the ETC cut = the post-cut residual.
                if self.decprobe {
                    // Cold getK-decomposition tap on the production M_ORD_W path (explicit measurement,
                    // wins over the `wave`/`ord` defaults like `size`/`prof` do).
                    M_DECPROBE
                } else if self.kprobe {
                    // Cold getK-entry repeat-rate tap on the production M_ORD_W path.
                    M_KPROBE
                } else if self.rank {
                    // Cold first-losing-child rank tap. The base ordering is selected by the same
                    // `ord`/`ord_etc`/`wave` flags production uses, so the rank distribution is
                    // capturable under each ordering variant (the M1 per-variant capture):
                    // default → M_RANK (M_ORD_W twin); QUEENS_ORD=1 → M_RANK_O (M_ORD twin);
                    // QUEENS_ORD=0 → M_RANK_WV (M_WAVE twin); +QUEENS_WAVE=0 → M_RANK_N (M_NORMAL).
                    if self.ord {
                        if self.ord_etc {
                            M_RANK
                        } else {
                            M_RANK_O
                        }
                    } else if self.wave {
                        M_RANK_WV
                    } else {
                        M_RANK_N
                    }
                } else if self.cold {
                    // Cold entry-probe hit/miss tap on the production M_ORD_W path.
                    M_COLD
                } else if self.hitkey {
                    // Deep-tail (pc≥17) key+avail capture on the production M_ORD_W path.
                    M_HITKEY
                } else if self.size {
                    if self.size_wave {
                        M_SIZE_WAVE
                    } else {
                        M_SIZE
                    }
                } else if self.prof {
                    M_PROF
                } else if self.segment {
                    M_SEG
                } else if self.hist {
                    M_HIST
                } else if self.wave_b {
                    M_WAVE_B
                } else if self.l0 {
                    M_L0
                } else if self.wave_c {
                    M_WAVE_C
                } else if self.dhist {
                    // Production-candidate variant: M_ORD_W + the deep cutoff-history tiebreak.
                    M_DHIST
                } else if self.ord {
                    if self.ord_etc {
                        M_ORD_W
                    } else {
                        M_ORD
                    }
                } else if self.wave {
                    M_WAVE
                } else {
                    M_NORMAL
                }
            } else {
                M_NORMAL
            };
            let order8 = self.order8(q);
            let won = match mode {
                M_SIZE => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_SIZE>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_L0 => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_L0>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_WAVE_C => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_WAVE_C>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_ORD => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_ORD>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_ORD_W => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_ORD_W>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_DECPROBE => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_DECPROBE>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_KPROBE => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_KPROBE>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_RANK => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_RANK>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_RANK_O => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_RANK_O>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_RANK_WV => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_RANK_WV>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_RANK_N => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_RANK_N>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_COLD => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_COLD>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_HITKEY => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_HITKEY>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_DHIST => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_DHIST>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_SIZE_WAVE => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_SIZE_WAVE>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_WAVE_B => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_WAVE_B>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_SEG => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_SEG>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_HIST => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_HIST>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_PROF => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_PROF>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                M_WAVE => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_WAVE>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
                // The explicit-stack experiments (`QUEENS_ITER` plain loop, `QUEENS_ABDADA` deferral,
                // `QUEENS_STEAL` work-stealing) only diverge on the plain `M_NORMAL`, non-oracle path;
                // the choice is resolved here, once per subtree handoff (never per node), into a
                // distinct `const (ABDADA, STEAL)` monomorphisation. STEAL implies the ABDADA markers.
                // The handoff is wrapped in a `rayon` scope so STEAL can spawn stolen children onto
                // idle workers (the scope joins them before the handoff returns ⇒ all verdicts in the
                // TT). `!ORACLE` is const, so the arms DCE for the oracle instantiations; STEAL=false
                // runs no spawns ⇒ the scope wrapper is a node-identical inline of the loop.
                _ if !ORACLE && self.steal => {
                    let mut w = false;
                    rayon::in_place_scope(|s| {
                        w = self.wins_inc_iter::<COUNT, WINDOW, DK, true, true>(
                            s, q, att, orient, key, route, fp, order8, depth, &mut nodes,
                        );
                    });
                    w
                }
                _ if !ORACLE && self.abdada => {
                    let mut w = false;
                    rayon::in_place_scope(|s| {
                        w = self.wins_inc_iter::<COUNT, WINDOW, DK, true, false>(
                            s, q, att, orient, key, route, fp, order8, depth, &mut nodes,
                        );
                    });
                    w
                }
                _ if !ORACLE && self.iter_inc => {
                    let mut w = false;
                    rayon::in_place_scope(|s| {
                        w = self.wins_inc_iter::<COUNT, WINDOW, DK, false, false>(
                            s, q, att, orient, key, route, fp, order8, depth, &mut nodes,
                        );
                    });
                    w
                }
                _ => self.wins_inc::<ORACLE, COUNT, WINDOW, DK, M_NORMAL>(
                    q, att, orient, key, route, fp, order8, &mut nodes,
                ),
            };
            self.tt.flush_local_nodes(&mut nodes);
            return won;
        }
        self.tt.bump();
        let mut moves: [u8; MAXV] = [0; MAXV];
        let mut nc = 0usize;
        for &sq in self.order8(q) {
            if !avail_has8(avail, sq) {
                continue;
            }
            if avail.and_not(att08(att, sq)) == Bits::ZERO {
                self.par_tt_put::<COUNT>(key, pc, 1);
                return true;
            }
            moves[nc] = sq;
            nc += 1;
        }
        // QUEENS_PAR_ORD: order the upper-tree children by current child-degree ascending (the deep
        // dynamic key, here extended to par_wins_inc). Few split nodes (avail > min_avail, near-root),
        // so the cached-key sort is cheap; verdict-correct (only cutoff rank / par_iter order changes).
        if self.par_ord {
            moves[..nc].sort_by_cached_key(|&sq| avail.and_not(att08(att, sq)).popcount());
        }
        let kids = &moves[..nc];
        let recurse = |&sq: &u8| {
            let a = att_for8(att, sq);
            let child0 = avail.and_not(a[0]);
            let child = child_orient(orient, a, child0);
            let ckey = self.node_key(q, &child);
            !self.par_wins_inc::<ORACLE, COUNT, WINDOW, DK>(
                q,
                att,
                &child,
                ckey,
                depth + 1,
                min_avail,
            )
        };
        let won = if depth.is_multiple_of(2) {
            kids.par_iter().any(recurse)
        } else if depth == 1 && self.sched && IN_SCHED_ROOT.with(|f| f.get()) {
            // QUEENS_SCHED: the slow solo root (sq 0)'s 2nd-ply moves run sequentially here —
            // capture each move's [enter,exit] wall, node delta, child count, and refutation flag.
            self.sched_loop::<ORACLE, COUNT, WINDOW, DK>(q, att, orient, kids, depth, min_avail)
        } else if self.split && depth == 1 {
            // QUEENS_SPLIT=1: speculatively parallelize the 2nd-ply moves (odd/prove-win, normally
            // sequential `.any()`). Viable here because they're nearly independent (low overlap), so
            // the redundancy that closed deep DFS-parallelism is small; rayon cancels the losers once a
            // refutation wins. Trades speculation on non-refuting moves for finding the refutation in
            // wall-parallel instead of after exploring them all in series. Depth-1 only (the root split).
            kids.par_iter().any(recurse)
        } else if depth == 1 && (self.killer_k > 0 || self.root_timing) {
            // QUEENS_KILLER (or refutation logging under QUEENS_ROOT_TIMING): the once-per-root
            // sequential 2nd-ply loop, with cross-root killer replies optionally front-loaded.
            self.killer_loop::<ORACLE, COUNT, WINDOW, DK>(q, att, orient, kids, depth, min_avail, 0)
        } else if self.killer_deep && self.killer_k > 0 {
            // QUEENS_KILLER_DEEP: extend the killer jumps to the deeper odd (prove-win) plies of
            // the parallel upper tree (depth 3, 5 — one shared table per ply band). Same
            // mechanism: these loops hunt one refuting reply in static order today.
            let t = (((depth - 1) / 2) as usize).min(2);
            self.killer_loop::<ORACLE, COUNT, WINDOW, DK>(q, att, orient, kids, depth, min_avail, t)
        } else {
            kids.iter().any(recurse)
        };
        // Skip the write if the phase-1 deadline fired while computing the children — a child may
        // have returned the `false` sentinel, poisoning `won`. The completed subtrees below already
        // wrote their own correct entries; this incomplete parent simply isn't memoised. (`warming`
        // is false on every production path ⇒ the guard short-circuits and the put always runs.)
        if !(warming && self.warm_deadline.load(Ordering::Relaxed)) {
            self.par_tt_put::<COUNT>(key, pc, won as u8);
        }
        won
    }

    /// The killer-aware once-per-root 2nd-ply loop (`QUEENS_KILLER`, and refutation logging under
    /// `QUEENS_ROOT_TIMING`). Semantically identical to `kids.iter().any(recurse)` — it explores
    /// the same reply set and short-circuits on the first winner — but (a) with `killer_k > 0` it
    /// front-loads up to `killer_k` squares that already refuted *other* roots (highest global
    /// tally first, remaining moves in their existing order), and (b) it publishes this root's
    /// refuting square to [`KILLER_HITS`] and, under root-timing, logs `(root, reply, rank)` so the
    /// cross-root clustering is observable. Cold — runs once per root; the per-call `Vec` and the
    /// O(kids²) stable-prepend are outside every hot path.
    #[cold]
    #[allow(clippy::too_many_arguments)]
    fn killer_loop<const ORACLE: bool, const COUNT: bool, const WINDOW: bool, const DK: u32>(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        orient: &[Bits; 8],
        kids: &[u8],
        depth: u32,
        min_avail: u32,
        table: usize,
    ) -> bool {
        let hits = &KILLER_HITS[table];
        let avail = orient[0];
        // Remaining reply squares in their existing order; each iteration either jumps to the
        // hottest not-yet-tried killer (re-reading the GLOBAL table, so killers published by other
        // roots *mid-loop* — the slow roots run tens of seconds — are picked up) or takes the next
        // move in order. At most `killer_k` killer-jumps total caps the speculation if the shared
        // killers happen not to refute this root. O(kids) scan per pick — cold, once per root.
        let mut remaining: Vec<u8> = kids.to_vec();
        let mut jumps = 0u32;
        let mut rank = 0u32;
        while !remaining.is_empty() {
            let mut pick = 0usize;
            if jumps < self.killer_k {
                let (mut best_h, mut best_i) = (0u32, usize::MAX);
                for (i, &s) in remaining.iter().enumerate() {
                    let h = hits[s as usize].load(Ordering::Relaxed);
                    if h > best_h {
                        (best_h, best_i) = (h, i);
                    }
                }
                if best_i != usize::MAX {
                    pick = best_i;
                    // Only a real reorder counts as a speculative jump.
                    if best_i != 0 {
                        jumps += 1;
                    }
                }
            }
            let sq = remaining.remove(pick);
            rank += 1;
            let a = att_for8(att, sq);
            let child0 = avail.and_not(a[0]);
            let child = child_orient(orient, a, child0);
            let ckey = self.node_key(q, &child);
            let won_kid = !self.par_wins_inc::<ORACLE, COUNT, WINDOW, DK>(
                q,
                att,
                &child,
                ckey,
                depth + 1,
                min_avail,
            );
            if won_kid {
                hits[sq as usize].fetch_add(1, Ordering::Relaxed);
                if self.root_timing && table == 0 {
                    eprintln!(
                        "[killer] root sq {} refuted by sq {} at rank {}/{} (killer jumps {})",
                        CUR_ROOT_SQ.with(|c| c.get()),
                        sq,
                        rank,
                        kids.len(),
                        jumps,
                    );
                }
                return true;
            }
        }
        false
    }

    /// `QUEENS_SCHED` capture of the slow solo root (sq 0)'s 2nd-ply schedule. Mirrors the depth-1
    /// `recurse` closure body exactly (so it is verdict-identical to `kids.iter().any(recurse)`), but
    /// brackets each move with a [enter,exit] wall stamp + cumulative-node snapshot + child count, and
    /// records whether it refuted (the short-circuit). Only sq-0's depth-1 node reaches this (gated by
    /// `depth == 1 && IN_SCHED_ROOT`), and that loop is sequential on one thread, so attribution is exact.
    #[cold]
    fn sched_loop<const ORACLE: bool, const COUNT: bool, const WINDOW: bool, const DK: u32>(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        orient: &[Bits; 8],
        kids: &[u8],
        depth: u32,
        min_avail: u32,
    ) -> bool {
        let avail = orient[0];
        let base = *self.sched_t0.lock().unwrap();
        // `QUEENS_SCHED_FIRST=<sq,..>`: front-load these 2nd-ply squares (stable for the rest) — the
        // oracle/heuristic reorder experiment for sq-0's depth-1 `.any()` cutoff. Tries them first so a
        // refuting move can cut the big non-refuting subtrees. Read once here (this loop runs once).
        let first: Vec<u8> = std::env::var("QUEENS_SCHED_FIRST")
            .ok()
            .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_default();
        let mut ordered: Vec<u8> = Vec::with_capacity(kids.len());
        for &f in &first {
            if kids.contains(&f) {
                ordered.push(f);
            }
        }
        for &k in kids {
            if !first.contains(&k) {
                ordered.push(k);
            }
        }
        for &sq in &ordered {
            let n0 = self.tt.nodes();
            let t_enter = base.elapsed().as_micros() as u64;
            let a = att_for8(att, sq);
            let child0 = avail.and_not(a[0]);
            let child = child_orient(orient, a, child0);
            let ckey = self.node_key(q, &child);
            let won_kid = !self.par_wins_inc::<ORACLE, COUNT, WINDOW, DK>(
                q,
                att,
                &child,
                ckey,
                depth + 1,
                min_avail,
            );
            let t_exit = base.elapsed().as_micros() as u64;
            let n1 = self.tt.nodes();
            self.sched_recs.lock().unwrap().push(SchedRec {
                sq: sq as u32,
                t_enter_us: t_enter,
                t_exit_us: t_exit,
                nodes: n1.saturating_sub(n0),
                child_pc: child0.popcount(),
                won: won_kid,
            });
            if won_kid {
                return true; // `.any()` short-circuit on the refutation
            }
        }
        false
    }

    /// Write the captured sq-0 2nd-ply schedule as JSONL (one record per explored move, in execution
    /// order) to `QUEENS_SCHED_FILE` (default `.perf-analysis/sched.jsonl`). Cold; post-solve.
    fn dump_sched(&self) {
        use std::io::Write;
        let path = std::env::var("QUEENS_SCHED_FILE")
            .unwrap_or_else(|_| ".perf-analysis/sched.jsonl".to_string());
        let recs = self.sched_recs.lock().unwrap();
        let mut out = String::new();
        for (i, r) in recs.iter().enumerate() {
            out.push_str(&format!(
                "{{\"order\":{},\"sq\":{},\"t_enter_us\":{},\"t_exit_us\":{},\"dur_us\":{},\"nodes\":{},\"child_pc\":{},\"won\":{}}}\n",
                i,
                r.sq,
                r.t_enter_us,
                r.t_exit_us,
                r.t_exit_us.saturating_sub(r.t_enter_us),
                r.nodes,
                r.child_pc,
                r.won,
            ));
        }
        match std::fs::File::create(&path).and_then(|mut f| f.write_all(out.as_bytes())) {
            Ok(()) => eprintln!(
                "\x1b[90m(sched: {} sq-0 2nd-ply records → {})\x1b[0m",
                recs.len(),
                path
            ),
            Err(e) => eprintln!("\x1b[31m(sched dump failed: {e})\x1b[0m"),
        }
    }

    /// Resolve one root move: pick the `const DK` / oracle / counting monomorphisation **once**
    /// (per root, never per node) and return whether the *first player* wins via this move (the
    /// `!par_wins_inc` of the responder's win). Shared by the real fan and the warm-restart pass.
    #[inline]
    fn par_root(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        co: &[Bits; 8],
        ckey: Bits,
        min_avail: u32,
    ) -> bool {
        match (self.nimber_oracle, self.counting) {
            (true, true) => {
                !self.par_wins_inc::<true, true, false, 8>(q, att, co, ckey, 1, min_avail)
            }
            (true, false) => {
                !self.par_wins_inc::<true, false, false, 8>(q, att, co, ckey, 1, min_avail)
            }
            (false, true) => {
                !self.par_wins_inc::<false, true, false, 8>(q, att, co, ckey, 1, min_avail)
            }
            (false, false) => match (self.dense8.is_some(), self.dense_k) {
                (true, 20) => {
                    !self.par_wins_inc::<false, false, true, 20>(q, att, co, ckey, 1, min_avail)
                }
                (true, 19) => {
                    !self.par_wins_inc::<false, false, true, 19>(q, att, co, ckey, 1, min_avail)
                }
                (true, 18) => {
                    !self.par_wins_inc::<false, false, true, 18>(q, att, co, ckey, 1, min_avail)
                }
                (true, 17) => {
                    !self.par_wins_inc::<false, false, true, 17>(q, att, co, ckey, 1, min_avail)
                }
                (true, 16) => {
                    !self.par_wins_inc::<false, false, true, 16>(q, att, co, ckey, 1, min_avail)
                }
                (true, 15) => {
                    !self.par_wins_inc::<false, false, true, 15>(q, att, co, ckey, 1, min_avail)
                }
                (true, 14) => {
                    !self.par_wins_inc::<false, false, true, 14>(q, att, co, ckey, 1, min_avail)
                }
                (true, 13) => {
                    !self.par_wins_inc::<false, false, true, 13>(q, att, co, ckey, 1, min_avail)
                }
                (true, 12) => {
                    !self.par_wins_inc::<false, false, true, 12>(q, att, co, ckey, 1, min_avail)
                }
                (true, 11) => {
                    !self.par_wins_inc::<false, false, true, 11>(q, att, co, ckey, 1, min_avail)
                }
                (true, 10) => {
                    !self.par_wins_inc::<false, false, true, 10>(q, att, co, ckey, 1, min_avail)
                }
                (true, 9) => {
                    !self.par_wins_inc::<false, false, true, 9>(q, att, co, ckey, 1, min_avail)
                }
                // iso-window: W8 dense table, no W9+ ceiling.
                (true, _) => {
                    !self.par_wins_inc::<false, false, true, 8>(q, att, co, ckey, 1, min_avail)
                }
                // iso-flat: no dense layer.
                (false, _) => {
                    !self.par_wins_inc::<false, false, false, 8>(q, att, co, ckey, 1, min_avail)
                }
            },
        }
    }
}

impl Solver for IsoFlat {
    fn name(&self) -> &'static str {
        self.name
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        let att = self.att(q);
        let orient = orient_of(q, q.board.and_not(blocked));
        let key = self.node_key(q, &orient);
        let (route, fp) = QueensTt::hash128(key);
        let mut nodes = 0;
        // WINDOW only matters on the production (`!ORACLE && !COUNT`) path, and the dense
        // ceiling `DK` is resolved once here from `dense_k` (the single runtime decision per
        // solve, never per node). The oracle/counting arms fix `WINDOW=false, DK=8`.
        let won = match (self.nimber_oracle, self.counting) {
            (true, true) => self.wins_inc::<true, true, false, 8, M_NORMAL>(
                q,
                att,
                &orient,
                key,
                route,
                fp,
                self.order8(q),
                &mut nodes,
            ),
            (true, false) => self.wins_inc::<true, false, false, 8, M_NORMAL>(
                q,
                att,
                &orient,
                key,
                route,
                fp,
                self.order8(q),
                &mut nodes,
            ),
            (false, true) => self.wins_inc::<false, true, false, 8, M_NORMAL>(
                q,
                att,
                &orient,
                key,
                route,
                fp,
                self.order8(q),
                &mut nodes,
            ),
            (false, false) => match (self.dense8.is_some(), self.dense_k) {
                (true, 20) => self.wins_inc::<false, false, true, 20, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 19) => self.wins_inc::<false, false, true, 19, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 18) => self.wins_inc::<false, false, true, 18, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 17) => self.wins_inc::<false, false, true, 17, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 16) => self.wins_inc::<false, false, true, 16, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 15) => self.wins_inc::<false, false, true, 15, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 14) => self.wins_inc::<false, false, true, 14, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 13) => self.wins_inc::<false, false, true, 13, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 12) => self.wins_inc::<false, false, true, 12, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 11) => self.wins_inc::<false, false, true, 11, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 10) => self.wins_inc::<false, false, true, 10, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                (true, 9) => self.wins_inc::<false, false, true, 9, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                // iso-window: W8 dense table but no W9+ ceiling.
                (true, _) => self.wins_inc::<false, false, true, 8, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
                // iso-flat: no dense layer at all.
                (false, _) => self.wins_inc::<false, false, false, 8, M_NORMAL>(
                    q,
                    att,
                    &orient,
                    key,
                    route,
                    fp,
                    self.order8(q),
                    &mut nodes,
                ),
            },
        };
        self.tt.flush_local_nodes(&mut nodes);
        self.tt.drain_local(); // sequential path: only this thread accumulated
        self.drain_oracle_local();
        self.drain_hist_local();
        self.drain_prof_local();
        won
    }
    fn first_player_wins(&self, q: &Queens) -> bool {
        if q.is_odd() {
            return true; // centre + 180° mirror strategy
        }
        let att = self.att(q);
        let min_avail = min_avail_for(self.par_min_avail, q.n);
        self.eff_min_avail.store(min_avail, Ordering::Relaxed);
        let mut moves = q.distinct_first_moves();
        // QUEENS_ONLY_ROOT=<sq>: MEASUREMENT — solve only this one root (skip the others) so per-move
        // counters are clean of the ~2 concurrent roots. The returned bool is that root's value, NOT the
        // board verdict (this run is for characterization only). Read once.
        let only_root: Option<u32> = std::env::var("QUEENS_ONLY_ROOT")
            .ok()
            .and_then(|s| s.trim().parse().ok());
        self.root_total.store(moves.len() as u64, Ordering::Relaxed);
        self.root_done.store(0, Ordering::Relaxed);
        let root = orient_of(q, q.board);
        let mut pending: Vec<([Bits; 8], Bits)> = Vec::with_capacity(moves.len());
        for &sq in &moves {
            let a = att_for(att, sq);
            let co = child_orient(&root, a, q.board.and_not(a[0]));
            let ckey = self.node_key(q, &co);
            pending.push((co, ckey));
        }
        // QUEENS_FIRST_ROOTS=<sq,..>: schedule these root squares first in the parallel fan (the slow
        // critical-path roots otherwise sit deep in `distinct_first_moves` order, so no worker picks
        // them up until the early roots free cores — their serial tail then runs SOLO at the end
        // [telemetry: the wall-determining root starts ~8.6s late]). Fanning them first overlaps their
        // serial tail with the parallel bulk. Stable (non-listed roots keep order); the elder-brother
        // `pending[0]` (the sequential TT-warm root) stays fixed; verdict-neutral (order ⊥ value). Read
        // once per solve (cold), not per node.
        if std::env::var("QUEENS_REORDER").as_deref() == Ok("1") {
            let fr = std::env::var("QUEENS_FIRST_ROOTS").unwrap_or_default();
            let prio: Vec<u32> = fr
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect();
            // Insert the priority roots at fan-offset `QUEENS_FIRST_AT` (default 0), NOT at the very
            // front: slow-first runs them COLD (no cross-root TT warming ⇒ +4% nodes / +9% wall). A
            // small offset lets the first few fast roots warm the shared TT before the slow roots'
            // deep tail probes it, while still starting them far earlier than their natural position
            // (~28 ⇒ they otherwise run SOLO at the end). The non-priority roots keep their order.
            let at = env_u32("QUEENS_FIRST_AT", 0) as usize;
            if !prio.is_empty() && pending.len() > 2 {
                let rank = |sq: u32| prio.iter().position(|&p| p == sq);
                let tail: Vec<(u32, ([Bits; 8], Bits))> = moves[1..]
                    .iter()
                    .copied()
                    .zip(pending[1..].iter().copied())
                    .collect();
                let mut prio_items: Vec<(u32, ([Bits; 8], Bits))> = tail
                    .iter()
                    .filter(|(sq, _)| rank(*sq).is_some())
                    .copied()
                    .collect();
                prio_items.sort_by_key(|(sq, _)| rank(*sq).unwrap());
                let rest_items: Vec<(u32, ([Bits; 8], Bits))> = tail
                    .iter()
                    .filter(|(sq, _)| rank(*sq).is_none())
                    .copied()
                    .collect();
                let split = at.min(rest_items.len());
                let reordered = rest_items[..split]
                    .iter()
                    .chain(prio_items.iter())
                    .chain(rest_items[split..].iter());
                for (i, (sq, pe)) in reordered.enumerate() {
                    moves[i + 1] = *sq;
                    pending[i + 1] = *pe;
                }
            }
        }
        // Per-root wall intervals (µs since t0): the `QUEENS_ROOT_TIMING=1` diagnostic. Cold and
        // allocated **only when enabled**, so a normal `iso-flat`/`iso-window`/`iso-dense` run is
        // unchanged. The closure stamps `start` on entry and `end` on exit; `t0` is read ≤ 2× per
        // root, never per node. `idx` is unused when timing is off (zero-cost).
        let timing = self.root_timing;
        let (starts, ends): (Vec<AtomicU64>, Vec<AtomicU64>) = if timing {
            (
                (0..pending.len()).map(|_| AtomicU64::new(0)).collect(),
                (0..pending.len()).map(|_| AtomicU64::new(0)).collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let t0 = Instant::now();
        if self.sched {
            // Rebase the schedule clock to the search start and clear any prior run's records.
            *self.sched_t0.lock().unwrap() = Instant::now();
            self.sched_recs.lock().unwrap().clear();
        }
        // Work-stealing watchdog: arm the publish gate only after `steal_delay` seconds, so the early
        // fully-parallel all-roots phase runs untouched and stealing fires only on the dominant-root
        // tail. Reset first (the solver may be reused). Spawned only when stealing is on (else the
        // flag stays false forever and the gate const-folds to a no-op alongside `const STEAL`).
        self.steal_armed.store(false, Ordering::Relaxed);
        if self.steal {
            if self.steal_delay == 0 {
                // Arm immediately (no watchdog race): the always-on regime, for tests / A/B isolation.
                self.steal_armed.store(true, Ordering::Relaxed);
            } else {
                let flag = self.steal_armed.clone();
                let secs = self.steal_delay;
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_secs(secs));
                    flag.store(true, Ordering::Relaxed);
                });
            }
        }
        // Warm-restart self-gates to n≥15 — below that the whole solve finishes inside the 2s warm
        // window, so phase 1 fully resolves it and phase 2 is instant TT hits (a no-op that would
        // only spawn a needless watchdog, e.g. in the test suite). Even n=16/18 is where it earns.
        let do_warm = self.warm_restart && q.n >= 15;
        // Per-root completion flags for the warm-restart pass: phase 1 marks the roots it finished
        // (their values are now in the TT), so phase 2 staggers only the unfinished (slow) ones.
        let warm_done: Vec<AtomicBool> = if do_warm {
            (0..pending.len()).map(|_| AtomicBool::new(false)).collect()
        } else {
            Vec::new()
        };

        // ── PHASE 1 — warm the shared TT ─────────────────────────────────────────────────────────
        // Run the full fan for `warm_secs`; a watchdog then flips `warm_deadline` and every split node
        // *cooperatively* winds down — returning a sentinel and writing nothing (`panic = "abort"` rules
        // out unwinding). In-flight sequential handoffs still finish, warming the TT with their correct
        // values; only completed subtrees are memoised, so the warmed entries stay sound. Result discarded.
        if do_warm {
            self.warm_phase.store(true, Ordering::Relaxed);
            self.warm_deadline.store(false, Ordering::Relaxed);
            let flag = self.warm_deadline.clone();
            let secs = self.warm_secs;
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(secs));
                flag.store(true, Ordering::Relaxed);
            });
            pending.par_iter().enumerate().for_each(|(i, (co, ckey))| {
                let _ = self.par_root(q, att, co, *ckey, min_avail);
                // A root that returned before the deadline genuinely finished (its value is in the
                // TT); one that returned after only wound down via the sentinel — leave it "slow" so
                // phase 2 restarts it (a real-but-late finisher just replays as fast TT hits).
                if !self.warm_deadline.load(Ordering::Relaxed) {
                    warm_done[i].store(true, Ordering::Relaxed);
                }
            });
            self.warm_phase.store(false, Ordering::Relaxed);
            self.root_done.store(0, Ordering::Relaxed);
        }

        // ── PHASE 2 — the real run over the (now-warm) TT ────────────────────────────────────────
        let resolve = |idx: usize, co: &[Bits; 8], ckey: Bits| {
            // Stagger the slow (phase-1-unfinished) roots so each warms the shared region before the
            // next hits it, instead of racing. Fast roots (finished in phase 1) are instant TT hits
            // and never sleep. `rank` = how many earlier slow roots precede this one.
            if do_warm && self.warm_stagger_ms > 0 && !warm_done[idx].load(Ordering::Relaxed) {
                // Rank among the slow (phase-1-unfinished) roots, **capped** so the cumulative beat
                // stays "a bit" (≤ 4 beats) — without the cap, a short warm leaves ~all roots slow
                // and `rank × stagger` would idle the box for tens of seconds. Beat = `stagger_ms`.
                let rank = warm_done[..idx]
                    .iter()
                    .filter(|d| !d.load(Ordering::Relaxed))
                    .count()
                    .min(4) as u64;
                if rank > 0 {
                    std::thread::sleep(Duration::from_millis(self.warm_stagger_ms * rank));
                }
            }
            if timing {
                starts[idx].store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // QUEENS_SKIP18: arm this worker's pc==18-skip flag for the whole of this root iff it is a
            // configured slow root (or, with no list, all roots). Set here — on the worker that runs the
            // root — so the entire subtree sees it (steal-off ⇒ no mid-root migration). Empty list = all.
            IN_SKIP18_ROOT.with(|f| {
                f.set(
                    self.skip18
                        && (self.skip18_squares.is_empty()
                            || self.skip18_squares.contains(&(moves[idx] as u8))),
                )
            });
            // QUEENS_SCHED: arm the 2nd-ply schedule capture for the slow solo root (sq 0), or for
            // whichever single root QUEENS_ONLY_ROOT isolates (so its depth-1 schedule is captured clean).
            IN_SCHED_ROOT
                .with(|f| f.set(self.sched && (moves[idx] == 0 || only_root == Some(moves[idx]))));
            // Killer-loop attribution: which root this worker is resolving (steal-off ⇒ stable).
            CUR_ROOT_SQ.with(|c| c.set(moves[idx]));
            // The dense ceiling `dense_k` (and the WINDOW flag) is read **once here**, per root, to
            // select the `const DK` monomorphisation — never in the deep `wins_inc` loop or a probe.
            let wins = self.par_root(q, att, co, ckey, min_avail);
            self.root_done.fetch_add(1, Ordering::Relaxed);
            if timing {
                ends[idx].store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            wins
        };
        let won = if let Some(only) = only_root {
            // MEASUREMENT (QUEENS_ONLY_ROOT): run just this root so per-move counters are clean of the
            // other concurrent roots. The result is this root's value, not the board verdict.
            let idx = moves.iter().position(|&m| m == only).unwrap_or(0);
            resolve(idx, &pending[idx].0, pending[idx].1)
        } else if std::env::var("QUEENS_NO_ELDER").as_deref() == Ok("1") {
            // QUEENS_NO_ELDER: fan ALL roots at once (no sequential elder-brother warm phase).
            // Experiment for the killer regime, where the slow roots dominate the wall and the
            // elder's ~1.3s TT warm may no longer pay. Read once per solve (cold).
            pending
                .par_iter()
                .enumerate()
                .any(|(i, (co, ckey))| resolve(i, co, *ckey))
        } else {
            let (first, rest) = pending.split_first().unwrap();
            // Root 0 is the sequential elder-brother (warms the TT); roots 1.. fan via rayon.
            resolve(0, &first.0, first.1)
                || rest
                    .par_iter()
                    .enumerate()
                    .any(|(i, (co, ckey))| resolve(i + 1, co, *ckey))
        };
        if timing {
            print_root_timing(q.n, &moves, &starts, &ends);
        }
        if self.sched {
            self.dump_sched();
        }
        self.tt.drain_all(); // fold every worker's tail tally into the shared totals
        self.drain_oracle_all();
        self.drain_hist_all();
        self.drain_prof_all();
        if self.size {
            self.drain_size_all();
            self.print_size_report();
        }
        if self.decprobe {
            self.drain_dec_all();
            self.print_dec_report();
        }
        if self.kprobe {
            self.drain_kprobe_all();
            self.print_kprobe_report();
        }
        if self.rank {
            self.drain_rank_all();
            self.print_rank_report();
        }
        if self.cold {
            self.drain_cold_all();
            self.print_cold_report();
        }
        if self.hitkey {
            self.drain_hitkey_all();
            self.write_hitkey_file(q.n);
        }
        if self.steal {
            // Work-stealing diagnostics (TTY text; the same data is in the `--to-file` JSON via
            // `steal_report`): how many subtrees were split off, their available-popcount
            // distribution, and how many publishes fell back to a PASS1 re-expansion.
            let r = self.build_steal_report();
            let pct = |x: u64| {
                if r.published > 0 {
                    100.0 * x as f64 / r.published as f64
                } else {
                    0.0
                }
            };
            eprintln!(
                "(work-stealing: split off {} subtrees · avail-pc {}..{} mean {:.1} · {} fallback \
                 re-expansions [{:.1}% of splits] · width {} min-pc {} max {} delay {}s)",
                r.published,
                r.pc_lo,
                r.pc_hi,
                r.pc_mean,
                r.fallback,
                pct(r.fallback),
                r.width,
                r.min_pc,
                r.max,
                r.delay,
            );
            let dist: String = r
                .pc_hist
                .iter()
                .map(|&(pc, c)| format!("{pc}:{c}"))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("(work-stealing split-pc histogram: {dist})");
        }
        won
    }
    fn nodes(&self) -> u64 {
        self.tt.nodes()
    }
    fn cap_bytes(&self) -> u64 {
        self.tt.capacity().1
    }
    fn report(&self) -> Option<CountReport> {
        self.tt.report()
    }
    fn steal_report(&self) -> Option<StealReport> {
        self.steal.then(|| self.build_steal_report())
    }
    fn working_set(&self) -> Option<Vec<(Bits, u8)>> {
        self.tt.working_set()
    }
    fn pc_hist(&self) -> Option<Vec<u64>> {
        self.hist.then(|| {
            self.pc_hist
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect()
        })
    }
    fn prof_data(&self) -> Option<Vec<u64>> {
        self.prof.then(|| {
            self.prof_data
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect()
        })
    }
    fn per_worker_nodes(&self) -> Option<Vec<u64>> {
        Some(self.tt.per_worker_nodes())
    }
    fn root_progress(&self) -> Option<(u64, u64)> {
        let total = self.root_total.load(Ordering::Relaxed);
        (total > 0).then(|| (self.root_done.load(Ordering::Relaxed).min(total), total))
    }
    fn stats(&self) -> String {
        let (done, total) = (
            self.root_done.load(Ordering::Relaxed),
            self.root_total.load(Ordering::Relaxed),
        );
        let ma = self.eff_min_avail.load(Ordering::Relaxed);
        let ma = if ma == u32::MAX {
            "off".to_string()
        } else {
            ma.to_string()
        };
        let oracle = if self.nimber_oracle {
            let attempts = self.oracle_attempts.load(Ordering::Relaxed);
            let hits = self.oracle_hits.load(Ordering::Relaxed);
            let comp_hits = self.oracle_comp_hits.load(Ordering::Relaxed);
            let comp_misses = self.oracle_comp_misses.load(Ordering::Relaxed);
            let hit_pct = if attempts > 0 {
                100.0 * hits as f64 / attempts as f64
            } else {
                0.0
            };
            format!(
                ", oracle {hits}/{attempts} ({hit_pct:.1}%) · comp-cache {comp_hits}/{comp_misses}"
            )
        } else {
            String::new()
        };
        let dense = self
            .dense8
            .as_ref()
            .map(|d| format!(", W8 {:.0} MB", d.bytes() as f64 / (1 << 20) as f64))
            .unwrap_or_default();
        format!(
            "{} rayon workers, {done}/{total} root moves, par-depth {}/min-avail {ma}, iso<= {}{dense} · {}",
            rayon::current_num_threads(),
            self.par_depth,
            self.iso_max_avail,
            self.tt.summary() + &oracle,
        )
    }
    // No `tt()` for now: a flat QueensTt could checkpoint, but its image header (TT_CANON_ID)
    // does not yet distinguish a selective iso/D4-keyed table from a plain D4 one, so a
    // cross-mode `--resume` would mis-key. A key-mode header tag is the follow-up.
}
