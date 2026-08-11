//! The lockless transposition table, its checkpoint image format, the BuRR
//! freeze stream, and the df-pn proof-number table.

use super::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Flush a worker's thread-local tally into the shared atomics ≈ once a second (at the
/// ~M-node/s rates here). Per-node the search touches only thread-local memory -- no
/// cross-CCX atomic on the shared `nodes` counter, which on this 2-CCX box bounces over the
/// Infinity Fabric and measured a ~2× throughput drag every node (mirror of the BurrStore
/// fix, `store.rs`). The shared counters are exact again after a [`QueensTt::drain_all`].
const FLUSH_NODES: u64 = 1 << 18;

/// Resolve the node-flush cadence once at TT construction (`QUEENS_FLUSH_NODES`, default
/// [`FLUSH_NODES`]). Lowering it makes the shared `nodes` / `per_worker` counters update more often —
/// finer time-series resolution during the slow serial collapses (`QUEENS_TS_FILE`) — at the cost of
/// more cross-CCX atomic traffic, so it is for measurement runs only.
fn flush_nodes_env() -> u64 {
    std::env::var("QUEENS_FLUSH_NODES")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(FLUSH_NODES)
}

/// Per-worker, **non-atomic** accumulators for the hot-loop node counter and the distinct
/// estimator. Each rayon worker owns one (thread-local), incremented with plain integer ops
/// per node; flushed to the shared atomic + HLL once every [`FLUSH_NODES`] nodes and drained
/// at search end.
struct Acc {
    nodes: u64,
    /// Thread-local HLL registers (`2^p` bytes), lazily sized to the shared estimator on
    /// first feed and merged by max at flush; empty when this table isn't counting.
    hll: Vec<u8>,
}

thread_local! {
    static ACC: RefCell<Acc> = const { RefCell::new(Acc { nodes: 0, hll: Vec::new() }) };
}

// --------------------------------------------------------------------------- //
// Transposition table
// --------------------------------------------------------------------------- //

/// A fixed-size, **lockless** open-addressing transposition table keyed by a board
/// mask -> a `u8` value (win/loss as 0/1, or a Sprague-Grundy nimber). Memory is
/// capped at `2^bits` slots; a fingerprint mismatch is a miss, so eviction only
/// costs recompute (and a foreign same-slot+same-fingerprint hit is a ~`2^-55`
/// wrong, cross-checked vs the known verdict).
///
/// Each slot is a single [`Slot`] = one `u64`, so the table is a flat
/// `Box<[AtomicU64]>` shared lock-free across rayon workers: `get`/`put` are a
/// `Relaxed` `load`/`store`. No mutex, no sharding (Session 5, lead L1). This is
/// safe by construction -- an `AtomicU64` `load` cannot tear, and the value stored
/// for a key is deterministic (a position's win/loss or nimber is fixed), so even
/// a concurrent write for the *same* key stores the *same* value; a write for a
/// *different* key is rejected by the fingerprint. That removes a lock/unlock and
/// the mutex cache-line bounce from every node, attacking the DRAM-latency wall
/// and the mutex contention in the ~18x parallel ceiling. (Hyatt's XOR-key trick
/// is unnecessary: the 55-bit fingerprint already self-validates identity.)
pub struct QueensTt {
    slots: Box<[AtomicU64]>,
    /// Slot count (any value, not just a power of two -- see [`QueensTt::index`]).
    len: u64,
    nodes: AtomicU64,
    /// Per-rayon-worker node tally for live throughput telemetry (`QUEENS_TELEM`). Attributed
    /// at the ~1/[`FLUSH_NODES`] flush via `rayon::current_thread_index()`, so it costs nothing
    /// on the per-node path. 256 slots covers any core count; the watcher reads deltas.
    per_worker: Box<[AtomicU64]>,
    /// Node-flush cadence (`QUEENS_FLUSH_NODES`, default [`FLUSH_NODES`]); resolved once. Lower ⇒ finer
    /// time-series resolution during the serial collapses, more atomic traffic (measurement only).
    flush_nodes: u64,
    /// Optional distinct-position instrumentation (Chunk 1). `None` for an
    /// ordinary solve, so the production path pays only a predictable null check.
    counter: Option<Counter>,
    /// `QUEENS_TT_SEGMENT=1`: route by available-popcount into a per-pc band of the same
    /// flat table (`index_seg`), so the DFS working set is TLB-local without shrinking the
    /// table. Resolved once at construction; the flat [`index`](Self::index) path is left
    /// byte-identical as the A/B control (the solver monomorphises on it, never branches
    /// per node). `false` ⇒ `band_*` are empty and unused.
    segment: bool,
    /// Exclusive prefix-sum start of each popcount band (`band_base[pc]`), indexed by
    /// available-popcount `0..=256`. Empty unless [`segment`](Self::segment).
    band_base: Box<[u64]>,
    /// Slot count of each popcount band (`band_size[pc] ≥ 1`); `Σ band_size == len`.
    /// Sized so each band carries a comparable load factor (∝ the put distribution).
    band_size: Box<[u64]>,
    /// `QUEENS_TT_ASSOC=1` (only with [`segment`](Self::segment)): make each band probe a
    /// cache-line **bucket** of [`TT_ASSOC_WAYS`] slots ([`index_seg`](Self::index_seg)'s
    /// single-slot route becomes [`bucket_base`](Self::bucket_base)). The whole bucket rides
    /// in one 64-byte line the probe already fetches, so scanning all ways is free, but a
    /// collision now evicts only when *all* ways are full — far fewer conflict misses (→
    /// fewer re-expansions) than the direct-mapped slot. Resolved once at construction; the
    /// flat/seg paths stay byte-identical (the solver monomorphises `M_SEG_ASSOC`, never
    /// branches per node). `false` ⇒ the single-slot [`index_seg`](Self::index_seg) is used.
    assoc: bool,
}

/// Band count for the segmented TT: one per available-popcount, `0..=256` (the n=16 board
/// has `16*16 = 256` squares). Mirrors `MAXPC` in the solver's histogram.
const TT_MAXPC: usize = 257;

/// Floor slots per band, so every band is non-empty (Lemire `fastrange` needs a non-zero
/// modulus) and an unweighted popcount can't divide-by-zero. Negligible against a multi-GB
/// table (`64 * 257 ≈ 16 K` slots reserved).
const TT_MIN_BAND: u64 = 64;

/// Set-associativity for the segmented-TT `QUEENS_TT_ASSOC` bucket: 8 × `u64` slots = one
/// 64-byte cache line, so a bucket probe is a single line fetch and the whole bucket is
/// scanned for free. Bounds `TT_MIN_BAND` from below (a band must hold ≥ one bucket).
const TT_ASSOC_WAYS: usize = 8;

/// A compact 8-byte transposition slot (Chunk 2): one `u64` packing a used flag
/// (bit 0), the 8-bit value (bits 1..9 -- the win/loss bit for the search, or a
/// small Sprague-Grundy nimber for [`Nimber`]), and a 55-bit fingerprint of the
/// canonical key (bits 9..64).
///
/// We store a *fingerprint* of the key, not the full 256-bit key. The slot index
/// already pins ~`bits` bits of the routing hash, and the fingerprint comes from
/// an *independent* 64-bit hash half (see [`QueensTt::hash128`]), so a wrong "hit"
/// -- a different key landing in the same slot *and* matching the fingerprint --
/// has probability ~`2^-55` per colliding probe: negligible even across a
/// Jenrich-scale (~`10^11`) search, and the final verdict is cross-checked against
/// the known result. This shrinks the slot 40 B -> 8 B (5x more entries per byte
/// of RAM) -- the Chunk-2 dynamic-tier win -- while keeping the canonical
/// `available`-mask key, so every transposition still merges exactly as before (no
/// lost merges, unlike re-keying on the queen set). The old strict "collision =
/// miss, never wrong" weakens to "wrong with vanishing probability"; a fingerprint
/// *mismatch* is still just a miss that re-searches.
#[derive(Clone, Copy, Default)]
#[repr(transparent)] // hot-struct discipline (CLAUDE.md #4): explicit layout = the inner `u64`.
pub(crate) struct Slot(pub(crate) u64);

// #7: lock the one-word slot — a field-add that grew it would silently double the TT footprint.
const _: () = assert!(std::mem::size_of::<Slot>() == 8 && std::mem::align_of::<Slot>() == 8);

impl Slot {
    /// Fingerprint width: `64 - 1 (used) - 8 (val)` bits.
    pub(crate) const FP_BITS: u32 = 55;
    const FP_SHIFT: u32 = 9;
    const VAL_SHIFT: u32 = 1;

    #[inline]
    pub(crate) const fn fp_mask() -> u64 {
        (1u64 << Self::FP_BITS) - 1
    }
    /// Pack `val` and the low `FP_BITS` of `fp` into an occupied slot.
    #[inline]
    pub(crate) fn pack(fp: u64, val: u8) -> Slot {
        Slot(1 | ((val as u64) << Self::VAL_SHIFT) | ((fp & Self::fp_mask()) << Self::FP_SHIFT))
    }
    #[inline]
    pub(crate) fn used(self) -> bool {
        self.0 & 1 != 0
    }
    #[inline]
    pub(crate) fn val(self) -> u8 {
        (self.0 >> Self::VAL_SHIFT) as u8
    }
    #[inline]
    pub(crate) fn fp(self) -> u64 {
        self.0 >> Self::FP_SHIFT
    }
    /// ABDADA in-flight sentinel stored in `val`: a worker writes this when it *begins*
    /// expanding a node (the slot is otherwise empty until the completing put), so a second
    /// worker that probes the same key learns the subtree is already being computed and can
    /// defer it instead of re-expanding. Real verdicts only ever occupy `val ∈ {0, 1}`, so
    /// `0xFF` is an unambiguous marker. The completing put overwrites it with the real value.
    pub(crate) const IN_FLIGHT: u8 = 0xFF;
}

/// Tri-state TT probe for the ABDADA in-flight protocol ([`QueensTt::get_inflight_hashed`]):
/// a fingerprint-matching slot is either a resolved verdict ([`Probe3::Hit`]) or a marker left
/// by a worker mid-expansion ([`Probe3::InFlight`]); anything else is a [`Probe3::Miss`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Probe3 {
    Hit(u8),
    InFlight,
    Miss,
}

/// 1024 shards for [`PnTt`] (still mutex-sharded; `pn` is a tiny-board experiment,
/// not under memory pressure). [`QueensTt`] is lockless and unsharded.
const SHARD_BITS: u32 = 10;

/// Allocate `size` zeroed [`AtomicU64`] slots backed by transparent huge pages
/// (Session 5, L1 cluster). The table is probed at random, so a multi-GB table on
/// 4 KB pages thrashes the TLB on every node; `MADV_HUGEPAGE` cuts that hard. We
/// allocate via `vec![0u64; _]` -- the allocator's `alloc_zeroed`, so the OS hands
/// back lazily-zeroed pages (a 17 GB table does not commit until probed) -- then
/// reinterpret the buffer as `AtomicU64`.
pub(crate) fn zeroed_huge_atomics(size: usize) -> Box<[AtomicU64]> {
    let mut v: Vec<u64> = vec![0u64; size];
    #[cfg(target_os = "linux")]
    unsafe {
        // SAFETY: `madvise(MADV_HUGEPAGE)` is advisory -- it only changes page backing,
        // never contents. The start address MUST be page-aligned or madvise returns
        // EINVAL, and `Vec<u64>` is only 8-byte aligned (glibc returns a pointer 16
        // bytes past its mmap chunk header, e.g. `0x..010`), so we advise the
        // page-aligned interior. The kernel then huge-backs the 2 MB-aligned VAs inside
        // it; the sub-page unaligned prefix stays small-paged (negligible on a multi-GB
        // table). Before this fix the call silently EINVAL'd on *every* table, so the
        // random-access probe ran entirely on 4 KB pages -- the dTLB thrash this was
        // meant to cut. The result is still ignored: a genuine no-op (THP off) is fine.
        let base = v.as_mut_ptr() as usize;
        let bytes = std::mem::size_of_val(v.as_slice());
        let page = libc::sysconf(libc::_SC_PAGESIZE).max(1) as usize;
        let aligned = base.next_multiple_of(page);
        let off = aligned - base;
        if bytes > off {
            let len = bytes - off;
            let ptr = aligned as *mut libc::c_void;
            libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
            // `MADV_HUGEPAGE` is only a hint: the kernel forms a 2 MB page lazily, and only
            // when a fully-aligned 2 MB region faults in contiguously. A multi-GB table is
            // first-touched at *random* slots, so on this box only ~70% of it ever reaches
            // THP -- the 4 KB remnant is ~half the dTLB misses on the hot probe (measured:
            // 17 GB RSS, 12 GB AnonHugePages). `MADV_COLLAPSE` (Linux 6.1+) forces a
            // synchronous collapse of the whole range into 2 MB pages now, allocating/
            // compacting as needed. Best-effort and env-gated: it commits the range up
            // front (a full n=16 solve touches almost all of it anyway) and an older kernel
            // just returns EINVAL (ignored, as MADV_HUGEPAGE's result already is).
            //
            // Default-ON for large tables (the n>=16 regime where the dTLB thrash dominates
            // — measured ~5% wall on iso-window with near-identical node counts, so a genuine
            // per-node TLB win, not noise). Small tables skip it: TLB isn't their bottleneck
            // and the up-front commit would defeat their lazy allocation. `QUEENS_TT_COLLAPSE`
            // overrides either way (`0` forces off, anything else forces on).
            let collapse = match std::env::var("QUEENS_TT_COLLAPSE").ok().as_deref() {
                Some("0") => false,
                Some(_) => true,
                None => len >= (1usize << 32), // ~4 GB+ table (n>=16); n<=14 is ~1 GB
            };
            if collapse {
                // ABI-stable advice values not yet in the `libc` crate on this toolchain
                // (include/uapi/asm-generic/mman-common.h).
                const MADV_POPULATE_WRITE: libc::c_int = 23; // Linux 5.14+
                const MADV_COLLAPSE: libc::c_int = 25; // Linux 6.1+
                                                       // MADV_COLLAPSE only merges *populated* pages, and the table is lazily
                                                       // zeroed (unpopulated) here, so prefault the whole range first. This
                                                       // commits the table up front (a full n=16 solve touches ~all of it anyway)
                                                       // and lets COLLAPSE promote the randomly-faulted 4 KB remnant — the ~27%
                                                       // that THP leaves small-paged — into 2 MB pages, cutting dTLB pressure.
                                                       //
                                                       // Both calls fault/compact the *whole* 17 GB range synchronously — single-threaded,
                                                       // this is the multi-second silent startup gap. Split the range into 2 MB-aligned
                                                       // chunks (one per rayon worker) and fault+collapse them in parallel; the work is
                                                       // memory-bandwidth-bound, so it scales until the controllers saturate (≫1 core).
                let nthreads = rayon::current_num_threads().max(1);
                let huge = 2 * 1024 * 1024usize;
                let chunk = (len / nthreads).next_multiple_of(huge).max(huge);
                let base = ptr as usize;
                rayon::broadcast(|ctx| {
                    let start = ctx.index() * chunk;
                    if start >= len {
                        return;
                    }
                    let clen = chunk.min(len - start);
                    // SAFETY: each worker advises a disjoint, in-bounds sub-range of the same
                    // mapping; `madvise` only faults/compacts pages (no user-data write), so
                    // concurrent advice on disjoint ranges cannot race.
                    let cptr = (base + start) as *mut libc::c_void;
                    libc::madvise(cptr, clen, MADV_POPULATE_WRITE);
                    libc::madvise(cptr, clen, MADV_COLLAPSE);
                });
            }
        }
    }
    let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
    std::mem::forget(v);
    // SAFETY: `AtomicU64` has the same size, alignment, and representation as `u64`
    // (std guarantee), and we take sole ownership of the same `(ptr, len, cap)`
    // allocation exactly once; `len == cap`, so `into_boxed_slice` cannot realloc.
    unsafe { Vec::from_raw_parts(ptr.cast::<AtomicU64>(), len, cap) }.into_boxed_slice()
}

/// Resolve the `QUEENS_TT_SLOTS` exact-slot-count override once (at table
/// construction, never per node). `Some(n)` clamps to at least 2 slots; `None` keeps
/// the `2^bits` default. Lets a run fill all RAM via `fastrange` sizing (Chunk 2b).
fn tt_slots_override() -> Option<usize> {
    std::env::var("QUEENS_TT_SLOTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| n.max(2))
}

/// Embedded first-cut band weights: the **n=14** flat-TT put distribution by
/// available-popcount (`QUEENS_PC_HIST=1 queens solve 14 iso-window`), stored from `pc = 9`
/// (the first non-empty popcount). The fallback when `QUEENS_TT_BANDS` is not given — good
/// for n ≤ 14 validation; the n=16 A/B feeds real n=16 weights via the file (the shape is
/// similar but the range extends higher). See the iso-window handoff.
const N14_PUTS_FROM9: [u64; 57] = [
    4_953_830, 3_618_177, 2_931_227, 2_478_592, 1_938_913, 1_333_717, 863_474, 645_630, 628_131,
    682_025, 685_056, 589_228, 429_732, 264_266, 145_087, 84_190, 69_043, 75_378, 83_798, 83_381,
    71_300, 52_281, 32_133, 16_558, 7_474, 3_329, 2_048, 2_611, 4_589, 7_279, 11_239, 15_500,
    18_631, 19_159, 16_191, 11_884, 7_347, 3_924, 1_722, 628, 182, 103, 90, 129, 232, 367, 571,
    623, 806, 702, 626, 430, 293, 121, 44, 19, 8,
];

/// Resolve the segmented-TT band table for a `len`-slot table: `QUEENS_TT_SEGMENT=1` enables
/// it (else `(false, empty, empty)`). Band weights come from `QUEENS_TT_BANDS=<path>` (one
/// per-popcount count per line, the format [`QUEENS_PC_HIST`] writes via `QUEENS_PC_HIST_OUT`)
/// if set and readable, else the embedded n=14 distribution. Resolved once at construction.
fn resolve_segment_bands(len: u64) -> (bool, Box<[u64]>, Box<[u64]>) {
    if std::env::var("QUEENS_TT_SEGMENT").as_deref() != Ok("1") {
        return (false, Box::new([]), Box::new([]));
    }
    let counts = load_band_counts().unwrap_or_else(default_band_counts);
    let (base, size) = build_bands(&counts, len);
    (true, base, size)
}

/// Resolve `QUEENS_TT_ASSOC=1` once at construction. Set-associative band buckets only make
/// sense over a segmented table (they refine `index_seg`'s single slot into a cache-line
/// bucket), so the flag is honoured only when `segment` is on; set without `QUEENS_TT_SEGMENT`
/// it is a no-op (warned, so a mis-set A/B isn't silently the flat control).
fn resolve_assoc(_segment: bool) -> bool {
    // `QUEENS_TT_ASSOC=1` works two ways: with `QUEENS_TT_SEGMENT=1` it buckets *within* each
    // popcount band (seg-assoc); without it, it buckets the *flat* table (flat-assoc) — the
    // band-free variant that compares directly to the production flat direct-mapped TT (the
    // K=16 getK ceiling voids the pc≤16 bands, so the embedded band weights mis-size). The
    // `bucket_base` flat path keys off the empty band tables.
    std::env::var("QUEENS_TT_ASSOC").as_deref() == Ok("1")
}

/// The embedded n=14 weights as a full per-popcount count array.
fn default_band_counts() -> [u64; TT_MAXPC] {
    let mut c = [0u64; TT_MAXPC];
    for (i, &v) in N14_PUTS_FROM9.iter().enumerate() {
        c[9 + i] = v;
    }
    c
}

/// Parse `QUEENS_TT_BANDS=<path>`: one non-negative integer per line, count for `pc = line
/// index` (`0..TT_MAXPC`); missing trailing lines are zero. Returns `None` if unset or
/// unreadable (caller falls back to the embedded table).
fn load_band_counts() -> Option<[u64; TT_MAXPC]> {
    let path = std::env::var("QUEENS_TT_BANDS").ok()?;
    let text = std::fs::read_to_string(&path)
        .map_err(|e| eprintln!("QUEENS_TT_BANDS: cannot read {path}: {e}; using embedded n=14"))
        .ok()?;
    let mut c = [0u64; TT_MAXPC];
    for (pc, line) in text.lines().take(TT_MAXPC).enumerate() {
        c[pc] = line.trim().parse().unwrap_or(0);
    }
    Some(c)
}

/// Partition `len` slots into per-popcount bands with size ∝ `counts[pc]` (each ≥
/// [`TT_MIN_BAND`], so a band is never empty), returning `(band_base, band_size)` with
/// `Σ band_size == len`. The rounding/floor remainder goes to the heaviest band.
fn build_bands(counts: &[u64; TT_MAXPC], len: u64) -> (Box<[u64]>, Box<[u64]>) {
    let total: u128 = counts.iter().map(|&c| c as u128).sum();
    let reserved = TT_MIN_BAND * TT_MAXPC as u64;
    // Distribute the slots above the per-band floor proportionally to the weights.
    let free = len.saturating_sub(reserved);
    let mut size = vec![0u64; TT_MAXPC];
    for (pc, s) in size.iter_mut().enumerate() {
        let share = if total > 0 {
            (counts[pc] as u128 * free as u128 / total) as u64
        } else {
            free / TT_MAXPC as u64 // no weights: uniform
        };
        *s = TT_MIN_BAND + share;
    }
    // Floor division leaves a remainder (`len - Σ size ≥ 0`); give it to the heaviest band
    // so `Σ size == len` exactly and the band that needs slots most gets the surplus.
    let assigned: u64 = size.iter().sum();
    let heaviest = (0..TT_MAXPC).max_by_key(|&pc| counts[pc]).unwrap_or(0);
    size[heaviest] += len - assigned;
    // Exclusive prefix sum for the band starts.
    let mut base = vec![0u64; TT_MAXPC];
    let mut acc = 0u64;
    for pc in 0..TT_MAXPC {
        base[pc] = acc;
        acc += size[pc];
    }
    debug_assert_eq!(acc, len);
    (base.into_boxed_slice(), size.into_boxed_slice())
}

// --------------------------------------------------------------------------- //
// Dumpable / reloadable image (checkpoint + resume; proposal 2026-06-15)
// --------------------------------------------------------------------------- //

/// Magic for a [`QueensTt`] image file. Bumped only if the wire layout changes.
const TT_MAGIC: [u8; 8] = *b"QNSTT\0\0\0";
/// `Slot` layout version (`{used:1, val:8, fp:55}` + `fastrange` routing). Bump on
/// any change to `Slot` packing or the `index` function.
const TT_FORMAT_VERSION: u32 = 1;
/// [`QueensTt::hash128`] seeds/constants version. Bump if either hash half changes
/// (a stale fingerprint would silently mis-route).
const TT_HASH_ID: u32 = 1;
/// `canon`/`pos_key` version. Bump if the canonical key changes (every stored key
/// would then refer to a different position).
const TT_CANON_ID: u32 = 1;
/// Arch/endianness tag: raw little-endian `u64` slots. `1` = x86_64-LE.
const TT_ARCH_X86_64_LE: u8 = 1;
/// Fixed header size in bytes (the rest is the raw slot image).
pub(crate) const TT_HEADER_LEN: usize = 64;

/// The on-disk header of a dumped [`QueensTt`]. The fixed fields tag the *exact*
/// slot layout, hash, canonicalisation, and arch a reload depends on -- a mismatch
/// is a hard error (`io::ErrorKind::InvalidData`), never a silently-voided hit.
///
/// `len` is the slot **count**, not `bits`: routing is `fastrange(route, len)`
/// (see [`QueensTt::index`]), so a table of a different size re-routes every entry
/// and the stored fingerprint -- an independent hash half, not the key -- cannot be
/// recomputed. **An image only reloads into a table of the same `len`.** `epoch` is
/// reserved for delta checkpoints (proposal Phase 2); `fill` is reporting only.
pub struct TtHeader {
    pub n: u8,
    pub len: u64,
    pub fill: u64,
    pub epoch: u32,
    /// Search nodes (TT misses) accumulated when the image was dumped, so a `--resume`
    /// restores the node counter and the progress reflects the *total* work, not just
    /// the post-resume continuation. Stored in the previously-reserved header bytes, so
    /// older images (which have zeroes there) simply resume from 0 -- backward-compatible.
    pub nodes: u64,
}

impl TtHeader {
    fn to_bytes(&self) -> [u8; TT_HEADER_LEN] {
        let mut b = [0u8; TT_HEADER_LEN];
        b[0..8].copy_from_slice(&TT_MAGIC);
        b[8..12].copy_from_slice(&TT_FORMAT_VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&TT_HASH_ID.to_le_bytes());
        b[16..20].copy_from_slice(&TT_CANON_ID.to_le_bytes());
        b[20..24].copy_from_slice(&self.epoch.to_le_bytes());
        b[24..32].copy_from_slice(&self.len.to_le_bytes());
        b[32..40].copy_from_slice(&self.fill.to_le_bytes());
        b[40] = self.n;
        b[41] = TT_ARCH_X86_64_LE;
        b[42..50].copy_from_slice(&self.nodes.to_le_bytes());
        // b[50..64] reserved (zero)
        b
    }

    /// Validate and parse a header, hard-erroring on any tag mismatch so a stale or
    /// foreign dump is rejected rather than quietly producing wrong hits.
    fn parse(b: &[u8]) -> io::Result<TtHeader> {
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
        if b.len() < TT_HEADER_LEN {
            return Err(bad("truncated TT header".into()));
        }
        if b[0..8] != TT_MAGIC {
            return Err(bad("not a queens TT image (bad magic)".into()));
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let check = |got: u32, want: u32, what: &str| {
            (got == want)
                .then_some(())
                .ok_or_else(|| bad(format!("{what} mismatch: image {got}, this build {want}")))
        };
        check(u32_at(8), TT_FORMAT_VERSION, "format_version")?;
        check(u32_at(12), TT_HASH_ID, "hash_id")?;
        check(u32_at(16), TT_CANON_ID, "canon_id")?;
        if b[41] != TT_ARCH_X86_64_LE {
            return Err(bad(format!(
                "arch mismatch: image {}, expected x86_64-LE",
                b[41]
            )));
        }
        Ok(TtHeader {
            epoch: u32_at(20),
            len: u64::from_le_bytes(b[24..32].try_into().unwrap()),
            fill: u64::from_le_bytes(b[32..40].try_into().unwrap()),
            n: b[40],
            nodes: u64::from_le_bytes(b[42..50].try_into().unwrap()),
        })
    }
}

/// Slots transferred per read/write block (`BLOCK * 8` bytes ≈ 512 KB) -- amortises
/// per-call overhead over the streamed image without a large buffer.
const TT_IO_BLOCK: usize = 1 << 16;

impl QueensTt {
    /// A lockless table of `2^bits` slots (each 8 bytes; see [`Slot`]). `bits` is the
    /// memory cap knob. `QUEENS_TT_SLOTS` overrides with an exact slot **count** (any
    /// value, not just a power of two) -- resolved once here, never per node -- so a run
    /// can fill *all* available RAM rather than the next power of two below it (Chunk 2b;
    /// at 8 B/slot the 2^31 = 17 GB → 2^32 = 34 GB gap straddles a 26 GB box's sweet
    /// spot). Indexing is Lemire `fastrange` ([`QueensTt::index`]), which maps a hash to
    /// `[0, len)` for any `len`.
    pub fn new(bits: u32) -> Self {
        let size = tt_slots_override().unwrap_or_else(|| 1usize << bits.max(1));
        let (segment, band_base, band_size) = resolve_segment_bands(size as u64);
        QueensTt {
            slots: zeroed_huge_atomics(size),
            len: size as u64,
            nodes: AtomicU64::new(0),
            per_worker: (0..256).map(|_| AtomicU64::new(0)).collect(),
            flush_nodes: flush_nodes_env(),
            counter: None,
            segment,
            band_base,
            band_size,
            assoc: resolve_assoc(segment),
        }
    }

    /// Whether this table routes by per-popcount band ([`index_seg`](Self::index_seg))
    /// rather than the flat [`index`](Self::index). Resolved once at construction.
    #[inline]
    pub fn is_segmented(&self) -> bool {
        self.segment
    }

    /// Whether the segmented table uses set-associative cache-line buckets
    /// ([`bucket_base`](Self::bucket_base)) rather than a single slot per band route. Implies
    /// [`is_segmented`](Self::is_segmented). Resolved once at construction.
    #[inline]
    pub fn is_assoc(&self) -> bool {
        self.assoc
    }

    /// Segmented slot index: route the key into its popcount band, then `fastrange` within
    /// that band. A pure function of `(route, pc)` and `pc` is a pure function of the key, so
    /// the same key always lands in the same slot (transposition-safe — no lost merges, unlike
    /// worker-sharding). The whole DFS working set at a given depth shares a band, so its
    /// random probes stay within a small, TLB-resident slice of the table.
    #[inline]
    fn index_seg(&self, route: u64, pc: u32) -> usize {
        // SAFETY-of-bounds: `band_base`/`band_size` have `TT_MAXPC` entries and `pc` is an
        // available-popcount ≤ 256; clamp defensively so a stray pc can never index OOB.
        let pc = (pc as usize).min(TT_MAXPC - 1);
        let base = self.band_base[pc];
        let size = self.band_size[pc];
        let off = ((route as u128).wrapping_mul(size as u128) >> 64) as u64;
        (base + off) as usize
    }

    /// [`get_hashed`](Self::get_hashed) routed by popcount band ([`index_seg`](Self::index_seg)).
    /// The counter hook is skipped: segmented runs are production (`!COUNT`).
    #[inline]
    pub(crate) fn get_seg_hashed(&self, route: u64, fp: u64, pc: u32) -> Option<u8> {
        let s = Slot(self.slots[self.index_seg(route, pc)].load(Ordering::Relaxed));
        (s.used() && s.fp() == (fp & Slot::fp_mask())).then(|| s.val())
    }
    /// [`put_hashed`](Self::put_hashed) routed by popcount band.
    #[inline]
    pub(crate) fn put_seg_hashed(&self, route: u64, fp: u64, pc: u32, val: u8) {
        self.slots[self.index_seg(route, pc)].store(Slot::pack(fp, val).0, Ordering::Relaxed);
    }
    /// [`prefetch_hashed`](Self::prefetch_hashed) routed by popcount band.
    #[inline]
    pub(crate) fn prefetch_seg_hashed(&self, route: u64, pc: u32) {
        let ptr = self.slots[self.index_seg(route, pc)].as_ptr();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // SAFETY: as [`prefetch_hashed`](Self::prefetch_hashed) -- warms a valid
            // in-allocation pointer, no architectural effect, cannot fault.
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// First slot of the set-associative bucket a key routes to within its popcount band
    /// (`QUEENS_TT_ASSOC`). Pick one of the band's `band_size/WAYS` buckets by `fastrange`,
    /// then align to a [`TT_ASSOC_WAYS`]-slot boundary so the whole bucket sits in one 64-byte
    /// cache line (the production table is mmap'd and thus page-aligned, so slot `% WAYS == 0`
    /// is a 64-byte-aligned address). Pure function of `(route, pc)` — like
    /// [`index_seg`](Self::index_seg), transposition-safe. The down-align can pull the base ≤
    /// `WAYS-1` slots into the previous band at a boundary; harmless — still in-allocation and
    /// fingerprint-checked, never wrong, at most a few extra slots shared across two bands.
    #[inline]
    fn bucket_base(&self, route: u64, pc: u32) -> usize {
        // Flat-assoc (no segmentation): bucket the whole table; `pc` is ignored. This is the
        // band-free path that compares apples-to-apples with the production flat direct-mapped
        // table (the segmented bands mis-size at K=16 — getK voids the pc≤16 bands). Seg-assoc
        // keeps the per-band bucketing.
        let (base, size) = if self.band_base.is_empty() {
            (0u64, self.len)
        } else {
            let pc = (pc as usize).min(TT_MAXPC - 1);
            (self.band_base[pc], self.band_size[pc])
        };
        // size ≥ TT_MIN_BAND (64) ≥ WAYS (seg) or = len ≫ WAYS (flat), so n_buckets ≥ 1.
        let n_buckets = size / TT_ASSOC_WAYS as u64;
        let bucket = ((route as u128).wrapping_mul(n_buckets as u128) >> 64) as u64;
        let slot0 = (base + bucket * TT_ASSOC_WAYS as u64) & !(TT_ASSOC_WAYS as u64 - 1);
        slot0 as usize
    }

    /// [`get_seg_hashed`](Self::get_seg_hashed), but scan the key's [`TT_ASSOC_WAYS`]-way
    /// cache-line bucket for a fingerprint match. The whole bucket is one cache line the probe
    /// already fetches, so the scan is L1-resident after the first load. On znver5 the scan is
    /// one AVX-512 load + mask-compare ([`get_assoc_avx512`](Self::get_assoc_avx512)); the
    /// profile showed the scalar 8-way loop's cost is pure instruction/branch count (+16%
    /// instr/node, CPI unchanged), so vectorising it is the whole win.
    #[inline]
    pub(crate) fn get_assoc_hashed(&self, route: u64, fp: u64, pc: u32) -> Option<u8> {
        let b = self.bucket_base(route, pc);
        let want = fp & Slot::fp_mask();
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        // SAFETY: `bucket_base` guarantees `b + TT_ASSOC_WAYS <= len`, so the 64-byte
        // (8×u64) bucket load starting at slot `b` stays in-allocation.
        unsafe {
            self.get_assoc_avx512(b, want)
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            for i in 0..TT_ASSOC_WAYS {
                // SAFETY: `bucket_base` guarantees `b + i < len` for `i < TT_ASSOC_WAYS`.
                let s = Slot(unsafe { self.slots.get_unchecked(b + i) }.load(Ordering::Relaxed));
                if s.used() && s.fp() == want {
                    return Some(s.val());
                }
            }
            None
        }
    }

    /// [`put_seg_hashed`](Self::put_seg_hashed), into the key's [`TT_ASSOC_WAYS`]-way bucket:
    /// refresh an existing entry for this key, else fill the first empty way, else evict a
    /// route-selected way. Only the all-ways-full case evicts (vs the direct-mapped slot's
    /// evict-on-every-collision) — that is the conflict-miss reduction. Lockless like the rest:
    /// a racing writer for a *different* key can only clobber a way (a lost memo entry =
    /// recompute, never a wrong value, since values are fingerprint-checked on read). The
    /// bucket load is free here: the store must bring the line in (write-allocate) regardless.
    #[inline]
    pub(crate) fn put_assoc_hashed(&self, route: u64, fp: u64, pc: u32, val: u8) {
        let b = self.bucket_base(route, pc);
        let want = fp & Slot::fp_mask();
        let packed = Slot::pack(fp, val).0;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        // SAFETY: as `get_assoc_hashed` — `bucket_base` guarantees `b + TT_ASSOC_WAYS <= len`.
        unsafe {
            self.put_assoc_avx512(b, want, route, packed);
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            let mut empty: usize = TT_ASSOC_WAYS; // sentinel: no empty way seen
            for i in 0..TT_ASSOC_WAYS {
                // SAFETY: `bucket_base` guarantees `b + i < len` for `i < TT_ASSOC_WAYS`.
                let s = Slot(unsafe { self.slots.get_unchecked(b + i) }.load(Ordering::Relaxed));
                if s.used() {
                    if s.fp() == want {
                        unsafe { self.slots.get_unchecked(b + i) }.store(packed, Ordering::Relaxed);
                        return;
                    }
                } else if empty == TT_ASSOC_WAYS {
                    empty = i;
                }
            }
            let way = if empty != TT_ASSOC_WAYS {
                empty
            } else {
                (route as usize >> 3) & (TT_ASSOC_WAYS - 1)
            };
            // SAFETY: `way < TT_ASSOC_WAYS`, so `b + way < len`.
            unsafe { self.slots.get_unchecked(b + way) }.store(packed, Ordering::Relaxed);
        }
    }

    /// AVX-512 body of [`get_assoc_hashed`](Self::get_assoc_hashed): load the 8-slot bucket as
    /// one `__m512i`, mask the lanes whose stored fingerprint (`slot >> 9`) equals `want` and
    /// whose used bit (bit 0) is set, and return the first such lane's value. The common case
    /// for a *searched* node is a miss (that is why it is being expanded): then it is just a
    /// load + two mask ops + a branch — no value extraction.
    ///
    /// # Safety
    /// `b + TT_ASSOC_WAYS <= self.slots.len()`. The vector load is not per-lane atomic, but the
    /// table tolerates torn reads by construction (lockless, fingerprint-validated): a garbled
    /// lane fails the `cmpeq` → a miss → recompute, never a wrong value.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline]
    unsafe fn get_assoc_avx512(&self, b: usize, want: u64) -> Option<u8> {
        use std::arch::x86_64::*;
        let v = _mm512_loadu_si512(self.slots.as_ptr().add(b) as *const __m512i);
        let fps = _mm512_srli_epi64::<{ Slot::FP_SHIFT }>(v);
        let fp_match = _mm512_cmpeq_epi64_mask(fps, _mm512_set1_epi64(want as i64));
        let used = _mm512_test_epi64_mask(v, _mm512_set1_epi64(1));
        let hit = fp_match & used;
        if hit == 0 {
            return None;
        }
        // Extract the matching lane's value from the snapshot `v` (race-free, unlike a scalar
        // re-read whose slot another thread could have overwritten with a different key).
        let mut lanes = [0u64; TT_ASSOC_WAYS];
        _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, v);
        Some(Slot(lanes[hit.trailing_zeros() as usize]).val())
    }

    /// AVX-512 body of [`put_assoc_hashed`](Self::put_assoc_hashed): load the bucket, pick the
    /// way (refresh a used lane with matching fp, else first empty lane, else a route-selected
    /// lane to evict), and store. One vector load + two mask ops + a scalar store.
    ///
    /// # Safety
    /// As [`get_assoc_avx512`](Self::get_assoc_avx512): `b + TT_ASSOC_WAYS <= self.slots.len()`.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline]
    unsafe fn put_assoc_avx512(&self, b: usize, want: u64, route: u64, packed: u64) {
        use std::arch::x86_64::*;
        let v = _mm512_loadu_si512(self.slots.as_ptr().add(b) as *const __m512i);
        let fps = _mm512_srli_epi64::<{ Slot::FP_SHIFT }>(v);
        let used = _mm512_test_epi64_mask(v, _mm512_set1_epi64(1));
        // Only a *used* lane with matching fp is a refresh (an unused lane reads fp 0, which
        // would spuriously match `want == 0`); masking with `used` matches the scalar path.
        let fp_match = _mm512_cmpeq_epi64_mask(fps, _mm512_set1_epi64(want as i64)) & used;
        let way = if fp_match != 0 {
            fp_match.trailing_zeros()
        } else {
            let empty = !used; // __mmask8: lanes with the used bit clear
            if empty != 0 {
                empty.trailing_zeros()
            } else {
                // Bucket full: evict a route-selected way (low route bits — disjoint from the
                // high bits `fastrange` used to pick the bucket — so evictions spread).
                (route >> 3) as u32 & (TT_ASSOC_WAYS as u32 - 1)
            }
        };
        // SAFETY: `way < TT_ASSOC_WAYS`, so `b + way < len`.
        self.slots
            .get_unchecked(b + way as usize)
            .store(packed, Ordering::Relaxed);
    }

    /// Amortised assoc probe: one bucket scan returns **both** the lookup result **and** the
    /// slot a subsequent miss should store into. A node's `get` (entry) and `put` (exit) hit
    /// the same bucket, so the single `used`/`fp_match` scan that answers the lookup also yields
    /// the put-target way (refresh / first-empty / route-evict) for free — the caller threads
    /// the returned slot to [`store_slot`](Self::store_slot) at exit, turning the put into a
    /// bare store (no second scan). Halves the per-node bucket traffic to seg's 1 load + 1
    /// store. On a hit the second tuple field is unused (the caller returns before storing).
    // Parked substrate for a future amortised assoc-TT revival; not on a live path yet.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn probe_assoc(&self, route: u64, fp: u64, pc: u32) -> (Option<u8>, usize) {
        let b = self.bucket_base(route, pc);
        let want = fp & Slot::fp_mask();
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        // SAFETY: `bucket_base` guarantees `b + TT_ASSOC_WAYS <= len`.
        unsafe {
            self.probe_assoc_avx512(b, want, route)
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            let mut empty: usize = TT_ASSOC_WAYS;
            for i in 0..TT_ASSOC_WAYS {
                // SAFETY: `bucket_base` guarantees `b + i < len` for `i < TT_ASSOC_WAYS`.
                let s = Slot(unsafe { self.slots.get_unchecked(b + i) }.load(Ordering::Relaxed));
                if s.used() {
                    if s.fp() == want {
                        return (Some(s.val()), b + i);
                    }
                } else if empty == TT_ASSOC_WAYS {
                    empty = i;
                }
            }
            let way = if empty != TT_ASSOC_WAYS {
                empty
            } else {
                (route as usize >> 3) & (TT_ASSOC_WAYS - 1)
            };
            (None, b + way)
        }
    }

    /// Blind store to a slot chosen earlier by [`probe_assoc`](Self::probe_assoc) — the
    /// amortised put: no scan, no index recompute, just the write (write-allocate brings the
    /// line the probe already located).
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn store_slot(&self, slot: usize, fp: u64, val: u8) {
        // SAFETY: `slot` comes from `probe_assoc` = `bucket_base(..) + way`, with
        // `way < TT_ASSOC_WAYS` and `bucket_base + TT_ASSOC_WAYS <= len`, so `slot < len`.
        unsafe { self.slots.get_unchecked(slot) }.store(Slot::pack(fp, val).0, Ordering::Relaxed);
    }

    /// AVX-512 body of [`probe_assoc`](Self::probe_assoc): one `__m512i` load, then derive the
    /// hit value and the put-target slot from the same `used`/`fp_match` masks.
    ///
    /// # Safety
    /// As [`get_assoc_avx512`](Self::get_assoc_avx512): `b + TT_ASSOC_WAYS <= self.slots.len()`.
    #[allow(dead_code)]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline]
    unsafe fn probe_assoc_avx512(&self, b: usize, want: u64, route: u64) -> (Option<u8>, usize) {
        use std::arch::x86_64::*;
        let v = _mm512_loadu_si512(self.slots.as_ptr().add(b) as *const __m512i);
        let fps = _mm512_srli_epi64::<{ Slot::FP_SHIFT }>(v);
        let used = _mm512_test_epi64_mask(v, _mm512_set1_epi64(1));
        let fp_match = _mm512_cmpeq_epi64_mask(fps, _mm512_set1_epi64(want as i64)) & used;
        if fp_match != 0 {
            // Hit: extract the matching lane's value from the snapshot (race-free). The slot is
            // unused by the caller (it returns before the put), so report the matching way.
            let idx = fp_match.trailing_zeros() as usize;
            let mut lanes = [0u64; TT_ASSOC_WAYS];
            _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, v);
            return (Some(Slot(lanes[idx]).val()), b + idx);
        }
        // Miss: put target = first empty way, else a route-selected way to evict.
        let empty = !used;
        let way = if empty != 0 {
            empty.trailing_zeros()
        } else {
            (route >> 3) as u32 & (TT_ASSOC_WAYS as u32 - 1)
        };
        (None, b + way as usize)
    }

    /// [`prefetch_seg_hashed`](Self::prefetch_seg_hashed) for the assoc bucket: warm the
    /// bucket's first slot — its cache line covers all [`TT_ASSOC_WAYS`] ways.
    #[inline]
    pub(crate) fn prefetch_assoc_hashed(&self, route: u64, pc: u32) {
        let ptr = self.slots[self.bucket_base(route, pc)].as_ptr();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // SAFETY: as [`prefetch_hashed`](Self::prefetch_hashed) -- warms a valid
            // in-allocation pointer, no architectural effect, cannot fault.
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// Lemire's `fastrange`: map a 64-bit hash uniformly into `[0, len)` with a single
    /// widening multiply + shift -- the power-of-two-free replacement for `hash & mask`,
    /// so the table can be sized to any slot count (Chunk 2b). The extra multiply is
    /// negligible against the random-probe DRAM latency the search is bound by.
    #[inline]
    fn index(&self, route: u64) -> usize {
        ((route as u128).wrapping_mul(self.len as u128) >> 64) as usize
    }

    /// The slot a `route` hashes to, **without** the slice bounds check. `index` is Lemire
    /// fastrange `(route·len)>>64`, which is `< len` for every `route`, so the index is always in
    /// bounds — the `slots[index]` bounds check (a `cmp`+conditional-branch to a panic path) is
    /// provably dead. perf put it at ~8% of `wins_inc`'s hot ETC-prefetch loop (every `prefetch_h`/
    /// `get_hashed`/`put_hashed` paid it). Removing it cuts that branch + frees the length register.
    #[inline(always)]
    fn slot(&self, route: u64) -> &AtomicU64 {
        // SAFETY: `index(route) = (route·len)>>64 < len` for all `route` (fastrange maps a u64 into
        // `[0, len)`), so the index is always within `self.slots`.
        unsafe { self.slots.get_unchecked(self.index(route)) }
    }

    /// A table that also counts the distinct positions it is queried for: every
    /// `get` folds the (canonical) key into a HyperLogLog of precision `hll_p`,
    /// and (when `exact`) into a hash set for an exact ground truth on small
    /// boards. Used by the `count` CLI mode to size the table's true working set.
    pub fn new_counting(bits: u32, hll_p: u32, exact: bool) -> Self {
        let mut tt = Self::new(bits);
        tt.counter = Some(Counter {
            hll: Hll::new(hll_p),
            exact: exact.then(|| Mutex::new(HashMap::new())),
        });
        tt
    }

    /// Whether this table is carrying distinct-position instrumentation.
    #[inline]
    pub(crate) fn is_counting(&self) -> bool {
        self.counter.is_some()
    }

    /// The distinct-position measurement, if this table was built with counting.
    pub fn report(&self) -> Option<CountReport> {
        self.counter.as_ref().map(|c| CountReport {
            estimate: c.hll.estimate(),
            exact: c.exact.as_ref().map(|s| s.lock().unwrap().len() as u64),
            registers: c.hll.registers.len() as u64,
        })
    }

    /// The exact working set as (canonical key, win/loss value) pairs, if an exact
    /// map was kept (`count --exact`). Values are the exact ones recorded at `put`,
    /// not peeked from the lossy TT. Cold post-search analysis only (`--iso`).
    pub fn working_set(&self) -> Option<Vec<(Bits, u8)>> {
        let map = self.counter.as_ref()?.exact.as_ref()?.lock().unwrap();
        Some(map.iter().map(|(&k, &v)| (k, v)).collect())
    }

    /// Total slot capacity and its byte footprint, for reporting the cap.
    pub fn capacity(&self) -> (u64, u64) {
        let slots = self.slots.len() as u64;
        (slots, slots * std::mem::size_of::<AtomicU64>() as u64)
    }

    /// Occupied slots, by a one-time scan (post-solve; cheap relative to the
    /// search). Combined with [`capacity`](Self::capacity) it gives the load
    /// factor, and `nodes > fill` reveals how much eviction forced re-expansion.
    pub fn fill(&self) -> u64 {
        self.slots
            .iter()
            .filter(|s| Slot(s.load(Ordering::Relaxed)).used())
            .count() as u64
    }

    /// A "TT {GB}, {load}% full" fragment for the solve summary.
    pub fn summary(&self) -> String {
        let (slots, bytes) = self.capacity();
        let load = self.fill() as f64 / slots as f64 * 100.0;
        format!("TT {:.2} GB, {load:.1}% full", bytes as f64 / 1e9)
    }

    /// Nodes actually searched (TT misses) -- the work done, since hits are free.
    pub fn nodes(&self) -> u64 {
        self.nodes.load(Ordering::Relaxed)
    }

    /// Count one searched node (a TT miss about to be expanded) in this worker's local
    /// tally; once it has accumulated [`FLUSH_NODES`] of them (≈ once a second) push the
    /// tally into the shared atomic. The per-node path touches only thread-local memory --
    /// no shared atomic, no cross-CCX coherence.
    #[inline]
    pub fn bump(&self) {
        ACC.with(|cell| {
            let mut a = cell.borrow_mut();
            a.nodes += 1;
            if a.nodes >= self.flush_nodes {
                self.flush_acc(&mut a);
            }
        });
    }

    /// Count one searched node through a caller-owned local accumulator. This is for hot
    /// sequential recursion that can carry a `u64` down the stack and avoid the per-node
    /// thread-local `RefCell` access in [`Self::bump`]. It preserves progress reporting by
    /// flushing to the shared counter at the same cadence.
    #[inline]
    pub(crate) fn bump_local(&self, nodes: &mut u64) {
        *nodes += 1;
        if *nodes >= self.flush_nodes {
            self.flush_local_nodes(nodes);
        }
    }

    /// Flush a caller-owned local node accumulator into the shared counter, attributing the batch
    /// to the flushing rayon worker for live per-core telemetry (off the per-node path).
    #[inline]
    pub(crate) fn flush_local_nodes(&self, nodes: &mut u64) {
        if *nodes != 0 {
            self.nodes.fetch_add(*nodes, Ordering::Relaxed);
            let w = rayon::current_thread_index().unwrap_or(0).min(255);
            self.per_worker[w].fetch_add(*nodes, Ordering::Relaxed);
            *nodes = 0;
        }
    }

    /// Snapshot the per-rayon-worker node tallies (for the live per-core throughput display). The
    /// watcher samples these each tick and reports per-worker deltas; index = worker, 0-padded.
    pub fn per_worker_nodes(&self) -> Vec<u64> {
        self.per_worker
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect()
    }

    /// Push a worker's local tally into the shared atomic + HLL and reset it. Called once
    /// per [`FLUSH_NODES`] nodes and at drain -- off the per-node path. (Caller holds the
    /// thread-local borrow.)
    fn flush_acc(&self, a: &mut Acc) {
        if a.nodes > 0 {
            self.nodes.fetch_add(a.nodes, Ordering::Relaxed);
            a.nodes = 0;
        }
        if !a.hll.is_empty() {
            if let Some(c) = &self.counter {
                c.hll.merge_from(&a.hll);
            }
        }
    }

    /// Flush every rayon worker's accumulator into the shared state and clear their local
    /// estimators, so [`nodes`](Self::nodes) and the distinct report are exact after a
    /// parallel search (the hot loop flushes only ≈ once a second). Run after a parallel
    /// `first_player_wins`. A checkpoint mid-search captures a ~1-s-stale node count unless
    /// drained first -- fine for progress.
    pub fn drain_all(&self) {
        rayon::broadcast(|_| ACC.with(|cell| self.drain_acc(&mut cell.borrow_mut())));
        ACC.with(|cell| self.drain_acc(&mut cell.borrow_mut()));
    }

    /// Drain only the calling thread's accumulator (the sequential `wins` path).
    pub fn drain_local(&self) {
        ACC.with(|cell| self.drain_acc(&mut cell.borrow_mut()));
    }

    fn drain_acc(&self, a: &mut Acc) {
        self.flush_acc(a);
        // Local registers are kept by max between flushes; clear so a later solve in this
        // process does not inherit this solve's distinct keys.
        a.hll.iter_mut().for_each(|b| *b = 0);
    }

    /// A 128-bit hash of the key as two independent `u64` halves: `route` drives
    /// the shard (low bits) and slot index (high bits, disjoint); `fp` is the
    /// fingerprint stored in the slot. The halves use different seeds and mixing
    /// constants so the fingerprint actually discriminates keys that share a slot
    /// (rather than re-deriving bits the index already pinned). `route` reproduces
    /// the legacy hash exactly, preserving the routing distribution.
    #[inline]
    pub(crate) fn hash128(key: Bits) -> (u64, u64) {
        let mut route = 0u64;
        let mut fp = 0x2545_F491_4F6C_DD1Du64;
        for &w in &key.0 {
            route = (route ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            route ^= route >> 29;
            fp = (fp ^ w).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            fp ^= fp >> 32;
        }
        (route, fp)
    }

    /// As [`get`](Self::get)/[`put`](Self::put)/[`prefetch`](Self::prefetch)/
    /// [`archive_key`](Self::archive_key) but taking a pre-computed `(route, fp)` hash,
    /// so a caller that needs several of these for one key (the [`BurrStore`] tiers +
    /// the archive key) pays [`hash128`](Self::hash128) **once** instead of per call.
    /// These skip the distinct-counter hook (the `BurrStore` counts at its own level).
    #[inline]
    pub(crate) fn get_hashed(&self, route: u64, fp: u64) -> Option<u8> {
        let s = Slot(self.slot(route).load(Ordering::Relaxed));
        (s.used() && s.fp() == (fp & Slot::fp_mask())).then(|| s.val())
    }
    #[inline]
    pub(crate) fn put_hashed(&self, route: u64, fp: u64, val: u8) {
        self.slot(route)
            .store(Slot::pack(fp, val).0, Ordering::Relaxed);
    }
    #[inline]
    pub(crate) fn archive_key_hashed(&self, route: u64, fp: u64) -> u64 {
        archive_key_of(self.index(route) as u64, fp & Slot::fp_mask())
    }
    /// Tri-state probe for the ABDADA protocol: distinguishes a resolved verdict from an
    /// in-flight marker (a worker currently expanding this key) so the caller can defer rather
    /// than re-expand. A fingerprint *mismatch* (a hash-colliding other key) reads as a `Miss`,
    /// exactly as [`get_hashed`](Self::get_hashed) would — only a true key-match yields `Hit`/
    /// `InFlight`. One relaxed load, same as the plain get.
    #[inline]
    pub(crate) fn get_inflight_hashed(&self, route: u64, fp: u64) -> Probe3 {
        let s = Slot(self.slot(route).load(Ordering::Relaxed));
        if s.used() && s.fp() == (fp & Slot::fp_mask()) {
            if s.val() == Slot::IN_FLIGHT {
                Probe3::InFlight
            } else {
                Probe3::Hit(s.val())
            }
        } else {
            Probe3::Miss
        }
    }
    /// Claim a key as in-flight: store the [`Slot::IN_FLIGHT`] marker. A blind relaxed store
    /// (replace-always, as every put) — two workers racing to claim the same empty slot both
    /// expand once (the marker only *defers* concurrent probers, never blocks), and the
    /// completing [`put_hashed`](Self::put_hashed) overwrites the marker with the real verdict.
    /// Worst case the marker is evicted or lost to a race ⇒ a deferral is missed and the node is
    /// re-expanded (baseline behaviour); never a wrong verdict, since only the final put records
    /// a value and that value is deterministic.
    #[inline]
    pub(crate) fn mark_inflight_hashed(&self, route: u64, fp: u64) {
        self.slot(route)
            .store(Slot::pack(fp, Slot::IN_FLIGHT).0, Ordering::Relaxed);
    }
    #[inline]
    pub(crate) fn prefetch_hashed(&self, route: u64) {
        let ptr = self.slot(route).as_ptr();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // SAFETY: as [`prefetch`](Self::prefetch) -- warms a valid in-allocation
            // pointer, no architectural effect, cannot fault.
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// The stored value for `key`, if a slot's fingerprint matches.
    #[inline]
    pub fn get(&self, key: Bits) -> Option<u8> {
        // Counting hook: every node the search enters is looked up here exactly
        // once, so folding the key in on each `get` measures the distinct set of
        // positions visited -- the table's working set -- deduplicated by the
        // estimator regardless of transposition revisits or eviction.
        if let Some(c) = &self.counter {
            // Fold the key into this worker's *local* HLL registers (a plain byte max, no
            // atomic) -- merged into the shared estimator off the hot loop at flush/drain.
            ACC.with(|cell| {
                let mut a = cell.borrow_mut();
                if a.hll.len() != c.hll.register_count() {
                    a.hll = vec![0u8; c.hll.register_count()];
                }
                c.hll.add_local(key, &mut a.hll);
            });
        }
        let (route, fp) = Self::hash128(key);
        let raw = self.slot(route).load(Ordering::Relaxed);
        let s = Slot(raw);
        (s.used() && s.fp() == (fp & Slot::fp_mask())).then(|| s.val())
    }

    /// Store `val` for `key` (replace-always on collision).
    #[inline]
    pub fn put(&self, key: Bits, val: u8) {
        let (route, fp) = Self::hash128(key);
        self.slot(route)
            .store(Slot::pack(fp, val).0, Ordering::Relaxed);
        // Record the exact value for the post-search `--iso` analysis (cold; only
        // when an exact map is kept). Here the value is known and eviction-proof.
        if let Some(c) = &self.counter {
            c.record(key, val);
        }
    }

    /// [`get`](Self::get) with the `(route, fp)` of `key` precomputed by the caller
    /// (hash-carry): the hot search hashes each key **once** when it is created and reuses
    /// the halves for the prefetch, this lookup, and the eventual [`put_h`](Self::put_h),
    /// instead of re-deriving them via `hash128` at every touch. `key` is still threaded so
    /// the counting build can fold it into the HLL (a predicted-away null check otherwise).
    #[inline]
    pub fn get_h(&self, key: Bits, route: u64, fp: u64) -> Option<u8> {
        if let Some(c) = &self.counter {
            ACC.with(|cell| {
                let mut a = cell.borrow_mut();
                if a.hll.len() != c.hll.register_count() {
                    a.hll = vec![0u8; c.hll.register_count()];
                }
                c.hll.add_local(key, &mut a.hll);
            });
        }
        let s = Slot(self.slot(route).load(Ordering::Relaxed));
        (s.used() && s.fp() == (fp & Slot::fp_mask())).then(|| s.val())
    }

    /// [`put`](Self::put) with the `(route, fp)` of `key` precomputed (hash-carry twin of
    /// [`get_h`](Self::get_h)).
    #[inline]
    pub fn put_h(&self, key: Bits, route: u64, fp: u64, val: u8) {
        self.slot(route)
            .store(Slot::pack(fp, val).0, Ordering::Relaxed);
        if let Some(c) = &self.counter {
            c.record(key, val);
        }
    }

    /// [`prefetch`](Self::prefetch) with the route half precomputed (hash-carry).
    #[inline]
    pub fn prefetch_h(&self, route: u64) {
        let ptr = self.slot(route).as_ptr();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // SAFETY: as in `prefetch` -- warms a valid pointer into the live allocation.
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// As [`prefetch_h`](Self::prefetch_h) but with an **L2** hint (`_MM_HINT_T1`) instead of L1.
    /// Used for the long-distance gather-time prefetch of recurse children, which sort *last* in the
    /// degree-ordered descent: the intervening cheap-getK children stream the W8 arena and would
    /// evict an L1 (T0) line before the recurse arm is reached, but the working set fits L2, so a T1
    /// line survives — turning the recurse child's ~165-cyc cold DRAM probe into an ~L2 hit.
    #[inline]
    pub fn prefetch_h_t1(&self, route: u64) {
        let ptr = self.slot(route).as_ptr();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // SAFETY: as in `prefetch`.
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(ptr as *const i8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// Prefetch the slot `key` will land in, so the demand `get` that follows finds
    /// it warm -- overlapping the random-probe DRAM round-trip with the work in
    /// between (Session 5, L1 cluster). x86_64 only; a no-op elsewhere.
    #[inline]
    pub fn prefetch(&self, key: Bits) {
        let idx = self.index(Self::hash128(key).0);
        let ptr = self.slots[idx].as_ptr();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // SAFETY: `_mm_prefetch` only warms the cache for a valid pointer into
            // our live allocation; it has no architectural effect and cannot fault.
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// Stream this table as a raw image (`header || little-endian slot u64s`) to
    /// `w` (proposal Approach A). Each slot is read with a single relaxed atomic
    /// load, so a *live* dump under concurrent writers is a valid partial memo --
    /// each `u64` is never torn and every stored value is a final verdict, so the
    /// snapshot is sound to reload (good-enough-live; it only misses in-flight
    /// `put`s). `n` tags the board the image belongs to. The empty slots are zero,
    /// so the stream compresses well -- wrap `w` in a zstd encoder at the call site.
    pub fn dump_image<W: Write>(&self, w: &mut W, n: u8) -> io::Result<()> {
        self.dump_image_with(w, n, |_, _| {})
    }

    /// As [`dump_image`](Self::dump_image), but invoking `progress(slots_written,
    /// total_slots)` after each block -- so a CLI can paint a checkpoint progress bar.
    /// The slot count is the natural progress metric: the compressed byte size on disk
    /// is smaller and not known until the stream finishes. The callback runs on the
    /// dumping thread between block writes, so it must be cheap (and throttle its own
    /// output).
    pub fn dump_image_with<W: Write, F: FnMut(u64, u64)>(
        &self,
        w: &mut W,
        n: u8,
        mut progress: F,
    ) -> io::Result<()> {
        let header = TtHeader {
            n,
            len: self.len,
            fill: 0, // reporting-only; a full pre-scan every checkpoint isn't worth it
            epoch: 0,
            nodes: self.nodes.load(Ordering::Relaxed), // restored on --resume
        };
        w.write_all(&header.to_bytes())?;
        let mut buf = Vec::with_capacity(TT_IO_BLOCK * 8);
        let mut written = 0u64;
        for chunk in self.slots.chunks(TT_IO_BLOCK) {
            buf.clear();
            for slot in chunk {
                buf.extend_from_slice(&slot.load(Ordering::Relaxed).to_le_bytes());
            }
            w.write_all(&buf)?;
            written += chunk.len() as u64;
            progress(written, self.len);
        }
        Ok(())
    }

    /// Reload a raw image written by [`dump_image`](Self::dump_image) into a fresh
    /// table, hard-erroring if the header's format/hash/canon/arch tags or `n` don't
    /// match this build (a mismatch would silently void every hit). The table is
    /// sized to the image's `len` -- routing is `fastrange(route, len)`, so it cannot
    /// be re-keyed into a different size. `counter` is `None`; attach one with
    /// [`attach_counter`](Self::attach_counter) for a `--distinct` resume.
    pub fn load_image<R: Read>(r: &mut R, expected_n: u8) -> io::Result<QueensTt> {
        Self::load_image_with(r, expected_n, |_, _| {})
    }

    /// As [`load_image`](Self::load_image), but invoking `progress(slots_read,
    /// total_slots)` after each block so a CLI can paint a load progress bar -- the
    /// n=16 image is multi-GB and the decompress + zeroed-huge alloc commit takes a
    /// while. The node counter is restored from the header (so a resume's progress
    /// reflects the snapshot's prior work, not a fresh 0).
    pub fn load_image_with<R: Read, F: FnMut(u64, u64)>(
        r: &mut R,
        expected_n: u8,
        mut progress: F,
    ) -> io::Result<QueensTt> {
        let mut hbuf = [0u8; TT_HEADER_LEN];
        r.read_exact(&mut hbuf)?;
        let header = TtHeader::parse(&hbuf)?;
        if header.n != expected_n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "image is for n={}, but this run is n={expected_n}",
                    header.n
                ),
            ));
        }
        let size = header.len as usize;
        let slots = zeroed_huge_atomics(size);
        let mut buf = vec![0u8; TT_IO_BLOCK * 8];
        let mut i = 0usize;
        while i < size {
            let take = TT_IO_BLOCK.min(size - i);
            let bytes = &mut buf[..take * 8];
            r.read_exact(bytes)?;
            for (j, slot) in slots[i..i + take].iter().enumerate() {
                let word = u64::from_le_bytes(bytes[j * 8..j * 8 + 8].try_into().unwrap());
                slot.store(word, Ordering::Relaxed);
            }
            i += take;
            progress(i as u64, size as u64);
        }
        Ok(QueensTt {
            slots,
            len: header.len,
            nodes: AtomicU64::new(header.nodes),
            per_worker: (0..256).map(|_| AtomicU64::new(0)).collect(),
            flush_nodes: flush_nodes_env(),
            counter: None,
            // Resume does not support segmentation (the image is a flat-keyed snapshot).
            segment: false,
            band_base: Box::new([]),
            band_size: Box::new([]),
            assoc: false,
        })
    }

    /// Attach distinct-position instrumentation to an already-built table (e.g. a
    /// reloaded image), so a `--resume` run can still report its working set. See
    /// [`new_counting`](Self::new_counting).
    pub fn attach_counter(&mut self, hll_p: u32, exact: bool) {
        self.counter = Some(Counter {
            hll: Hll::new(hll_p),
            exact: exact.then(|| Mutex::new(HashMap::new())),
        });
    }

    /// Stream this *live* table's occupied entries as `(archive_key, val)` pairs --
    /// the in-memory freeze source for a [`BurrStore`](crate::queens::BurrStore)
    /// segment (the live twin of [`for_each_image_entry`], which streams a dump).
    /// Each slot is one relaxed atomic load; a concurrent writer may be missed or
    /// included, which only costs a later re-expansion (never a wrong value), so no
    /// lock is taken. The archive key matches [`archive_key`](Self::archive_key), so a
    /// live query resolves to the same entry this freeze stored.
    #[inline]
    pub fn for_each_entry<F: FnMut(u64, u8)>(&self, mut f: F) {
        for (idx, slot) in self.slots.iter().enumerate() {
            let s = Slot(slot.load(Ordering::Relaxed));
            if s.used() {
                f(archive_key_of(idx as u64, s.fp()), s.val());
            }
        }
    }

    /// Zero every slot (relaxed stores), returning the table to empty so it can be
    /// reused as a fresh memtable after a freeze. The node counter and any distinct
    /// counter are left untouched -- they are cumulative search state, not per-memtable.
    /// A concurrent `put` racing the clear is simply lost (re-expanded later) -- sound,
    /// never wrong.
    pub fn clear(&self) {
        for slot in self.slots.iter() {
            slot.store(0, Ordering::Relaxed);
        }
    }

    /// The BuRR archive key a live `key` resolves to in *this* table (Chunk 4).
    /// A frozen [`burr::Archive`](crate::burr::Archive) is keyed by the slot
    /// identity `(index, fingerprint)` recovered from a dump (see
    /// [`archive_key_of`]); querying it during search recomputes that pair from the
    /// position's canonical `key`. The archive **must** be frozen from a dump of a
    /// table with the same `len` -- the slot index is `fastrange(route, len)`, so a
    /// different size re-routes every key.
    #[inline]
    pub fn archive_key(&self, key: Bits) -> u64 {
        let (route, fp) = Self::hash128(key);
        archive_key_of(self.index(route) as u64, fp & Slot::fp_mask())
    }
}

/// Derive the BuRR archive key for a TT slot identity `(slot_index, fingerprint)`.
///
/// The dumped TT image stores only a 55-bit fingerprint per slot, not the position
/// key, so an archived entry is identified by the same pair the live table resolves
/// a position to: its slot **index** and its stored **fingerprint**. Two positions
/// sharing both already collide in the live TT (the accepted ~`2^-55` event), so
/// keying the archive on this pair reproduces the table's resolution exactly -- no
/// new merge loss. The query path recomputes the pair via [`QueensTt::archive_key`].
#[inline]
pub fn archive_key_of(slot_index: u64, fingerprint: u64) -> u64 {
    // Fold both halves through the mixer so neither dominates the low bits the
    // ribbon's start/coeff hashes consume.
    mix64(mix64(slot_index) ^ fingerprint.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
}

/// Stream a dumped [`QueensTt`] image, invoking `f(archive_key, val)` for each
/// occupied slot -- the freeze source for a BuRR [`burr::Archive`](crate::burr::Archive).
/// Validates the header (the same hard format/hash/canon/arch/`n` checks as
/// [`QueensTt::load_image`]) and returns it. Reads block by block, so it never
/// materialises the whole table -- a 17 GB n=16 dump streams in ~512 KB chunks,
/// which is what lets the freeze run on a box too small to also hold the table.
pub fn for_each_image_entry<R: Read, F: FnMut(u64, u8)>(
    r: &mut R,
    expected_n: u8,
    mut f: F,
) -> io::Result<TtHeader> {
    let mut hbuf = [0u8; TT_HEADER_LEN];
    r.read_exact(&mut hbuf)?;
    let header = TtHeader::parse(&hbuf)?;
    if header.n != expected_n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "image is for n={}, but this run is n={expected_n}",
                header.n
            ),
        ));
    }
    let size = header.len as usize;
    let mut buf = vec![0u8; TT_IO_BLOCK * 8];
    let mut idx = 0usize;
    while idx < size {
        let take = TT_IO_BLOCK.min(size - idx);
        let bytes = &mut buf[..take * 8];
        r.read_exact(bytes)?;
        for (j, word8) in bytes.chunks_exact(8).enumerate() {
            let s = Slot(u64::from_le_bytes(word8.try_into().unwrap()));
            if s.used() {
                f(archive_key_of((idx + j) as u64, s.fp()), s.val());
            }
        }
        idx += take;
    }
    Ok(header)
}

/// The proof-number table for [`Pn`]: a fixed-size sharded open-addressing table
/// keyed by canonical mask -> `(proof, disproof)` numbers. Same structure and
/// guarantees as [`QueensTt`] (collision = miss = re-expand, never wrong).
pub struct PnTt {
    shards: Vec<Mutex<Box<[PnSlot]>>>,
    shard_mask: u64,
    slot_mask: u64,
    nodes: AtomicU64,
}

#[derive(Clone, Copy, Default)]
struct PnSlot {
    key: [u64; WORDS],
    phi: u32,
    delta: u32,
    used: u8,
}

impl PnTt {
    pub fn new(bits: u32) -> Self {
        let bits = bits.max(SHARD_BITS);
        let shards = 1usize << SHARD_BITS;
        let per = 1usize << (bits - SHARD_BITS);
        PnTt {
            shards: (0..shards)
                .map(|_| Mutex::new(vec![PnSlot::default(); per].into_boxed_slice()))
                .collect(),
            shard_mask: shards as u64 - 1,
            slot_mask: per as u64 - 1,
            nodes: AtomicU64::new(0),
        }
    }

    pub fn capacity(&self) -> (u64, u64) {
        let slots = (self.shard_mask + 1) * (self.slot_mask + 1);
        (slots, slots * std::mem::size_of::<PnSlot>() as u64)
    }

    /// Occupied slots, by a one-time scan -- see [`QueensTt::fill`].
    pub fn fill(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| {
                s.lock()
                    .unwrap()
                    .iter()
                    .filter(|slot| slot.used != 0)
                    .count() as u64
            })
            .sum()
    }

    /// A "TT {GB}, {load}% full" fragment for the solve summary.
    pub fn summary(&self) -> String {
        let (slots, bytes) = self.capacity();
        let load = self.fill() as f64 / slots as f64 * 100.0;
        format!("TT {:.2} GB, {load:.1}% full", bytes as f64 / 1e9)
    }

    pub fn nodes(&self) -> u64 {
        self.nodes.load(Ordering::Relaxed)
    }

    #[inline]
    pub(crate) fn bump(&self) {
        self.nodes.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn get(&self, key: Bits) -> Option<(u32, u32)> {
        let h = QueensTt::hash128(key).0; // PnTt keeps the full key, so it needs only the routing half
        let idx = ((h >> 32) & self.slot_mask) as usize;
        let s = self.shards[(h & self.shard_mask) as usize].lock().unwrap()[idx];
        (s.used != 0 && s.key == key.0).then_some((s.phi, s.delta))
    }

    #[inline]
    pub(crate) fn put(&self, key: Bits, phi: u32, delta: u32) {
        let h = QueensTt::hash128(key).0;
        let idx = ((h >> 32) & self.slot_mask) as usize;
        self.shards[(h & self.shard_mask) as usize].lock().unwrap()[idx] = PnSlot {
            key: key.0,
            phi,
            delta,
            used: 1,
        };
    }
}

/// Standalone MLP latency-overlap microbench (step-0 for the batched-probe / idle-core
/// prefetch-prep lever): random-probe throughput into the real huge-page flat TT as a function
/// of the software-pipeline depth (probes kept in flight). It is the ceiling measurement for the
/// 176-cyc pc 13–21 DRAM probe — if throughput rises with depth, batching/prefetch-warming the
/// scattered probes recovers the exposed latency; if flat, one-ahead already saturates and MLP is
/// dead. `#[ignore]`d (timing, not a gate). Run:
///   RUSTFLAGS="-C target-cpu=znver5 -C link-arg=-fuse-ld=mold" \
///     cargo test --release --lib mlp_probe_depth_sweep -- --ignored --nocapture
/// Env: `MLP_BITS` (table size, default 30 = 8 GiB), `MLP_N` (probes, default 20M).
#[cfg(test)]
mod mlp_bench {
    use super::*;
    use std::time::Instant;

    #[inline]
    fn xs(s: &mut u64) -> u64 {
        let mut x = *s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *s = x;
        x
    }

    #[test]
    #[ignore = "timing microbench; run explicitly with --ignored --nocapture"]
    fn mlp_probe_depth_sweep() {
        let env = |k: &str, d: u64| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let bits = env("MLP_BITS", 30) as u32;
        let nprobe = env("MLP_N", 20_000_000) as usize;
        let tt = QueensTt::new(bits);
        // Prefault every page (sequential store) so the random probes below hit distinct DRAM
        // lines, not the shared zero page — otherwise an unfaulted slot would be a fake L1 hit.
        for slot in tt.slots.iter() {
            slot.store(1, Ordering::Relaxed);
        }
        // Random probe routes/fps (mostly miss; a miss costs the same DRAM load as a hit, so the
        // hit rate is irrelevant to the latency this isolates).
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut routes = vec![0u64; nprobe];
        let mut fps = vec![0u64; nprobe];
        for i in 0..nprobe {
            routes[i] = xs(&mut s);
            fps[i] = xs(&mut s);
        }
        println!(
            "[mlp] bits={bits} ({} slots, {:.1} GiB) nprobe={nprobe}",
            tt.len,
            (tt.len as f64 * 8.0) / (1u64 << 30) as f64,
        );
        // depth = probes kept in flight (prefetch issued `depth` iterations before its get).
        let sweep = |label: &str, routes: &[u64], fps: &[u64]| {
            for &depth in &[0usize, 1, 2, 4, 8, 16, 32] {
                let t = Instant::now();
                let mut acc = 0u64;
                for i in 0..nprobe + depth {
                    if i < nprobe {
                        tt.prefetch_hashed(routes[i]);
                    }
                    if i >= depth {
                        let j = i - depth;
                        acc += tt.get_hashed(routes[j], fps[j]).map_or(0, u64::from);
                    }
                }
                let el = t.elapsed();
                let ns = el.as_nanos() as f64 / nprobe as f64;
                let mps = nprobe as f64 / el.as_secs_f64() / 1e6;
                println!("[mlp] {label:<6} depth={depth:>2}  {ns:6.1} ns/probe  {mps:7.1} M/s  (acc={acc})");
            }
        };
        sweep("random", &routes, &fps);
        // Sort the same probes by *target slot* (`fastrange(route, len)`) → sequential access, the
        // DDD/streaming "dense aligned chunk" model. The random sweep above is still latency-bound at
        // depth-16 (~7 GB/s ≪ peak); this measures the bandwidth headroom sorting unlocks — the A''/DDD
        // ceiling. (Sort cost is excluded; in A'' the idle cores pay it off the critical path.)
        let len = tt.len as u128;
        let mut order: Vec<u32> = (0..nprobe as u32).collect();
        order.sort_by_key(|&i| ((routes[i as usize] as u128 * len) >> 64) as u64);
        let routes_s: Vec<u64> = order.iter().map(|&i| routes[i as usize]).collect();
        let fps_s: Vec<u64> = order.iter().map(|&i| fps[i as usize]).collect();
        sweep("sorted", &routes_s, &fps_s);
    }

    /// Phase-0 of the [sorted-frontier-wave proposal](../../notes/proposal-2026-06-20-sorted-frontier-wave.md):
    /// does the single-thread ~5.7× sorted-stream ceiling (random depth-0 → sorted depth-32 from
    /// [`mlp_probe_depth_sweep`]) **survive multi-thread bandwidth contention**? Approach B (idle-core
    /// producer/consumer pipeline) needs a few hot consumer cores to stream sorted chunks at the
    /// single-thread ceiling while other cores prep — so the open question is whether aggregate sorted
    /// throughput keeps scaling past 1 thread, or saturates the memory channels.
    ///
    /// Strong-scaling test: a **fixed** `MLP_N` probe set is partitioned across `nt` threads (each
    /// thread owns a contiguous, independently-sorted slice — matching A'' where each producer sorts
    /// its own frontier piece), all streaming the shared huge-page TT concurrently. Aggregate
    /// `M/s = total probes / wall`, so the curve over `nt` reads directly: rising ⇒ bandwidth headroom
    /// (the lever survives); flat ⇒ saturated (the multi-core lift is bounded). Reports `sorted/random`
    /// per thread count = the realizable multiplier under that contention.
    ///   RUSTFLAGS="-C target-cpu=znver5 -C link-arg=-fuse-ld=mold" \
    ///     cargo test --release --lib mlp_probe_threads_sweep -- --ignored --nocapture
    /// Env: `MLP_BITS` (table, default 30 = 8 GiB), `MLP_N` (total probes, default 20M),
    /// `MLP_THREADS` (csv, default "1,2,4,8"), `MLP_DEPTHS` (csv, default "1,16,32").
    #[test]
    #[ignore = "timing microbench; run explicitly with --ignored --nocapture"]
    fn mlp_probe_threads_sweep() {
        let env = |k: &str, d: u64| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let csv = |k: &str, d: &str| -> Vec<usize> {
            std::env::var(k)
                .unwrap_or_else(|_| d.to_string())
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        };
        let bits = env("MLP_BITS", 30) as u32;
        let nprobe = env("MLP_N", 20_000_000) as usize;
        let thread_counts = csv("MLP_THREADS", "1,2,4,8");
        let depths = csv("MLP_DEPTHS", "1,16,32");

        let tt = QueensTt::new(bits);
        // Prefault every page so random probes hit distinct DRAM lines, not the shared zero page.
        for slot in tt.slots.iter() {
            slot.store(1, Ordering::Relaxed);
        }
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut routes = vec![0u64; nprobe];
        let mut fps = vec![0u64; nprobe];
        for i in 0..nprobe {
            routes[i] = xs(&mut s);
            fps[i] = xs(&mut s);
        }
        let len = tt.len as u128;
        println!(
            "[mlp-t] bits={bits} ({:.1} GiB) nprobe={nprobe} threads={thread_counts:?} depths={depths:?}",
            (tt.len as f64 * 8.0) / (1u64 << 30) as f64,
        );

        // Run `nt` threads (one per pre-built slice), each streaming its slice at `depth`; return
        // aggregate M/s over all threads. Sort cost is excluded (pre-built), as in A'' (idle-core prep).
        let run = |routes_t: &[Vec<u64>], fps_t: &[Vec<u64>], depth: usize| -> f64 {
            let total: usize = routes_t.iter().map(Vec::len).sum();
            let t = Instant::now();
            std::thread::scope(|scope| {
                for (rt, ft) in routes_t.iter().zip(fps_t.iter()) {
                    let tt = &tt;
                    scope.spawn(move || {
                        let n = rt.len();
                        let mut acc = 0u64;
                        for i in 0..n + depth {
                            if i < n {
                                tt.prefetch_hashed(rt[i]);
                            }
                            if i >= depth {
                                let j = i - depth;
                                acc += tt.get_hashed(rt[j], ft[j]).map_or(0, u64::from);
                            }
                        }
                        std::hint::black_box(acc);
                    });
                }
            });
            total as f64 / t.elapsed().as_secs_f64() / 1e6
        };

        for &nt in &thread_counts {
            let chunk = nprobe / nt;
            // Partition the probe set into `nt` contiguous slices; `sorted` ⇒ sort each slice by slot.
            let mk = |sorted: bool| -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
                let mut rts = Vec::with_capacity(nt);
                let mut fts = Vec::with_capacity(nt);
                for t in 0..nt {
                    let lo = t * chunk;
                    let hi = if t + 1 == nt { nprobe } else { lo + chunk };
                    let mut idx: Vec<usize> = (lo..hi).collect();
                    if sorted {
                        idx.sort_by_key(|&i| ((routes[i] as u128 * len) >> 64) as u64);
                    }
                    rts.push(idx.iter().map(|&i| routes[i]).collect());
                    fts.push(idx.iter().map(|&i| fps[i]).collect());
                }
                (rts, fts)
            };
            let (r_rand, f_rand) = mk(false);
            let (r_sort, f_sort) = mk(true);
            for &depth in &depths {
                let rnd = run(&r_rand, &f_rand, depth);
                let srt = run(&r_sort, &f_sort, depth);
                println!(
                    "[mlp-t] threads={nt:>2} depth={depth:>2}  random {rnd:7.1} M/s  sorted {srt:7.1} M/s  sorted/random {:.2}x",
                    srt / rnd,
                );
            }
        }
    }
}
