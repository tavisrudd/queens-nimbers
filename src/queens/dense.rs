use rayon::prelude::*;
use std::sync::OnceLock;

// getK evaluator memo (`QUEENS_GETK_MEMO`) — MEASURED-NEGATIVE, reverted. A thread-local exact-fp
// memo over get12..get16 (to collapse the recursion's factorial path-redundancy) cost +13.4%
// cyc/node at n=16: α-β cutoffs already prune most redundancy, and the memo's random L2/L3 probe
// is slower than just recomputing the pext/popcnt math. "Math is cheaper than mem" on this
// latency-bound search — the same reason the W_K hierarchy is memo-less by design.

/// Largest dense layer the `getK` evaluators reach. The complete tables stop at
/// [`W8_K`]; `W9..=W{MAX_DENSE_K}` are evaluators over the layer below (see [`DenseW8::get9`]).
/// W9..W11 keep their labelled code (`K*(K-1)/2` ≤ 55 bits) in a `u64`; W12 (66-bit), W13
/// (78-bit), W14 (91-bit), W15 (105-bit) and W16 (120-bit) run on a `u128` two-word `pext`
/// path. W13..W16 children can be 12..15-vertex (66..105-bit) codes, so their projection uses
/// the `u128`-returning [`pext128_wide`]. K=16 is the `u128` ceiling for the labelled code
/// (16·15/2 = 120 ≤ 128); K=17 = 136 bits would need a wider code. The adjacency rows stay
/// `≤15` bits (`u64`), so [`pext128`] still recovers them at every K.
pub(crate) const MAX_DENSE_K: usize = 16;
const W8_K: usize = 8;
const W12_K: usize = 12;
const W13_K: usize = 13;
const W14_K: usize = 14;
const W15_K: usize = 15;
const W16_K: usize = 16;
/// `u64`-path induced table size: every W9..W11 child is a `≤11`-bit alive mask.
const MAX_INDUCED: usize = 1 << 11;
/// `u128`-path induced table sizes (W12's alive mask is 12-bit, W13's is 13-bit).
const W12_INDUCED: usize = 1 << W12_K;
const W13_INDUCED: usize = 1 << W13_K;
const W14_INDUCED: usize = 1 << W14_K;
const W15_INDUCED: usize = 1 << W15_K;
const W16_INDUCED: usize = 1 << W16_K;

/// Per-vertex incident masks and per-alive-subset induced masks for the `k`-vertex
/// upper-triangular edge layout. `incident[i]` selects the code bits of every edge
/// touching vertex `i` (used to recover `adj[i]` via `pext`); `induced[alive]` selects
/// the bits of every edge whose endpoints both lie in `alive` (a `k`-bit subset).
/// Because `pext` preserves bit order, projecting a sub-subset's `induced` mask yields
/// the relabelled subgraph's canonical upper-triangular code directly — the property
/// that lets a `k`-vertex child code feed straight into `W{child_pc}` (table or `getK`).
/// Entries past `k`/`2^k` stay zero (the consts size to `MAX_*` so all layers share a type).
const fn wk_masks(k: usize) -> ([u64; MAX_DENSE_K], [u64; MAX_INDUCED]) {
    let mut incident = [0u64; MAX_DENSE_K];
    let mut induced = [0u64; MAX_INDUCED];
    let mut bit = 0usize;
    let mut i = 0usize;
    while i < k {
        let mut j = i + 1;
        while j < k {
            let edge = 1u64 << bit;
            incident[i] |= edge;
            incident[j] |= edge;
            let mut alive = 0usize;
            while alive < (1usize << k) {
                if (alive & (1 << i)) != 0 && (alive & (1 << j)) != 0 {
                    induced[alive] |= edge;
                }
                alive += 1;
            }
            bit += 1;
            j += 1;
        }
        i += 1;
    }
    (incident, induced)
}

// The W8 incident/induced masks let the *table build* recover adjacency + project children with
// `pext` (the runtime `get9..` machinery), instead of the scalar `adj_from_code`/`projected_code`
// per code — the k=8 build is 2^28 codes and dominates startup prep.
const W8_MASKS: ([u64; MAX_DENSE_K], [u64; MAX_INDUCED]) = wk_masks(8);
const W9_MASKS: ([u64; MAX_DENSE_K], [u64; MAX_INDUCED]) = wk_masks(9);
const W10_MASKS: ([u64; MAX_DENSE_K], [u64; MAX_INDUCED]) = wk_masks(10);
const W11_MASKS: ([u64; MAX_DENSE_K], [u64; MAX_INDUCED]) = wk_masks(11);

/// [`wk_masks`] for the `u128` layers (`k` ≥ 12, code > 64 bits): the incident/induced masks
/// are `u128` and the induced table is `2^k = INDUCED` entries. `incident` is sized to
/// `MAX_DENSE_K` so all the W≥12 layers share the [`extract_adj128`] signature.
///
/// The induced table is built **incrementally** over subsets in increasing order:
/// `induced[alive] = induced[alive \ {h}] | edges(h, alive \ {h})` where `h` is `alive`'s top
/// vertex — O(2^k · popcount) instead of the naïve O(2^k · k²/2) edge×subset scan. That keeps the
/// const-eval under the `long_running_const_eval` limit up to k=16 (the `u128` ceiling), where the
/// naïve form (7.9 M steps) trips it. Result is identical (`direct_wK_matches_scalar_recurrence`).
const fn wk_masks128<const INDUCED: usize>(k: usize) -> ([u128; MAX_DENSE_K], [u128; INDUCED]) {
    let mut incident = [0u128; MAX_DENSE_K];
    // `ebit[i][j]` (i<j) = the code-bit position of edge (i,j) in the upper-triangular layout.
    let mut ebit = [[0u8; MAX_DENSE_K]; MAX_DENSE_K];
    let mut bit = 0u32;
    let mut i = 0usize;
    while i < k {
        let mut j = i + 1;
        while j < k {
            let edge = 1u128 << bit;
            incident[i] |= edge;
            incident[j] |= edge;
            ebit[i][j] = bit as u8;
            bit += 1;
            j += 1;
        }
        i += 1;
    }
    let mut induced = [0u128; INDUCED];
    let mut alive = 1usize;
    while alive < (1usize << k) {
        let h = (usize::BITS - 1 - alive.leading_zeros()) as usize; // top set vertex
        let without_h = alive & !(1usize << h);
        let mut acc = induced[without_h];
        let mut rest = without_h;
        while rest != 0 {
            let v = rest.trailing_zeros() as usize; // v < h, an edge (v,h)
            rest &= rest - 1;
            acc |= 1u128 << (ebit[v][h] as u32);
        }
        induced[alive] = acc;
        alive += 1;
    }
    (incident, induced)
}

const W12_MASKS: ([u128; MAX_DENSE_K], [u128; W12_INDUCED]) = wk_masks128::<W12_INDUCED>(W12_K);
const W13_MASKS: ([u128; MAX_DENSE_K], [u128; W13_INDUCED]) = wk_masks128::<W13_INDUCED>(W13_K);
const W14_MASKS: ([u128; MAX_DENSE_K], [u128; W14_INDUCED]) = wk_masks128::<W14_INDUCED>(W14_K);
const W15_MASKS: ([u128; MAX_DENSE_K], [u128; W15_INDUCED]) = wk_masks128::<W15_INDUCED>(W15_K);
const W16_MASKS: ([u128; MAX_DENSE_K], [u128; W16_INDUCED]) = wk_masks128::<W16_INDUCED>(W16_K);

// ===== Wide layers W17..W20: 17..20-vertex graphs (137..190-bit code, 3 words). =====
// The `u128` ceiling is K=16 (120 bits); K=17..20 = 136..190 edge bits need a 3-word code.
// Children of a K-vertex node have ≤K-1 vertices: a ≤16-vertex child (≤120 bits) feeds the
// existing `get16..get9`/`get` machinery (`u128`); a 17..19-vertex child stays wide (3-word) and
// recurses via [`DenseW8::get_dyn_wide`]. The adjacency rows are ≤K-1 ≤19 bits (`u32`), and the
// alive/child subset masks are K-bit (the index into the induced table), so the sweep uses `u32`,
// not the `u16` of the ≤16 machinery. The induced mask tables (2^K × 3 words: 3 MiB..24 MiB) are
// built at RUNTIME — const-eval can't afford them past K=16.
const MAX_WIDE_K: usize = 20;
type Code192 = [u64; 3];

/// Per-vertex incident masks for a `k`-vertex wide layer (`const`, padded to `MAX_WIDE_K`):
/// `incident[i]` selects the 3-word-code bits of every edge touching vertex `i`, so
/// `pext(code, incident[i])` packs `adj[i]` into the low bits.
const fn wide_incident(k: usize) -> [Code192; MAX_WIDE_K] {
    let mut inc = [[0u64; 3]; MAX_WIDE_K];
    let mut bit = 0usize;
    let mut i = 0;
    while i < k {
        let mut j = i + 1;
        while j < k {
            inc[i][bit >> 6] |= 1u64 << (bit & 63);
            inc[j][bit >> 6] |= 1u64 << (bit & 63);
            bit += 1;
            j += 1;
        }
        i += 1;
    }
    inc
}
const W17_INCIDENT: [Code192; MAX_WIDE_K] = wide_incident(17);
const W18_INCIDENT: [Code192; MAX_WIDE_K] = wide_incident(18);
const W19_INCIDENT: [Code192; MAX_WIDE_K] = wide_incident(19);
const W20_INCIDENT: [Code192; MAX_WIDE_K] = wide_incident(20);

/// Build the runtime induced-mask table for a `k`-vertex wide layer: `induced[alive]` selects the
/// 3-word-code bits of every edge with both endpoints in `alive` (a `k`-bit subset), so
/// `pext(code, induced[alive])` yields the relabelled subgraph's canonical upper-triangular code.
/// Built incrementally over subsets in increasing order — `induced[alive] = induced[alive\{h}] |
/// edges(h, alive\{h})` (`h` = top set vertex) — O(2^k · popcount), the `wk_masks128` trick. ~3..24
/// MiB; called once per used `k` via [`wide_induced`].
fn build_wide_induced(k: usize) -> Box<[Code192]> {
    // `ebit[i][j]` (i<j) = the code-bit position of edge (i,j) in the upper-triangular layout.
    let mut ebit = [[0u16; MAX_WIDE_K]; MAX_WIDE_K];
    let mut bit = 0u16;
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for j in (i + 1)..k {
            ebit[i][j] = bit;
            bit += 1;
        }
    }
    let n = 1usize << k;
    let mut induced = vec![[0u64; 3]; n].into_boxed_slice();
    for alive in 1..n {
        let h = (usize::BITS - 1 - alive.leading_zeros()) as usize; // top set vertex
        let without_h = alive & !(1usize << h);
        let mut acc = induced[without_h];
        let mut rest = without_h;
        while rest != 0 {
            let v = rest.trailing_zeros() as usize; // v < h, edge (v,h)
            rest &= rest - 1;
            let b = ebit[v][h] as usize;
            acc[b >> 6] |= 1u64 << (b & 63);
        }
        induced[alive] = acc;
    }
    collapse_huge(&induced);
    induced
}

/// Huge-back a fully-built hot table: `MADV_HUGEPAGE` + a synchronous `MADV_COLLAPSE` on the
/// page-aligned interior. The W8 arena (32 MB) and the wide induced tables (3..24 MB) are
/// random-accessed on every getK leaf; on 4 KB pages each lookup risks a dTLB walk (measured
/// ~8 dTLB misses/node at n=16 with the TT already huge-backed), while 2 MB pages cover a
/// whole table with a handful of dTLB entries. The tables are fully populated by their build
/// before this is called, so no prefault is needed (unlike the lazily-zeroed TT). Advisory —
/// any failure (old kernel, THP off, misalignment) is ignored. `QUEENS_DENSE_HUGE=0` disables
/// (the A/B control). Prep-time only; the search itself is byte-identical.
fn collapse_huge<T>(slice: &[T]) {
    #[cfg(target_os = "linux")]
    {
        if matches!(std::env::var("QUEENS_DENSE_HUGE").as_deref(), Ok("0")) {
            return;
        }
        let base = slice.as_ptr() as usize;
        let bytes = std::mem::size_of_val(slice);
        // SAFETY: advisory page-backing hints on the page-aligned interior of our own live
        // allocation (same alignment dance as `zeroed_huge_atomics` — a `Vec` pointer is not
        // page-aligned, and an unaligned start EINVALs); contents are never read or written.
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE).max(1) as usize;
            let aligned = base.next_multiple_of(page);
            let off = aligned - base;
            if bytes > off {
                const MADV_COLLAPSE: libc::c_int = 25; // Linux 6.1+
                let ptr = aligned as *mut libc::c_void;
                let len = bytes - off;
                libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
                libc::madvise(ptr, len, MADV_COLLAPSE);
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = slice;
}

/// `&'static` induced-mask table for wide layer `k` (17..20), built once (`OnceLock` per `k`).
#[inline]
fn wide_induced(k: usize) -> &'static [Code192] {
    static W17: OnceLock<Box<[Code192]>> = OnceLock::new();
    static W18: OnceLock<Box<[Code192]>> = OnceLock::new();
    static W19: OnceLock<Box<[Code192]>> = OnceLock::new();
    static W20: OnceLock<Box<[Code192]>> = OnceLock::new();
    match k {
        17 => W17.get_or_init(|| build_wide_induced(17)),
        18 => W18.get_or_init(|| build_wide_induced(18)),
        19 => W19.get_or_init(|| build_wide_induced(19)),
        _ => W20.get_or_init(|| build_wide_induced(20)),
    }
}

/// Pre-build the wide induced tables up to the ceiling `k` (idempotent). Called once from
/// `new_dense` when the dense ceiling reaches K≥17, so the ~ms..tens-of-ms builds happen at
/// startup, not on a hot getK.
pub(crate) fn warm_wide(k: usize) {
    for kk in 17..=k.min(MAX_WIDE_K) {
        let _ = wide_induced(kk);
    }
}

/// Three-word BMI2 `pext` returning a `u64` — for a wide adjacency row (≤19 selected bits).
#[inline]
fn pext192_u64(code: &Code192, mask: &Code192) -> u64 {
    use std::arch::x86_64::_pext_u64;
    // SAFETY: production is built with target-cpu=znver5, which includes BMI2.
    unsafe {
        let l = _pext_u64(code[0], mask[0]);
        let m = _pext_u64(code[1], mask[1]);
        let h = _pext_u64(code[2], mask[2]);
        let nl = mask[0].count_ones();
        let nm = mask[1].count_ones();
        l | (m << nl) | (h << (nl + nm))
    }
}

/// Deposit the low bits of `val` into a 3-word code at bit offset `off` (≤ ~190), handling the
/// 64-bit word straddle. Used to stitch the three `pext` halves of a wide child code.
#[inline]
fn or_at(out: &mut Code192, val: u64, off: u32) {
    if val == 0 {
        return;
    }
    let w = (off >> 6) as usize;
    let b = off & 63;
    out[w] |= val << b;
    if b != 0 && w + 1 < 3 {
        out[w + 1] |= val >> (64 - b);
    }
}

/// Three-word BMI2 `pext` returning a 3-word code — for a wide child (17..19-vertex, ≤171 bits).
#[inline]
fn pext192_u192(code: &Code192, mask: &Code192) -> Code192 {
    use std::arch::x86_64::_pext_u64;
    // SAFETY: as `pext192_u64`.
    unsafe {
        let l = _pext_u64(code[0], mask[0]);
        let m = _pext_u64(code[1], mask[1]);
        let h = _pext_u64(code[2], mask[2]);
        let nl = mask[0].count_ones();
        let nm = mask[1].count_ones();
        let mut out = [0u64; 3];
        or_at(&mut out, l, 0);
        or_at(&mut out, m, nl);
        or_at(&mut out, h, nl + nm);
        out
    }
}

/// Recover the `K` adjacency rows (`u32`, ≤K-1 bits) from a wide 3-word `code`. Twin of
/// [`extract_adj`]: `pext` packs the edges touching vertex `i` into the low bits, then the
/// self-gap is re-inserted at bit `i` so `adj[i] & (1<<j)` is set iff edge `(i,j)` exists.
#[inline]
fn extract_adj_wide<const K: usize>(code: &Code192, incident: &[Code192; MAX_WIDE_K]) -> [u32; 20] {
    let mut adj = [0u32; 20];
    #[allow(clippy::needless_range_loop)]
    for i in 0..K {
        let packed = pext192_u64(code, &incident[i]) as u32;
        let below = (1u32 << i) - 1;
        adj[i] = (packed & below) | ((packed & !below) << 1);
    }
    adj
}

/// Sweep order for a degree-ordered getK (`QUEENS_GETK_ORD`): vertex indices `0..K` sorted by
/// degree DESCENDING (highest first ⇒ smallest child ⇒ most-forcing ⇒ earliest cutoff). `deg[i]`
/// is the precomputed degree of vertex `i` (`adj[i].count_ones()`, ≤ K-1 ≤ 16). A stable counting
/// sort over the 0..K degree buckets — no comparison branch (the same branchless shape as the
/// recurse path's `sort_moves_by_degree` win). Returns the order in `order[..K]`.
#[inline]
fn deg_order_desc<const K: usize>(deg: &[u8; 21]) -> [u8; 21] {
    let mut cnt = [0u8; 22];
    for &d in &deg[..K] {
        cnt[d as usize] += 1;
    }
    // Descending: a degree-d vertex is placed after every higher-degree vertex.
    let mut start = [0u8; 22];
    let mut acc = 0u8;
    let mut d = 21usize;
    while d > 0 {
        d -= 1;
        start[d] = acc;
        acc += cnt[d];
    }
    let mut order = [0u8; 21];
    #[allow(clippy::needless_range_loop)]
    for i in 0..K {
        let d = deg[i] as usize;
        order[start[d] as usize] = i as u8;
        start[d] += 1;
    }
    order
}

/// Identity sweep order `[0,1,..,20]` — the label-order (`ord` off) path, so the getK sweep is a
/// single array-driven loop (no body duplication); the extra per-child order load is negligible
/// and cancels in the `QUEENS_GETK_ORD` A/B (both arms drive the same loop).
const IDENTITY21: [u8; 21] = {
    let mut a = [0u8; 21];
    let mut i = 0;
    while i < 21 {
        a[i] = i as u8;
        i += 1;
    }
    a
};

/// The getK sweep order for the `u16`-adjacency layers (get9..get16): degree-descending when
/// `ord` (`QUEENS_GETK_ORD`), else identity (label order). Returns the order in `[..K]`.
#[inline]
fn getk_order_u16<const K: usize>(ord: bool, adj: &[u16; MAX_DENSE_K]) -> [u8; 21] {
    if ord {
        let mut deg = [0u8; 21];
        #[allow(clippy::needless_range_loop)]
        for i in 0..K {
            deg[i] = adj[i].count_ones() as u8;
        }
        deg_order_desc::<K>(&deg)
    } else {
        IDENTITY21
    }
}

/// [`getk_order_u16`] for the wide `u32`-adjacency layers (get17..get20).
#[inline]
fn getk_order_u32<const K: usize>(ord: bool, adj: &[u32; 20]) -> [u8; 21] {
    if ord {
        let mut deg = [0u8; 21];
        #[allow(clippy::needless_range_loop)]
        for i in 0..K {
            deg[i] = adj[i].count_ones() as u8;
        }
        deg_order_desc::<K>(&deg)
    } else {
        IDENTITY21
    }
}

/// Two-word BMI2 `pext` over a `≤128`-bit `code`/`mask`. The low and high 64-bit halves
/// are extracted independently, then the high half's bits are shifted above the low half's
/// (`pext` preserves order, and the high-word edges hold the higher labelled-code bit
/// positions). Every call site extracts `≤ 55` bits (a `≤11`-vertex child code) or `≤ 11`
/// bits (an adjacency row), so the result fits `u64` and `lo_bits < 64`.
#[inline]
fn pext128(code: u128, mask: u128) -> u64 {
    use std::arch::x86_64::_pext_u64;
    // SAFETY: production is built with target-cpu=znver5, which includes BMI2.
    let lo = unsafe { _pext_u64(code as u64, mask as u64) };
    let hi = unsafe { _pext_u64((code >> 64) as u64, (mask >> 64) as u64) };
    let lo_bits = (mask as u64).count_ones();
    debug_assert!(lo_bits < 64);
    lo | (hi << lo_bits)
}

/// [`pext128`] for a child code that can exceed 64 bits: a 12-vertex child of a 13-vertex
/// node has a 66-bit code. The low half can hold up to 64 selected bits, so `lo_bits` may
/// be 64 and the high half is shifted within the `u128` (no `u64` overflow as in `pext128`).
#[inline]
fn pext128_wide(code: u128, mask: u128) -> u128 {
    use std::arch::x86_64::_pext_u64;
    // SAFETY: production is built with target-cpu=znver5, which includes BMI2.
    let lo = unsafe { _pext_u64(code as u64, mask as u64) } as u128;
    let hi = unsafe { _pext_u64((code >> 64) as u64, (mask >> 64) as u64) } as u128;
    let lo_bits = (mask as u64).count_ones();
    lo | (hi << lo_bits)
}

/// One-ahead software prefetch of the next child's induced-mask entry in a getK sweep.
/// The deep layers' masks tables (W12 64 KB .. W16 1 MB, wide K=17..20 up to 24 MB) miss
/// L1d/L2, and that load heads each iteration's serial chain (mask → pext → nested
/// evaluator) — the n=16 cache-miss profile concentrates in get13..16 on exactly this load.
/// Issued one child ahead so the current child's evaluation hides the miss; an early cutoff
/// wastes at most one prefetched line.
#[inline(always)]
fn prefetch_mask<T>(entry: &T) {
    // SAFETY: prefetch is a hint with no architectural effect; the reference is a valid address.
    unsafe {
        std::arch::x86_64::_mm_prefetch(
            entry as *const T as *const i8,
            std::arch::x86_64::_MM_HINT_T0,
        )
    }
}

/// [`extract_adj`] for a `u128` code (W12/W13): same self-gap re-insertion, but the adjacency
/// row (`≤ K-1 ≤ 12` bits, fits `u64`) is recovered with [`pext128`].
#[inline]
fn extract_adj128<const K: usize>(
    code: u128,
    incident: &[u128; MAX_DENSE_K],
) -> ([u16; MAX_DENSE_K], u16) {
    let mut adj = [0u16; MAX_DENSE_K];
    let mut iso = 0u16; // bits where adj[i]==0 (isolated vertices) — fused for the pair-strip
    for i in 0..K {
        let packed = pext128(code, incident[i]) as u16;
        let below = (1u16 << i) - 1;
        let a = (packed & below) | ((packed & !below) << 1);
        adj[i] = a;
        iso |= ((a == 0) as u16) << i;
    }
    (adj, iso)
}

/// Recover the `K` adjacency rows from a labelled upper-triangular edge `code`.
/// `incident[i]` packs the edges touching vertex `i` into the low bits via `pext`;
/// re-inserting the self-gap at bit `i` gives `adj[i]` with `adj[i] & (1<<j)` set iff
/// edge `(i,j)` exists. `K` is `const` so the loop unrolls (this is on the hot pc==K path).
#[inline]
fn extract_adj<const K: usize>(
    code: u64,
    incident: &[u64; MAX_DENSE_K],
) -> ([u16; MAX_DENSE_K], u16) {
    use std::arch::x86_64::_pext_u64;
    let mut adj = [0u16; MAX_DENSE_K];
    let mut iso = 0u16; // bits where adj[i]==0 (isolated vertices) — fused for the pair-strip
    for i in 0..K {
        // SAFETY: production is built with target-cpu=znver5, which includes BMI2.
        let packed = unsafe { _pext_u64(code, incident[i]) } as u16;
        let below = (1u16 << i) - 1;
        let a = (packed & below) | ((packed & !below) << 1);
        adj[i] = a;
        iso |= ((a == 0) as u16) << i;
    }
    (adj, iso)
}

/// Isolated-vertex **pair-strip**: a zero-degree vertex is a `K1` component with Grundy value 1
/// (the only move empties it), so any *two* of them cancel in the Sprague-Grundy sum —
/// `g(G ⊔ 2·isolated) = g(G)` — and removing an even number is exactly win/loss-preserving. From
/// the `K` adjacency rows, return `(nrem, keep)` with `nrem` the even count of isolated vertices to
/// drop and `keep` the `K`-bit survivor mask, or `None` if `<2` isolated (no reduction). This
/// collapses the deep "peel one isolated vertex per ply" recursion the `getK` sweep would otherwise
/// do into a single smaller `getK` call (the reduced graph has `≤1` isolated vertex left).
/// MEASURED NET-NEGATIVE, gated off (`const` ⇒ DCE ⇒ production byte-identical to no-strip).
/// The K1 pair-strip fires too rarely to pay: getK peels isolated vertices *one ply at a time*, so a
/// level seldom has ≥2 isolated verts at once to cancel; the `iso_strip` check on every getK call then
/// costs more than the rare firings save (n=16 A/B: +3.4% cyc/node). The common *1-isolated* case needs
/// the core's nimber (`g(core ⊔ {v}) = g(core) XOR 1`), not boolean win/loss — i.e. the parked
/// component-nimber branch. Kept as substrate for that revival / a clique-pair generalisation (K2/K3).
const ISO_STRIP: bool = false;

#[inline]
fn iso_strip<const K: usize>(iso: u16) -> Option<(usize, u16)> {
    if !ISO_STRIP {
        return None; // gated off: inlines to None ⇒ the strip + the fused `iso` in extract_adj DCE.
    }
    let nrem = (iso.count_ones() & !1) as usize; // largest even ≤ count (leave ≤1)
    if nrem < 2 {
        return None;
    }
    let mut removed = 0u16;
    let mut r = iso;
    for _ in 0..nrem {
        let b = r & r.wrapping_neg(); // lowest set bit
        removed |= b;
        r ^= b;
    }
    let full = if K >= 16 { u16::MAX } else { (1u16 << K) - 1 };
    Some((nrem, full & !removed))
}

#[inline]
const fn slots(k: usize) -> usize {
    // `k < 2` ⇒ a single-slot table (the empty / single-vertex graph); guard the `k - 1`
    // so const-eval of [`table_offsets`] doesn't underflow at `k == 0`.
    let edges = if k < 2 { 0 } else { k * (k - 1) / 2 };
    1usize << edges
}

#[inline]
const fn words(k: usize) -> usize {
    slots(k).div_ceil(64)
}

/// Word offset of each `W{k}` table inside the flat arena (cumulative [`words`]); the final
/// entry `TABLE_OFF[W8_K + 1]` is the arena's total word count. The complete `W0..=W8` tables
/// are concatenated into ONE contiguous `&'static [u64]` so a leaf lookup is a single load —
/// `arena[TABLE_OFF[k] + code/64]` — instead of the `&[Box<[u64]>]` double indirection
/// (load the box fat-pointer, then the word) plus its bounds-check `len` load. Every `getK`
/// evaluator bottoms out in [`DenseW8::get`], so this shortens the critical chain of the whole
/// dense bucket (~36 % of search cycles).
/// Padded to 16 entries (10..15 = 0) so the hot [`DenseW8::get`] can index with `k & 15` —
/// provably in-bounds, which drops the `cmp/jae panic_bounds_check` pair from every getK
/// inner-loop iteration (the sweep loops are frontend-bound; the dead branch costs fetch).
/// A padding hit is impossible under the caller invariant (`k ≤ W8_K`, debug-asserted), and
/// even then offset 0 keeps the arena access in-bounds for any `code < slots(8)`.
const fn table_offsets() -> [usize; 16] {
    let mut o = [0usize; 16];
    let mut k = 0;
    while k <= W8_K {
        o[k + 1] = o[k] + words(k);
        k += 1;
    }
    o
}
const TABLE_OFF: [usize; 16] = table_offsets();

#[inline]
fn bit_get(bits: &[u64], idx: usize) -> bool {
    (bits[idx >> 6] & (1u64 << (idx & 63))) != 0
}

fn adj_from_code(k: usize, code: u128) -> [u16; MAX_DENSE_K] {
    let mut adj = [0u16; MAX_DENSE_K];
    let mut bit = 0usize;
    for i in 0..k {
        for j in (i + 1)..k {
            if (code >> bit) & 1 != 0 {
                adj[i] |= 1u16 << j;
                adj[j] |= 1u16 << i;
            }
            bit += 1;
        }
    }
    adj
}

fn projected_code(adj: &[u16; MAX_DENSE_K], alive: u16) -> (usize, u128) {
    let k = alive.count_ones() as usize;
    let mut verts = [0usize; MAX_DENSE_K];
    let mut n = 0usize;
    let mut rem = alive;
    while rem != 0 {
        let v = rem.trailing_zeros() as usize;
        rem &= rem - 1;
        verts[n] = v;
        n += 1;
    }

    // `u128`: a 12-vertex child (of a 13-vertex node) has a 66-bit code.
    let mut code = 0u128;
    let mut bit = 0usize;
    for x in 0..k {
        let vx = verts[x];
        for &vy in verts.iter().take(k).skip(x + 1) {
            code |= (((adj[vx] >> vy) & 1) as u128) << bit;
            bit += 1;
        }
    }
    (k, code)
}

fn graph_wins(k: usize, code: usize, tables: &[Box<[u64]>]) -> bool {
    let adj = adj_from_code(k, code as u128);
    let full = (1u16 << k) - 1;
    for i in 0..k {
        let child = full & !((1u16 << i) | adj[i]);
        let (ck, ccode) = projected_code(&adj, child);
        // Only reached for `k ≤ 8` (build-time), where the child code is `≤ 21` bits.
        if !bit_get(&tables[ck], ccode as usize) {
            return true;
        }
    }
    false
}

/// `pext` twin of `graph_wins` specialised to `k == 8` — the dominant build cost (2^28 codes,
/// ~all of startup prep). Recovers adjacency via [`extract_adj`] and projects each child code with
/// one BMI2 `pext` (the same machinery the runtime [`DenseW8::get9`] uses), replacing the scalar
/// `adj_from_code` (28-bit decode) + `projected_code` (per-child nested scan) — ~2× the codes/s.
/// Result is bit-identical to `graph_wins(8, code, _)` (asserted in `graph_wins8_matches_scalar`).
#[inline]
fn graph_wins8(code: u64, tables: &[Box<[u64]>]) -> bool {
    use std::arch::x86_64::_pext_u64;
    let (adj, _) = extract_adj::<8>(code, &W8_MASKS.0);
    let full = (1u16 << 8) - 1;
    // `i` is both the removed-vertex bit (`1 << i`) and the `adj[i]` index, so the range loop
    // is the natural form (mirrors `get9`).
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        let child = full & !((1u16 << i) | adj[i]);
        // SAFETY: target-cpu=znver5 ⇒ BMI2. `induced[child]` (child < 256) selects the
        // sub-graph's edge bits in upper-triangular order, so `pext` yields its canonical code.
        let child_code = unsafe { _pext_u64(code, W8_MASKS.1[child as usize]) } as usize;
        if !bit_get(&tables[child.count_ones() as usize], child_code) {
            return true;
        }
    }
    false
}

/// Recursive scalar reference for any `k` (the slow ground truth `getK` is validated
/// against). Bottoms out in the complete `W{≤8}` tables and recurses one ply per layer
/// above, exactly mirroring `W_K(G) = ∃v · ¬W_{K-1}(G∖N[v])` — no `pext`, no mask tables.
#[cfg(test)]
fn wins_rec(k: usize, code: u128, tables: &[Box<[u64]>]) -> bool {
    if k <= W8_K {
        return bit_get(&tables[k], code as usize);
    }
    let adj = adj_from_code(k, code);
    let full = ((1u32 << k) - 1) as u16; // u16 form; (1u16 << 16) would overflow at k=16
    for i in 0..k {
        let child = full & !((1u16 << i) | adj[i]);
        let (ck, ccode) = projected_code(&adj, child);
        if !wins_rec(ck, ccode, tables) {
            return true;
        }
    }
    false
}

fn build_table(k: usize, tables: &[Box<[u64]>]) -> Box<[u64]> {
    (0..words(k))
        .into_par_iter()
        .map(|word| {
            let mut out = 0u64;
            let base = word << 6;
            let limit = slots(k).saturating_sub(base).min(64);
            for b in 0..limit {
                // `k` is loop-invariant so the branch hoists; k==8 (the 2^28-code dominant table)
                // takes the fast `pext` path, the tiny k≤7 tables keep the scalar reference.
                let win = if k == 8 {
                    graph_wins8((base + b) as u64, tables)
                } else {
                    graph_wins(k, base + b, tables)
                };
                if win {
                    out |= 1u64 << b;
                }
            }
            out
        })
        .collect()
}

/// Complete win/loss tables for all labelled Node-Kayles graphs up to 8 vertices.
///
/// A row is the 28-bit upper-triangular edge code for vertices labelled `0..8`.
/// The value is invariant under relabelling, so callers may use any deterministic
/// labelling of the 8 live board squares and still get the correct result.
/// `QUEENS_W8_CACHE` on-disk image of the flat `W0..=W8` arena. The table is a pure function
/// (Node-Kayles win/loss of every labelled ≤8-vertex config) so it is build-once / cache-forever.
/// Bump the last magic byte whenever [`build_tables`] changes (stale-cache guard); the length +
/// FNV checksum reject a truncated/corrupt file (⇒ silent rebuild). Header: magic(4) · len_u64(8) ·
/// checksum(8) = 20 bytes, then `len` little-endian u64s.
const W8_CACHE_MAGIC: &[u8; 4] = b"QW81";

fn w8_checksum(arena: &[u64]) -> u64 {
    arena.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
        (h ^ x).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Load the flat arena from `path`, or `None` if absent / wrong magic-version / wrong length /
/// checksum mismatch (any of which falls back to a fresh build).
fn load_w8_cache(path: &str, expected_len: usize) -> Option<Box<[u64]>> {
    let data = std::fs::read(path).ok()?;
    if data.len() != 20 + expected_len * 8 || &data[0..4] != W8_CACHE_MAGIC {
        return None;
    }
    let len = u64::from_le_bytes(data[4..12].try_into().ok()?) as usize;
    let want_ck = u64::from_le_bytes(data[12..20].try_into().ok()?);
    if len != expected_len {
        return None;
    }
    let mut arena = vec![0u64; len].into_boxed_slice();
    for (slot, chunk) in arena.iter_mut().zip(data[20..].chunks_exact(8)) {
        *slot = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    (w8_checksum(&arena) == want_ck).then_some(arena)
}

/// Best-effort write of the flat arena to `path` (errors ignored — a failed write just means the
/// next run rebuilds).
fn save_w8_cache(path: &str, arena: &[u64]) {
    let mut buf = Vec::with_capacity(20 + arena.len() * 8);
    buf.extend_from_slice(W8_CACHE_MAGIC);
    buf.extend_from_slice(&(arena.len() as u64).to_le_bytes());
    buf.extend_from_slice(&w8_checksum(arena).to_le_bytes());
    for &x in arena {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    let _ = std::fs::write(path, buf);
}

pub(crate) struct DenseW8 {
    /// The complete `W0..=W8` win/loss bitsets concatenated into one contiguous allocation,
    /// indexed via [`TABLE_OFF`]. One flat `&'static [u64]` (not `&[Box<[u64]>]`) so the hot
    /// leaf [`get`](Self::get) is a single load with no pointer-chase or bounds check.
    arena: &'static [u64],
    /// `QUEENS_GETK_ORD=1`: sweep the getK children **degree-descending** (highest-degree vertex
    /// first ⇒ smallest child ⇒ most-forcing ⇒ earliest α-β cutoff) instead of label order. The
    /// recurse path already degree-sorts its moves (the −30% dynamic-ordering win); the dense
    /// evaluators historically swept unordered, so a winning child is found later and more
    /// (expensive) sibling subtrees are evaluated first. Resolved once at build; a predictable
    /// run-constant branch in the hot getK. Off ⇒ byte-identical label-order sweep.
    ord_getk: bool,
    /// Resolved-once `&'static` wide induced-mask table for the K=17 ceiling — mirrors [`arena`]:
    /// every pc==17 getK node read it via a per-call `wide_induced(17)` — an `OnceLock`-load, a `match`,
    /// and a `Box` deref (the `Vec<Box<…>>` pointer-chase the W8 flat-arena removed for −2.0%). Held here
    /// so `get17` is a direct field read. (get18..20, behind `QUEENS_WK`, still call `wide_induced`.)
    w17_induced: &'static [Code192],
}

impl DenseW8 {
    pub(crate) fn build() -> Self {
        static ARENA: OnceLock<Box<[u64]>> = OnceLock::new();
        let arena: &'static [u64] = ARENA.get_or_init(|| {
            let expected_len = TABLE_OFF[W8_K + 1];
            // QUEENS_W8_CACHE=<path>: load the build-once arena from disk (~ms) instead of rebuilding
            // it (~1.5s of the pre-search prep). Opt-in (default unset ⇒ build, unchanged); on a miss
            // (absent / stale magic / bad checksum) it builds then writes the cache for next time.
            let cache = std::env::var("QUEENS_W8_CACHE").ok();
            if let Some(ref path) = cache {
                if let Some(loaded) = load_w8_cache(path, expected_len) {
                    return loaded;
                }
            }
            let tables = build_tables();
            let mut flat = vec![0u64; expected_len].into_boxed_slice();
            for k in 0..=W8_K {
                let off = TABLE_OFF[k];
                flat[off..off + tables[k].len()].copy_from_slice(&tables[k]);
            }
            if let Some(ref path) = cache {
                save_w8_cache(path, &flat);
            }
            flat
        });
        // Idempotent (re-advising a collapsed range is a cheap no-op) — covers both the fresh
        // build and the `QUEENS_W8_CACHE` load path without restructuring the init closure.
        collapse_huge(arena);
        DenseW8 {
            arena,
            // ★ Default-ON (--18, promoted): the degree-ordered getK sweep is part of the FAST default.
            // Disabled by `QUEENS_GETK_ORD=0` or the whole-stack revert `QUEENS_FAST=0` (the A/B control).
            // Only matters for iso-dense (dense_k≥12 reaches the ordered get12+ layers); iso-window/
            // iso-flat never call get12+ so they are byte-identical regardless.
            ord_getk: !matches!(std::env::var("QUEENS_GETK_ORD").as_deref(), Ok("0"))
                && !matches!(std::env::var("QUEENS_FAST").as_deref(), Ok("0")),
            // Resolve the K=17 wide induced table once (idempotent get_or_init) so `get17` reads a
            // field, not a per-node `wide_induced(17)` OnceLock+Box deref. iso-window builds a 3 MB
            // table it never uses — negligible (non-default control; iso-flat has no DenseW8 at all).
            w17_induced: wide_induced(17),
        }
    }

    #[inline]
    pub(crate) fn get(&self, k: usize, code: usize) -> bool {
        debug_assert!(k <= W8_K);
        debug_assert!(code < slots(k));
        // SAFETY: every caller passes a child `code` of a `k`-vertex subgraph (`code < slots(k)`),
        // so `TABLE_OFF[k] + code/64 < TABLE_OFF[k + 1] <= arena.len()`. The flat arena removes the
        // `Box<[u64]>` fat-pointer load + bounds-check `len` load that `&[Box<[u64]>]` indexing
        // emitted on this hot leaf (the bottom of every getK sweep).
        let w = TABLE_OFF[k & 15] + (code >> 6);
        (unsafe { *self.arena.get_unchecked(w) } >> (code & 63)) & 1 != 0
    }

    /// Dispatch to the right `getK` for a runtime `k` — used by the isolated-vertex pair-strip's
    /// tail-call (the reduced graph has `≤1` isolated vertex, so the target `getK` won't recurse
    /// here again). `getK` for `k ≤ 11` take a `u64`; the reduced code always fits its layer.
    #[inline]
    pub(crate) fn get_dyn(&self, k: usize, code: u128) -> bool {
        match k {
            16 => self.get16(code),
            15 => self.get15(code),
            14 => self.get14(code),
            13 => self.get13(code),
            12 => self.get12(code),
            11 => self.get11(code as u64),
            10 => self.get10(code as u64),
            9 => self.get9(code as u64),
            _ => self.get(k, code as usize),
        }
    }

    /// Exact value of one labelled 9-vertex graph, computed directly from the
    /// complete W0..W8 tables. `code` is the 36-bit upper-triangular edge code.
    /// BMI2 extracts adjacency rows and every induced child code arithmetically;
    /// there is no W9 allocation, canonicalisation, hash lookup, or join. Every child
    /// drops `1+deg(v)` vertices, so it lands in `W{≤8}` — one table lookup each.
    #[inline]
    pub(crate) fn get9(&self, code: u64) -> bool {
        use std::arch::x86_64::_pext_u64;

        debug_assert!(code < (1u64 << 36));
        let (adj, iso) = extract_adj::<9>(code, &W9_MASKS.0);
        if let Some((nrem, keep)) = iso_strip::<9>(iso) {
            // SAFETY: znver5 ⇒ BMI2. Project onto the isolated-free survivors (pext preserves order).
            let rcode = unsafe { _pext_u64(code, W9_MASKS.1[keep as usize]) };
            return self.get_dyn(9 - nrem, rcode as u128);
        }
        let full = (1u16 << 9) - 1;
        // NO degree-ordering here: get9's children are cheap W≤8 lookups, so the counting-sort +
        // popcount overhead exceeds the cutoff savings (measured: the sort dominated get9's profile).
        // Ordering only pays from get14 up, where children are expensive nested evaluators.
        #[allow(clippy::needless_range_loop)]
        for i in 0..9 {
            let child = full & !((1u16 << i) | adj[i]);
            // `cpc` off the `child`→popcount critical path: a move removes `{i} ∪ N[i]`
            // (disjoint, ⊆ full), so `cpc = 9 - 1 - deg(i)`. `popcount(adj[i])` depends only on
            // `adj[i]`, so it issues in parallel with `child` — the W[cpc] table index is ready
            // before `child` finishes, shortening the chain to the arena load.
            let cpc = (8 - adj[i].count_ones()) as usize;
            // SAFETY: same BMI2 build invariant as `extract_adj`. Extracted edges retain
            // upper-triangle order, so the result directly indexes W[popcount].
            let child_code = unsafe { _pext_u64(code, W9_MASKS.1[child as usize]) } as usize;
            if !self.get(cpc, child_code) {
                return true;
            }
        }
        false
    }

    /// Exact value of one labelled 10-vertex graph (45-bit `code`) — the W10 layer of the
    /// `W_K(G) = ∃v · ¬W_{K-1}(G∖N[v])` hierarchy. A move drops `1+deg(v)` vertices, so a
    /// child has `9-deg(v)` ≤ 9 vertices: an isolated vertex (deg 0) yields a 9-vertex child
    /// resolved by one nested [`get9`], every other child lands in `W{≤8}` (a direct table
    /// lookup). No allocation, no TT traffic, no re-expansion below pc==10.
    #[inline]
    pub(crate) fn get10(&self, code: u64) -> bool {
        let (adj, iso) = extract_adj::<10>(code, &W10_MASKS.0);
        self.get10_adj(code, &adj, iso)
    }

    /// [`get10`] with the adjacency rows (and fused iso bits) already known — the pc==10
    /// `w10_get` root computes every row for the code build anyway (`adj_row_pext`), so the
    /// root call skips the [`extract_adj`] re-derivation (one `pext` + gap-reinsert per row).
    /// Nested (child) calls still enter via [`get10`]: a child only has its projected code.
    #[inline]
    pub(crate) fn get10_adj(&self, code: u64, adj: &[u16; MAX_DENSE_K], iso: u16) -> bool {
        use std::arch::x86_64::_pext_u64;

        debug_assert!(code < (1u64 << 45));
        if let Some((nrem, keep)) = iso_strip::<10>(iso) {
            // SAFETY: znver5 ⇒ BMI2. Project onto the isolated-free survivors.
            let rcode = unsafe { _pext_u64(code, W10_MASKS.1[keep as usize]) };
            return self.get_dyn(10 - nrem, rcode as u128);
        }
        let full = (1u16 << 10) - 1;
        // NO degree-ordering (see get9): children are cheap, the sort overhead exceeds the savings.
        #[allow(clippy::needless_range_loop)]
        for i in 0..10 {
            let child = full & !((1u16 << i) | adj[i]);
            // `cpc = 10 - 1 - deg(i)` off the `child`→popcount chain (see get9): `popcount(adj[i])`
            // issues in parallel with `child`, so the `cpc==9` branch + table dispatch resolve early.
            let cpc = (9 - adj[i].count_ones()) as usize;
            // SAFETY: same BMI2 build invariant as `extract_adj`; the projected code is the
            // child's canonical upper-triangular code (pext preserves edge order).
            let child_code = unsafe { _pext_u64(code, W10_MASKS.1[child as usize]) };
            let lost = if cpc == 9 {
                !self.get9(child_code)
            } else {
                !self.get(cpc, child_code as usize)
            };
            if lost {
                return true;
            }
        }
        false
    }

    /// Exact value of one labelled 11-vertex graph (55-bit `code`) — the W11 layer. A child
    /// has `10-deg(v)` ≤ 10 vertices, resolved by a nested [`get10`]/[`get9`] (one ply each)
    /// or a direct `W{≤8}` lookup. The recursion bottoms out in the complete tables, so it is
    /// bounded depth (≤ 2 nested levels) with no allocation, TT traffic, or re-expansion.
    #[inline]
    pub(crate) fn get11(&self, code: u64) -> bool {
        let (adj, iso) = extract_adj::<11>(code, &W11_MASKS.0);
        self.get11_adj(code, &adj, iso)
    }

    /// [`get11`] with precomputed adjacency rows (see [`get10_adj`]).
    #[inline]
    pub(crate) fn get11_adj(&self, code: u64, adj: &[u16; MAX_DENSE_K], iso: u16) -> bool {
        use std::arch::x86_64::_pext_u64;

        debug_assert!(code < (1u64 << 55));
        if let Some((nrem, keep)) = iso_strip::<11>(iso) {
            // SAFETY: znver5 ⇒ BMI2. Project onto the isolated-free survivors.
            let rcode = unsafe { _pext_u64(code, W11_MASKS.1[keep as usize]) };
            return self.get_dyn(11 - nrem, rcode as u128);
        }
        let full = (1u16 << 11) - 1;
        // NO degree-ordering (see get9): children are cheap, the sort overhead exceeds the savings.
        #[allow(clippy::needless_range_loop)]
        for i in 0..11 {
            let child = full & !((1u16 << i) | adj[i]);
            // `cpc = 11 - 1 - deg(i)` off the `child`→popcount chain (see get9).
            let cpc = (10 - adj[i].count_ones()) as usize;
            // SAFETY: same BMI2 build invariant as `extract_adj`.
            let child_code = unsafe { _pext_u64(code, W11_MASKS.1[child as usize]) };
            let lost = match cpc {
                10 => !self.get10(child_code),
                9 => !self.get9(child_code),
                _ => !self.get(cpc, child_code as usize),
            };
            if lost {
                return true;
            }
        }
        false
    }

    /// Exact value of one labelled 12-vertex graph — the W12 layer, the first past the
    /// `u64` code ceiling, so `code` is a 66-bit value in a `u128`. A child has `≤11`
    /// vertices (its code `≤55` bits, fits `u64`), resolved by a nested get11/get10/get9 or
    /// a `W{≤8}` lookup. [`pext128`] does every two-word projection; [`extract_adj128`]
    /// recovers the rows. No allocation, no TT traffic, no re-expansion below pc==12.
    #[inline]
    pub(crate) fn get12(&self, code: u128) -> bool {
        let (adj, iso) = extract_adj128::<12>(code, &W12_MASKS.0);
        self.get12_adj(code, &adj, iso)
    }

    /// [`get12`] with precomputed adjacency rows (see [`get10_adj`]).
    #[inline]
    pub(crate) fn get12_adj(&self, code: u128, adj: &[u16; MAX_DENSE_K], iso: u16) -> bool {
        debug_assert!(code < (1u128 << 66));
        if let Some((nrem, keep)) = iso_strip::<12>(iso) {
            return self.get_dyn(12 - nrem, pext128_wide(code, W12_MASKS.1[keep as usize]));
        }
        let full = (1u16 << 12) - 1;
        let order = getk_order_u16::<12>(self.ord_getk, adj);
        for j in 0..12 {
            let i = order[j] as usize;
            if j + 1 < 12 {
                // One-ahead mask prefetch (see `prefetch_mask`); `& 15` proves the adj index.
                let ni = order[j + 1] as usize & 15;
                let nchild = full & !((1u16 << ni) | adj[ni]);
                prefetch_mask(&W12_MASKS.1[nchild as usize]);
            }
            let child = full & !((1u16 << i) | adj[i]);
            // `cpc = 12 - 1 - deg(i)` off the `child`→popcount chain (see get9).
            let cpc = (11 - adj[i].count_ones()) as usize;
            let child_code = pext128(code, W12_MASKS.1[child as usize]);
            let lost = match cpc {
                11 => !self.get11(child_code),
                10 => !self.get10(child_code),
                9 => !self.get9(child_code),
                _ => !self.get(cpc, child_code as usize),
            };
            if lost {
                return true;
            }
        }
        false
    }

    /// Exact value of one labelled 13-vertex graph — the W13 layer (78-bit `code` in a
    /// `u128`). A child has `≤12` vertices: a 12-vertex child (isolated removed vertex) has a
    /// 66-bit code resolved by a nested [`get12`] (via [`pext128_wide`]); every smaller child
    /// has a `≤55`-bit code (`u64`) → get11/get10/get9 or a W≤8 lookup. Bounded-depth recursion
    /// into the complete tables, no allocation, no TT traffic, no re-expansion below pc==13.
    #[inline]
    pub(crate) fn get13(&self, code: u128) -> bool {
        let (adj, iso) = extract_adj128::<13>(code, &W13_MASKS.0);
        self.get13_adj(code, &adj, iso)
    }

    /// [`get13`] with precomputed adjacency rows (see [`get10_adj`]).
    #[inline]
    pub(crate) fn get13_adj(&self, code: u128, adj: &[u16; MAX_DENSE_K], iso: u16) -> bool {
        debug_assert!(code < (1u128 << 78));
        if let Some((nrem, keep)) = iso_strip::<13>(iso) {
            return self.get_dyn(13 - nrem, pext128_wide(code, W13_MASKS.1[keep as usize]));
        }
        let full = (1u16 << 13) - 1;
        let order = getk_order_u16::<13>(self.ord_getk, adj);
        for j in 0..13 {
            let i = order[j] as usize;
            if j + 1 < 13 {
                // One-ahead mask prefetch (see `prefetch_mask`); `& 15` proves the adj index.
                let ni = order[j + 1] as usize & 15;
                let nchild = full & !((1u16 << ni) | adj[ni]);
                prefetch_mask(&W13_MASKS.1[nchild as usize]);
            }
            let child = full & !((1u16 << i) | adj[i]);
            // `cpc = 13 - 1 - deg(i)` off the `child`→popcount chain (see get9).
            let cpc = (12 - adj[i].count_ones()) as usize;
            let mask = W13_MASKS.1[child as usize];
            // Right-size the child code (see `get16`): only a 12-vertex child (66-bit) needs the
            // `u128` `pext128_wide`; the ≤11-vertex majority uses the cheaper `u64` `pext128`.
            let lost = if cpc == 12 {
                !self.get12(pext128_wide(code, mask))
            } else {
                let cc = pext128(code, mask);
                match cpc {
                    11 => !self.get11(cc),
                    10 => !self.get10(cc),
                    9 => !self.get9(cc),
                    _ => !self.get(cpc, cc as usize),
                }
            };
            if lost {
                return true;
            }
        }
        false
    }

    /// Exact value of one labelled 14-vertex graph — the W14 layer (91-bit `code` in a `u128`,
    /// the `u128` ceiling for the labelled code). A child has `≤13` vertices: a 13-vertex child
    /// (78-bit) → nested [`get13`], a 12-vertex child (66-bit) → [`get12`] (both `>64` bits, via
    /// [`pext128_wide`]); every smaller child has a `≤55`-bit code (`u64`) → get11/get10/get9 or a
    /// W≤8 lookup. Bounded-depth recursion into the complete tables, no allocation, no TT traffic,
    /// no re-expansion below pc==14.
    #[inline]
    pub(crate) fn get14(&self, code: u128) -> bool {
        let (adj, iso) = extract_adj128::<14>(code, &W14_MASKS.0);
        self.get14_adj(code, &adj, iso)
    }

    /// [`get14`] with precomputed adjacency rows (see [`get10_adj`]).
    #[inline]
    pub(crate) fn get14_adj(&self, code: u128, adj: &[u16; MAX_DENSE_K], iso: u16) -> bool {
        debug_assert!(code < (1u128 << 91));
        if let Some((nrem, keep)) = iso_strip::<14>(iso) {
            return self.get_dyn(14 - nrem, pext128_wide(code, W14_MASKS.1[keep as usize]));
        }
        let full = (1u16 << 14) - 1;
        let order = getk_order_u16::<14>(self.ord_getk, adj);
        for j in 0..14 {
            let i = order[j] as usize;
            if j + 1 < 14 {
                // One-ahead mask prefetch (see `prefetch_mask`); `& 15` proves the adj index.
                let ni = order[j + 1] as usize & 15;
                let nchild = full & !((1u16 << ni) | adj[ni]);
                prefetch_mask(&W14_MASKS.1[nchild as usize]);
            }
            let child = full & !((1u16 << i) | adj[i]);
            // `cpc = 14 - 1 - deg(i)` off the `child`→popcount chain (see get9).
            let cpc = (13 - adj[i].count_ones()) as usize;
            let mask = W14_MASKS.1[child as usize];
            // Right-size the child code (see `get16`): `u64` `pext128` for the ≤11-vertex majority;
            // only a 12/13-vertex child (isolated removal, >64 bits) needs the `u128` `pext128_wide`.
            let lost = if cpc >= 12 {
                let cc = pext128_wide(code, mask);
                if cpc == 13 {
                    !self.get13(cc)
                } else {
                    !self.get12(cc)
                }
            } else {
                let cc = pext128(code, mask);
                match cpc {
                    11 => !self.get11(cc),
                    10 => !self.get10(cc),
                    9 => !self.get9(cc),
                    _ => !self.get(cpc, cc as usize),
                }
            };
            if lost {
                return true;
            }
        }
        false
    }

    /// Exact value of one labelled 15-vertex graph — the W15 layer (105-bit `code` in a `u128`).
    /// A child has `≤14` vertices: a 14-vertex child (91-bit) → nested [`get14`], a 13-vertex →
    /// [`get13`], a 12-vertex → [`get12`] (all `>64` bits, via [`pext128_wide`]); smaller children
    /// have a `≤55`-bit code (`u64`). Bounded-depth recursion into the complete tables.
    #[inline]
    pub(crate) fn get15(&self, code: u128) -> bool {
        let (adj, iso) = extract_adj128::<15>(code, &W15_MASKS.0);
        self.get15_adj(code, &adj, iso)
    }

    /// [`get15`] with precomputed adjacency rows (see [`get10_adj`]).
    #[inline]
    pub(crate) fn get15_adj(&self, code: u128, adj: &[u16; MAX_DENSE_K], iso: u16) -> bool {
        debug_assert!(code < (1u128 << 105));
        if let Some((nrem, keep)) = iso_strip::<15>(iso) {
            return self.get_dyn(15 - nrem, pext128_wide(code, W15_MASKS.1[keep as usize]));
        }
        let full = (1u16 << 15) - 1;
        let order = getk_order_u16::<15>(self.ord_getk, adj);
        for j in 0..15 {
            let i = order[j] as usize;
            if j + 1 < 15 {
                // One-ahead mask prefetch (see `prefetch_mask`); `& 15` proves the adj index.
                let ni = order[j + 1] as usize & 15;
                let nchild = full & !((1u16 << ni) | adj[ni]);
                prefetch_mask(&W15_MASKS.1[nchild as usize]);
            }
            let child = full & !((1u16 << i) | adj[i]);
            // `cpc = 15 - 1 - deg(i)` off the `child`→popcount chain (see get9).
            let cpc = (14 - adj[i].count_ones()) as usize;
            let mask = W15_MASKS.1[child as usize];
            // Right-size the child code (see `get16`): `u64` `pext128` for the ≤11-vertex majority.
            let lost = if cpc >= 12 {
                let cc = pext128_wide(code, mask);
                match cpc {
                    14 => !self.get14(cc),
                    13 => !self.get13(cc),
                    _ => !self.get12(cc),
                }
            } else {
                let cc = pext128(code, mask);
                match cpc {
                    11 => !self.get11(cc),
                    10 => !self.get10(cc),
                    9 => !self.get9(cc),
                    _ => !self.get(cpc, cc as usize),
                }
            };
            if lost {
                return true;
            }
        }
        false
    }

    /// Exact value of one labelled 16-vertex graph — the W16 layer (120-bit `code`, the `u128`
    /// ceiling). A child has `≤15` vertices: 15/14/13/12-vertex children (105..66-bit) nest into
    /// [`get15`]/[`get14`]/[`get13`]/[`get12`] via [`pext128_wide`]; smaller children fit `u64`.
    #[inline]
    pub(crate) fn get16(&self, code: u128) -> bool {
        let (adj, iso) = extract_adj128::<16>(code, &W16_MASKS.0);
        self.get16_adj(code, &adj, iso)
    }

    /// [`get16`] with precomputed adjacency rows (see [`get10_adj`]).
    #[inline]
    pub(crate) fn get16_adj(&self, code: u128, adj: &[u16; MAX_DENSE_K], iso: u16) -> bool {
        debug_assert!(code < (1u128 << 120));
        if let Some((nrem, keep)) = iso_strip::<16>(iso) {
            return self.get_dyn(16 - nrem, pext128_wide(code, W16_MASKS.1[keep as usize]));
        }
        let full = u16::MAX; // all 16 vertices (1u16 << 16 would overflow)
        let order = getk_order_u16::<16>(self.ord_getk, adj);
        for j in 0..16 {
            let i = order[j] as usize;
            if j + 1 < 16 {
                // One-ahead mask prefetch (see `prefetch_mask`); `& 15` proves the adj index.
                let ni = order[j + 1] as usize & 15;
                let nchild = full & !((1u16 << ni) | adj[ni]);
                prefetch_mask(&W16_MASKS.1[nchild as usize]);
            }
            let child = full & !((1u16 << i) | adj[i]);
            // `cpc = 16 - 1 - deg(i)` off the `child`→popcount chain (see get9). `full` is all 16
            // bits, so `{i} ∪ N[i] ⊆ full` (disjoint) and the identity is exact.
            let cpc = (15 - adj[i].count_ones()) as usize;
            let mask = W16_MASKS.1[child as usize];
            // Right-size the child code: most children are ≤11 vertices (≤55-bit code, `u64`) — only
            // the rare ≥12-vertex child (isolated-vertex removal) needs the `u128` `pext128_wide`
            // (128-bit shift). The cheap `u64` `pext128` path avoids that on the common case.
            let lost = if cpc >= 12 {
                let cc = pext128_wide(code, mask);
                match cpc {
                    15 => !self.get15(cc),
                    14 => !self.get14(cc),
                    13 => !self.get13(cc),
                    _ => !self.get12(cc),
                }
            } else {
                let cc = pext128(code, mask);
                match cpc {
                    11 => !self.get11(cc),
                    10 => !self.get10(cc),
                    9 => !self.get9(cc),
                    _ => !self.get(cpc, cc as usize),
                }
            };
            if lost {
                return true;
            }
        }
        false
    }

    /// Exact value of one labelled `K`-vertex wide graph (K=17..20; 3-word `code`). A child has
    /// ≤K-1 vertices: a ≤16-vertex child (≤120 bits) drops to the `u128` [`get_dyn`] machinery, a
    /// 17..19-vertex child stays wide via [`get_dyn_wide`]. `incident`/`induced` are the layer's
    /// masks. Children swept degree-descending when `ord_getk` (earliest cutoff). Bounded-depth
    /// recursion into the complete `W0..W8` tables — no TT, no allocation, no re-expansion.
    #[inline]
    fn get_wide<const K: usize>(
        &self,
        code: &Code192,
        incident: &[Code192; MAX_WIDE_K],
        induced: &[Code192],
    ) -> bool {
        let adj = extract_adj_wide::<K>(code, incident);
        let full = (1u32 << K) - 1;
        let order = getk_order_u32::<K>(self.ord_getk, &adj);
        for j in 0..K {
            let i = order[j] as usize;
            if j + 1 < K {
                // One-ahead mask prefetch (see `prefetch_mask`) — the wide induced tables are
                // 3..24 MB, so this load misses L2/L3.
                // SAFETY: `order[..K]` holds vertex indices `< K ≤ 20` (`deg_order_desc` /
                // `IDENTITY21` invariant), in bounds of `adj: [u32; 20]`.
                let ni = order[j + 1] as usize;
                let nadj = unsafe { *adj.get_unchecked(ni) };
                let nchild = full & !((1u32 << ni) | nadj);
                // SAFETY: `nchild < 2^K`, the induced table size (same invariant as `mask` below).
                prefetch_mask(unsafe { induced.get_unchecked(nchild as usize) });
            }
            let child = full & !((1u32 << i) | adj[i]);
            // `cpc = K - 1 - deg(i)` off the `child`→popcount chain (see get9): a move removes
            // `{i} ∪ N[i]` (disjoint, ⊆ full), so `popcount(adj[i])` (depends only on `adj[i]`)
            // issues in parallel with `child`.
            let cpc = (K - 1) - adj[i].count_ones() as usize;
            // SAFETY: `child < 2^K`, the induced table size.
            let mask = unsafe { induced.get_unchecked(child as usize) };
            let cc = pext192_u192(code, mask);
            let lost = if cpc >= 17 {
                !self.get_dyn_wide(cpc, &cc)
            } else {
                // ≤16-vertex child: its code is ≤120 bits, so the low two words are the `u128`.
                !self.get_dyn(cpc, (cc[0] as u128) | ((cc[1] as u128) << 64))
            };
            if lost {
                return true;
            }
        }
        false
    }

    /// Wide dispatch for a runtime `k` (a wide child of a wider node). `k ≥ 17` routes to the
    /// matching wide layer; `k ≤ 16` reconstructs the `u128` and uses [`get_dyn`].
    #[inline]
    pub(crate) fn get_dyn_wide(&self, k: usize, code: &Code192) -> bool {
        match k {
            20 => self.get20(code),
            19 => self.get19(code),
            18 => self.get18(code),
            17 => self.get17(code),
            _ => self.get_dyn(k, (code[0] as u128) | ((code[1] as u128) << 64)),
        }
    }

    /// Exact value of one labelled 17-vertex graph (136-bit, 3 words) — the first layer above the
    /// `u128` K=16 ceiling. Resolves pc==17 nodes directly as a getK leaf.
    #[inline]
    pub(crate) fn get17(&self, code: &Code192) -> bool {
        self.get_wide::<17>(code, &W17_INCIDENT, self.w17_induced)
    }

    /// Exact value of one labelled 18-vertex graph (153-bit, 3 words).
    #[inline]
    pub(crate) fn get18(&self, code: &Code192) -> bool {
        self.get_wide::<18>(code, &W18_INCIDENT, wide_induced(18))
    }

    /// Exact value of one labelled 19-vertex graph (171-bit, 3 words).
    #[inline]
    pub(crate) fn get19(&self, code: &Code192) -> bool {
        self.get_wide::<19>(code, &W19_INCIDENT, wide_induced(19))
    }

    /// Exact value of one labelled 20-vertex graph (190-bit, 3 words) — the 3-word code ceiling
    /// (20·19/2 = 190 ≤ 192).
    #[inline]
    pub(crate) fn get20(&self, code: &Code192) -> bool {
        self.get_wide::<20>(code, &W20_INCIDENT, wide_induced(20))
    }

    pub(crate) fn bytes(&self) -> u64 {
        std::mem::size_of_val(self.arena) as u64
    }
}

fn build_tables() -> Vec<Box<[u64]>> {
    let mut tables: Vec<Box<[u64]>> = Vec::with_capacity(W8_K + 1);
    tables.push(vec![0u64].into_boxed_slice());
    for k in 1..=W8_K {
        tables.push(build_table(k, &tables));
    }
    tables
}

// ========================= Grundy (Sprague-Grundy) dense layer =========================
//
// The nimber twin of the boolean `W0..=W8` layer, for the `queens nimber` sum-game engine
// (OEIS A344227): `G_k[code]` = the Sprague-Grundy value of the labelled `k ≤ 8`-vertex
// Node-Kayles graph, and `g9..g12` resolve 9..12-vertex graphs by a nested mex sweep over
// the same const projection masks the boolean `get9..get12` use (the masks are pure code-space
// geometry — value-independent). Unlike the boolean sweeps there is NO early-out: `mex` needs
// every child's value. A game on `k` vertices has value `≤ k` (children have `≤ k-1` vertices,
// so by induction mex is over values `≤ k-1`), hence one byte per code is ample.

/// Byte offset of the `G_k` table in the flat Grundy arena (one byte per labelled code).
/// The bytes twin of [`TABLE_OFF`].
const fn gtable_offsets() -> [usize; 16] {
    let mut off = [0usize; 16];
    let mut k = 0;
    while k <= W8_K {
        off[k + 1] = off[k] + slots(k);
        k += 1;
    }
    off
}
const GTABLE_OFF: [usize; 16] = gtable_offsets();

/// Build-time scalar mex of one labelled `k ≤ 7`-vertex graph (the tiny tables; mirrors
/// [`graph_wins`]).
fn grundy_graph(k: usize, code: usize, tables: &[Box<[u8]>]) -> u8 {
    let adj = adj_from_code(k, code as u128);
    let full = (1u16 << k) - 1;
    let mut seen = 0u32;
    for i in 0..k {
        let child = full & !((1u16 << i) | adj[i]);
        let (ck, ccode) = projected_code(&adj, child);
        seen |= 1u32 << tables[ck][ccode as usize];
    }
    (!seen).trailing_zeros() as u8
}

/// `pext` twin of [`grundy_graph`] specialised to `k == 8` — the dominant build cost
/// (2^28 codes). Mirrors [`graph_wins8`] with mex accumulation instead of the early-out.
#[inline]
fn grundy_graph8(code: u64, tables: &[Box<[u8]>]) -> u8 {
    use std::arch::x86_64::_pext_u64;
    let (adj, _) = extract_adj::<8>(code, &W8_MASKS.0);
    let full = (1u16 << 8) - 1;
    let mut seen = 0u32;
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        let child = full & !((1u16 << i) | adj[i]);
        // SAFETY: target-cpu=znver5 ⇒ BMI2; same build invariant as `graph_wins8`.
        let child_code = unsafe { _pext_u64(code, W8_MASKS.1[child as usize]) } as usize;
        seen |= 1u32 << tables[child.count_ones() as usize][child_code];
    }
    (!seen).trailing_zeros() as u8
}

fn build_grundy_table(k: usize, tables: &[Box<[u8]>]) -> Box<[u8]> {
    let mut out = vec![0u8; slots(k)].into_boxed_slice();
    const CHUNK: usize = 1 << 16;
    out.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base = ci * CHUNK;
            for (i, slot) in chunk.iter_mut().enumerate() {
                *slot = if k == 8 {
                    grundy_graph8((base + i) as u64, tables)
                } else {
                    grundy_graph(k, base + i, tables)
                };
            }
        });
    out
}

fn build_grundy_tables() -> Vec<Box<[u8]>> {
    let mut tables: Vec<Box<[u8]>> = Vec::with_capacity(W8_K + 1);
    tables.push(vec![0u8].into_boxed_slice()); // G(empty graph) = 0
    for k in 1..=W8_K {
        tables.push(build_grundy_table(k, &tables));
    }
    tables
}

/// Complete Sprague-Grundy (nimber) tables for all labelled Node-Kayles graphs up to 8
/// vertices, plus nested mex sweeps for 9..12 vertices — the Grundy twin of [`DenseW8`].
/// Build ≈ the boolean W8 build without the early-out (a few seconds, parallel); no disk
/// cache (nimber runs are one-off, unlike the per-solve boolean prep).
pub(crate) struct GrundyW8 {
    /// `G0..=G8` concatenated, one byte per code, indexed via [`GTABLE_OFF`].
    arena: &'static [u8],
}

impl GrundyW8 {
    pub(crate) fn build() -> Self {
        static ARENA: OnceLock<Box<[u8]>> = OnceLock::new();
        let arena: &'static [u8] = ARENA.get_or_init(|| {
            let tables = build_grundy_tables();
            let mut flat = vec![0u8; GTABLE_OFF[W8_K + 1]].into_boxed_slice();
            for k in 0..=W8_K {
                let off = GTABLE_OFF[k];
                flat[off..off + tables[k].len()].copy_from_slice(&tables[k]);
            }
            // Theory bound: a Node-Kayles game on k vertices has G ≤ k. A violation means a
            // build bug — fail loudly rather than return wrong nimbers.
            let mx = flat.iter().copied().max().unwrap_or(0);
            assert!(mx <= W8_K as u8, "GrundyW8 build: nimber {mx} > {W8_K}");
            flat
        });
        collapse_huge(arena);
        GrundyW8 { arena }
    }

    /// Nimber of a labelled `k ≤ 8`-vertex graph — one byte load.
    #[inline]
    pub(crate) fn get(&self, k: usize, code: usize) -> u8 {
        debug_assert!(k <= W8_K && code < slots(k));
        // SAFETY: as [`DenseW8::get`] — every caller passes `code < slots(k)`, and
        // `GTABLE_OFF[k] + code < GTABLE_OFF[k + 1] ≤ arena.len()`.
        unsafe { *self.arena.get_unchecked(GTABLE_OFF[k & 15] + code) }
    }

    /// Nimber of a labelled 9-vertex graph (36-bit code): one mex sweep, every child a
    /// direct `G{≤8}` lookup. Mirrors [`DenseW8::get9`] (no iso-strip, no early-out).
    pub(crate) fn g9(&self, code: u64) -> u8 {
        use std::arch::x86_64::_pext_u64;
        debug_assert!(code < (1u64 << 36));
        let (adj, _) = extract_adj::<9>(code, &W9_MASKS.0);
        let full = (1u16 << 9) - 1;
        let mut seen = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..9 {
            let child = full & !((1u16 << i) | adj[i]);
            let cpc = (8 - adj[i].count_ones()) as usize;
            // SAFETY: znver5 ⇒ BMI2; same invariant as `DenseW8::get9`.
            let child_code = unsafe { _pext_u64(code, W9_MASKS.1[child as usize]) } as usize;
            seen |= 1u32 << self.get(cpc, child_code);
        }
        (!seen).trailing_zeros() as u8
    }

    /// Nimber of a labelled 10-vertex graph (45-bit code); a deg-0 child nests one [`g9`].
    pub(crate) fn g10(&self, code: u64) -> u8 {
        use std::arch::x86_64::_pext_u64;
        debug_assert!(code < (1u64 << 45));
        let (adj, _) = extract_adj::<10>(code, &W10_MASKS.0);
        let full = (1u16 << 10) - 1;
        let mut seen = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..10 {
            let child = full & !((1u16 << i) | adj[i]);
            let cpc = (9 - adj[i].count_ones()) as usize;
            // SAFETY: znver5 ⇒ BMI2; same invariant as `DenseW8::get10`.
            let child_code = unsafe { _pext_u64(code, W10_MASKS.1[child as usize]) };
            let g = if cpc == 9 {
                self.g9(child_code)
            } else {
                self.get(cpc, child_code as usize)
            };
            seen |= 1u32 << g;
        }
        (!seen).trailing_zeros() as u8
    }

    /// Nimber of a labelled 11-vertex graph (55-bit code).
    pub(crate) fn g11(&self, code: u64) -> u8 {
        use std::arch::x86_64::_pext_u64;
        debug_assert!(code < (1u64 << 55));
        let (adj, _) = extract_adj::<11>(code, &W11_MASKS.0);
        let full = (1u16 << 11) - 1;
        let mut seen = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..11 {
            let child = full & !((1u16 << i) | adj[i]);
            let cpc = (10 - adj[i].count_ones()) as usize;
            // SAFETY: znver5 ⇒ BMI2; same invariant as `DenseW8::get11`.
            let child_code = unsafe { _pext_u64(code, W11_MASKS.1[child as usize]) };
            let g = match cpc {
                10 => self.g10(child_code),
                9 => self.g9(child_code),
                _ => self.get(cpc, child_code as usize),
            };
            seen |= 1u32 << g;
        }
        (!seen).trailing_zeros() as u8
    }

    /// Nimber of a labelled 12-vertex graph (66-bit code, `u128` — the first past the `u64`
    /// ceiling). Mirrors [`DenseW8::get12`] (label order; no prefetch/ordering micro-opts).
    pub(crate) fn g12(&self, code: u128) -> u8 {
        debug_assert!(code < (1u128 << 66));
        let (adj, _) = extract_adj128::<12>(code, &W12_MASKS.0);
        let full = (1u16 << 12) - 1;
        let mut seen = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..12 {
            let child = full & !((1u16 << i) | adj[i]);
            let cpc = (11 - adj[i].count_ones()) as usize;
            let child_code = pext128(code, W12_MASKS.1[child as usize]);
            let g = match cpc {
                11 => self.g11(child_code),
                10 => self.g10(child_code),
                9 => self.g9(child_code),
                _ => self.get(cpc, child_code as usize),
            };
            seen |= 1u32 << g;
        }
        (!seen).trailing_zeros() as u8
    }

    /// Nimber of a labelled 13-vertex graph (78-bit code). Mirrors [`DenseW8::get13`]'s
    /// child right-sizing (only a 12-vertex child needs the `u128` [`pext128_wide`]); no
    /// early-out — mex needs every child.
    pub(crate) fn g13(&self, code: u128) -> u8 {
        debug_assert!(code < (1u128 << 78));
        let (adj, _) = extract_adj128::<13>(code, &W13_MASKS.0);
        let full = (1u16 << 13) - 1;
        let mut seen = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..13 {
            let child = full & !((1u16 << i) | adj[i]);
            let cpc = (12 - adj[i].count_ones()) as usize;
            let mask = W13_MASKS.1[child as usize];
            let g = if cpc == 12 {
                self.g12(pext128_wide(code, mask))
            } else {
                let cc = pext128(code, mask);
                match cpc {
                    11 => self.g11(cc),
                    10 => self.g10(cc),
                    9 => self.g9(cc),
                    _ => self.get(cpc, cc as usize),
                }
            };
            seen |= 1u32 << g;
        }
        (!seen).trailing_zeros() as u8
    }

    /// Nimber of a labelled 14-vertex graph (91-bit code); 12/13-vertex children stay `u128`.
    pub(crate) fn g14(&self, code: u128) -> u8 {
        debug_assert!(code < (1u128 << 91));
        let (adj, _) = extract_adj128::<14>(code, &W14_MASKS.0);
        let full = (1u16 << 14) - 1;
        let mut seen = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..14 {
            let child = full & !((1u16 << i) | adj[i]);
            let cpc = (13 - adj[i].count_ones()) as usize;
            let mask = W14_MASKS.1[child as usize];
            let g = if cpc >= 12 {
                let cc = pext128_wide(code, mask);
                if cpc == 13 {
                    self.g13(cc)
                } else {
                    self.g12(cc)
                }
            } else {
                let cc = pext128(code, mask);
                match cpc {
                    11 => self.g11(cc),
                    10 => self.g10(cc),
                    9 => self.g9(cc),
                    _ => self.get(cpc, cc as usize),
                }
            };
            seen |= 1u32 << g;
        }
        (!seen).trailing_zeros() as u8
    }

    /// Nimber of a labelled 15-vertex graph (105-bit code).
    pub(crate) fn g15(&self, code: u128) -> u8 {
        debug_assert!(code < (1u128 << 105));
        let (adj, _) = extract_adj128::<15>(code, &W15_MASKS.0);
        let full = (1u16 << 15) - 1;
        let mut seen = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..15 {
            let child = full & !((1u16 << i) | adj[i]);
            let cpc = (14 - adj[i].count_ones()) as usize;
            let mask = W15_MASKS.1[child as usize];
            let g = if cpc >= 12 {
                let cc = pext128_wide(code, mask);
                match cpc {
                    14 => self.g14(cc),
                    13 => self.g13(cc),
                    _ => self.g12(cc),
                }
            } else {
                let cc = pext128(code, mask);
                match cpc {
                    11 => self.g11(cc),
                    10 => self.g10(cc),
                    9 => self.g9(cc),
                    _ => self.get(cpc, cc as usize),
                }
            };
            seen |= 1u32 << g;
        }
        (!seen).trailing_zeros() as u8
    }

    /// Nimber of a labelled 16-vertex graph (120-bit code — the `u128` ceiling, as
    /// [`DenseW8::get16`]).
    pub(crate) fn g16(&self, code: u128) -> u8 {
        debug_assert!(code < (1u128 << 120));
        let (adj, _) = extract_adj128::<16>(code, &W16_MASKS.0);
        let full = u16::MAX;
        let mut seen = 0u32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..16 {
            let child = full & !((1u16 << i) | adj[i]);
            let cpc = (15 - adj[i].count_ones()) as usize;
            let mask = W16_MASKS.1[child as usize];
            let g = if cpc >= 12 {
                let cc = pext128_wide(code, mask);
                match cpc {
                    15 => self.g15(cc),
                    14 => self.g14(cc),
                    13 => self.g13(cc),
                    _ => self.g12(cc),
                }
            } else {
                let cc = pext128(code, mask);
                match cpc {
                    11 => self.g11(cc),
                    10 => self.g10(cc),
                    9 => self.g9(cc),
                    _ => self.get(cpc, cc as usize),
                }
            };
            seen |= 1u32 << g;
        }
        (!seen).trailing_zeros() as u8
    }

    /// Dispatch to the right layer for a runtime `k ≤ 16`.
    #[inline]
    pub(crate) fn grundy_dyn(&self, k: usize, code: u128) -> u8 {
        match k {
            16 => self.g16(code),
            15 => self.g15(code),
            14 => self.g14(code),
            13 => self.g13(code),
            12 => self.g12(code),
            11 => self.g11(code as u64),
            10 => self.g10(code as u64),
            9 => self.g9(code as u64),
            _ => self.get(k, code as usize),
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.arena.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cached `W0..=W8` tables in the `Box<[u64]>` form the scalar reference [`wins_rec`] reads.
    /// Built independently of the production flat [`DenseW8`] arena (preserves the cross-check),
    /// cached so the 2^28-code W8 build runs once across all `direct_w*` tests.
    fn ref_tables() -> &'static [Box<[u64]>] {
        static T: OnceLock<Vec<Box<[u64]>>> = OnceLock::new();
        T.get_or_init(build_tables)
    }

    /// Pure recursive scalar mex reference (recurses to the empty graph — no tables, no
    /// `pext`, fully independent). Tractable to `k = 8` (`8! ≈ 40K` paths per code).
    fn grundy_rec_pure(k: usize, code: u128) -> u8 {
        if k == 0 {
            return 0;
        }
        let adj = adj_from_code(k, code);
        let full = ((1u32 << k) - 1) as u16;
        let mut seen = 0u32;
        for i in 0..k {
            let child = full & !((1u16 << i) | adj[i]);
            let (ck, ccode) = projected_code(&adj, child);
            seen |= 1u32 << grundy_rec_pure(ck, ccode);
        }
        (!seen).trailing_zeros() as u8
    }

    /// Scalar mex reference for `k = 9..=12` — the Grundy twin of [`wins_rec`]: recurses one
    /// ply per layer, bottoming at the reference `G{≤8}` tables (themselves validated against
    /// [`grundy_rec_pure`] — layered, like the boolean `direct_w*` chain).
    fn grundy_rec(k: usize, code: u128, gtables: &[Box<[u8]>]) -> u8 {
        if k <= W8_K {
            return gtables[k][code as usize];
        }
        let adj = adj_from_code(k, code);
        let full = ((1u32 << k) - 1) as u16;
        let mut seen = 0u32;
        for i in 0..k {
            let child = full & !((1u16 << i) | adj[i]);
            let (ck, ccode) = projected_code(&adj, child);
            seen |= 1u32 << grundy_rec(ck, ccode, gtables);
        }
        (!seen).trailing_zeros() as u8
    }

    /// Cached reference `G0..=G8` tables (independent instance of the build, shared across
    /// the Grundy tests like [`ref_tables`] is for the boolean chain).
    fn ref_gtables() -> &'static [Box<[u8]>] {
        static T: OnceLock<Vec<Box<[u8]>>> = OnceLock::new();
        T.get_or_init(build_grundy_tables)
    }

    /// The Grundy arena matches the scalar mex reference exhaustively for small `k`, sampled
    /// at `k = 7, 8`; and `G != 0 ⟺ W-win` against the independently built boolean tables
    /// (the production-validated ground truth) at every checked code.
    #[test]
    fn grundy_tables_match_scalar_and_boolean() {
        let g = GrundyW8::build();
        let w = ref_tables();
        #[allow(clippy::needless_range_loop)]
        for k in 0..=6usize {
            for code in 0..slots(k) {
                let got = g.get(k, code);
                assert_eq!(got, grundy_rec_pure(k, code as u128), "k={k} code={code}");
                assert_eq!(
                    got != 0,
                    bit_get(&w[k], code),
                    "k={k} code={code}: G!=0 must equal boolean win"
                );
            }
        }
        for k in [7usize, 8] {
            // Deterministic stride + LCG sample (the exhaustive space is 2^21 / 2^28).
            let mut x = 0x9E37_79B9u64;
            for s in 0..1024u64 {
                let code = if s % 2 == 0 {
                    (s * 65_537) as usize % slots(k)
                } else {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (x >> 16) as usize % slots(k)
                };
                let got = g.get(k, code);
                assert_eq!(got, grundy_rec_pure(k, code as u128), "k={k} code={code}");
                assert_eq!(got != 0, bit_get(&w[k], code), "k={k} code={code}");
            }
        }
    }

    /// The nested mex sweeps `g9..g16` match the scalar recursion and the boolean `wins_rec`
    /// sign on deterministic pseudo-random codes. Random codes are ~1/2 edge density, so the
    /// no-early-out references stay tractable through k=16 (children drop deep per ply).
    #[test]
    fn grundy_g9_to_g16_match_reference() {
        let g = GrundyW8::build();
        let w = ref_tables();
        let gt = ref_gtables();
        let mut x = 0xD1B5_4A32_D192_ED03u64;
        for k in 9..=16usize {
            let bits = k * (k - 1) / 2;
            for _ in 0..48 {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1442695040888963407);
                let hi = x;
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1442695040888963407);
                let code = (((hi as u128) << 64) | x as u128) & ((1u128 << bits) - 1);
                let got = g.grundy_dyn(k, code);
                assert_eq!(got, grundy_rec(k, code, gt), "k={k} code={code:#x}");
                assert_eq!(
                    got != 0,
                    wins_rec(k, code, w),
                    "k={k} code={code:#x}: G!=0 must equal boolean win"
                );
            }
        }
        // Sparser codes (~1/4 edge density — AND of two shifted copies) exercise the deep
        // high-cpc child nesting (isolated-removal chains) the dense samples rarely hit.
        // Kept to a few samples per k: the no-early-out references grow fast as density drops.
        for k in 13..=16usize {
            let bits = k * (k - 1) / 2;
            for _ in 0..8 {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1442695040888963407);
                let hi = x;
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1442695040888963407);
                let dense = ((hi as u128) << 64) | x as u128;
                let code = (dense & (dense >> 3)) & ((1u128 << bits) - 1);
                let got = g.grundy_dyn(k, code);
                assert_eq!(got, grundy_rec(k, code, gt), "sparse k={k} code={code:#x}");
                assert_eq!(
                    got != 0,
                    wins_rec(k, code, w),
                    "sparse k={k} code={code:#x}: G!=0 must equal boolean win"
                );
            }
        }
    }

    #[test]
    fn dense_w8_known_small_prefix() {
        let mut tables: Vec<Box<[u64]>> = Vec::new();
        tables.push(vec![0u64].into_boxed_slice());
        for k in 1..=4 {
            tables.push(build_table(k, &tables));
        }
        assert_eq!(tables[1][0].count_ones(), 1);
        assert_eq!(tables[2].iter().map(|w| w.count_ones()).sum::<u32>(), 1);
        assert_eq!(tables[3].iter().map(|w| w.count_ones()).sum::<u32>(), 5);
        assert_eq!(tables[4].iter().map(|w| w.count_ones()).sum::<u32>(), 41);
    }

    #[test]
    fn graph_wins8_matches_scalar() {
        // The `pext` k=8 build path must be bit-identical to the scalar reference. Build the
        // tiny k≤7 tables (both paths read them), then compare over a spread of 28-bit codes.
        let mut tables: Vec<Box<[u64]>> = Vec::new();
        tables.push(vec![0u64].into_boxed_slice());
        for k in 1..=7 {
            tables.push(build_table(k, &tables));
        }
        for x in 0..50_000u64 {
            let code = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) & ((1u64 << 28) - 1);
            assert_eq!(
                graph_wins8(code, &tables),
                graph_wins(8, code as usize, &tables),
                "graph_wins8 mismatch at code {code:#x}"
            );
        }
    }

    #[test]
    fn iso_strip_matches_scalar() {
        // The isolated-vertex pair-strip must stay bit-identical to the scalar `wins_rec` ground
        // truth (which has no strip). Random dense codes rarely have ≥2 isolated vertices, so force
        // SPARSE graphs (AND three shifted copies ⇒ ~1/8 edge density ⇒ many isolated vertices),
        // exercising the strip path across the u128 (16/14/12) and u64 (11/9) layers.
        let w8 = DenseW8::build();
        for x in 0..40_000u64 {
            let lo = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hi = x.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let dense = (lo as u128) | ((hi as u128) << 64);
            let sparse = dense & (dense >> 1) & (dense >> 3); // ~1/8 of edge bits survive
            let c16 = sparse & ((1u128 << 120) - 1);
            assert_eq!(
                w8.get16(c16),
                wins_rec(16, c16, ref_tables()),
                "get16 sparse {c16:#x}"
            );
            let c14 = sparse & ((1u128 << 91) - 1);
            assert_eq!(
                w8.get14(c14),
                wins_rec(14, c14, ref_tables()),
                "get14 sparse {c14:#x}"
            );
            let c12 = sparse & ((1u128 << 66) - 1);
            assert_eq!(
                w8.get12(c12),
                wins_rec(12, c12, ref_tables()),
                "get12 sparse {c12:#x}"
            );
            let c11 = (sparse as u64) & ((1u64 << 55) - 1);
            assert_eq!(
                w8.get11(c11),
                wins_rec(11, c11 as u128, ref_tables()),
                "get11 sparse {c11:#x}"
            );
            let c9 = (sparse as u64) & ((1u64 << 36) - 1);
            assert_eq!(
                w8.get9(c9),
                wins_rec(9, c9 as u128, ref_tables()),
                "get9 sparse {c9:#x}"
            );
        }
    }

    #[test]
    fn direct_w9_matches_scalar_recurrence() {
        let w8 = DenseW8::build();
        for x in 0..10_000u64 {
            let code = x.wrapping_mul(0x9E37_79B9) & ((1u64 << 36) - 1);
            assert_eq!(w8.get9(code), wins_rec(9, code as u128, ref_tables()));
        }
    }

    #[test]
    fn direct_w10_matches_scalar_recurrence() {
        let w8 = DenseW8::build();
        for x in 0..20_000u64 {
            let code = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) & ((1u64 << 45) - 1);
            assert_eq!(
                w8.get10(code),
                wins_rec(10, code as u128, ref_tables()),
                "W10 mismatch at code {code:#x}"
            );
        }
    }

    #[test]
    fn direct_w11_matches_scalar_recurrence() {
        let w8 = DenseW8::build();
        for x in 0..20_000u64 {
            let code = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) & ((1u64 << 55) - 1);
            assert_eq!(
                w8.get11(code),
                wins_rec(11, code as u128, ref_tables()),
                "W11 mismatch at code {code:#x}"
            );
        }
    }

    #[test]
    fn direct_w12_matches_scalar_recurrence() {
        let w8 = DenseW8::build();
        // Spread x across the full 66-bit code with two independent 64-bit mixes.
        for x in 0..30_000u64 {
            let lo = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hi = x.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let code = ((lo as u128) | ((hi as u128) << 64)) & ((1u128 << 66) - 1);
            assert_eq!(
                w8.get12(code),
                wins_rec(12, code, ref_tables()),
                "W12 mismatch at code {code:#x}"
            );
        }
    }

    #[test]
    fn direct_w13_matches_scalar_recurrence() {
        let w8 = DenseW8::build();
        for x in 0..30_000u64 {
            let lo = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hi = x.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let code = ((lo as u128) | ((hi as u128) << 64)) & ((1u128 << 78) - 1);
            assert_eq!(
                w8.get13(code),
                wins_rec(13, code, ref_tables()),
                "W13 mismatch at code {code:#x}"
            );
        }
    }

    #[test]
    fn direct_w14_matches_scalar_recurrence() {
        let w8 = DenseW8::build();
        for x in 0..30_000u64 {
            let lo = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hi = x.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let code = ((lo as u128) | ((hi as u128) << 64)) & ((1u128 << 91) - 1);
            assert_eq!(
                w8.get14(code),
                wins_rec(14, code, ref_tables()),
                "W14 mismatch at code {code:#x}"
            );
        }
    }

    #[test]
    fn direct_w15_matches_scalar_recurrence() {
        let w8 = DenseW8::build();
        for x in 0..30_000u64 {
            let lo = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hi = x.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let code = ((lo as u128) | ((hi as u128) << 64)) & ((1u128 << 105) - 1);
            assert_eq!(
                w8.get15(code),
                wins_rec(15, code, ref_tables()),
                "W15 mismatch at code {code:#x}"
            );
        }
    }

    #[test]
    fn direct_w16_matches_scalar_recurrence() {
        let w8 = DenseW8::build();
        for x in 0..30_000u64 {
            let lo = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hi = x.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            let code = ((lo as u128) | ((hi as u128) << 64)) & ((1u128 << 120) - 1);
            assert_eq!(
                w8.get16(code),
                wins_rec(16, code, ref_tables()),
                "W16 mismatch at code {code:#x}"
            );
        }
    }

    // ===== Wide (W17..W20) scalar reference (3-word code; the u128 `wins_rec` tops out at K=16). =====

    /// Decode the `k` adjacency rows (`u32`) from a `k(k-1)/2`-bit 3-word code — pure scalar.
    fn adjw_from_code(code: &[u64; 3], k: usize) -> [u32; 20] {
        let mut adj = [0u32; 20];
        let mut bit = 0usize;
        for i in 0..k {
            for j in (i + 1)..k {
                if (code[bit >> 6] >> (bit & 63)) & 1 != 0 {
                    adj[i] |= 1 << j;
                    adj[j] |= 1 << i;
                }
                bit += 1;
            }
        }
        adj
    }

    /// Project an `alive` subset (≤16 vertices) to its `(k, u128 code)` — relabel survivors `0..k`.
    fn projw_to_u128(adj: &[u32; 20], alive: u32) -> (usize, u128) {
        let verts: Vec<usize> = (0..20).filter(|&v| alive & (1 << v) != 0).collect();
        let k = verts.len();
        let mut code = 0u128;
        let mut bit = 0u32;
        for a in 0..k {
            for b in (a + 1)..k {
                if adj[verts[a]] & (1 << verts[b]) != 0 {
                    code |= 1u128 << bit;
                }
                bit += 1;
            }
        }
        (k, code)
    }

    /// Scalar minimax over a wide graph's adjacency: a node WINS iff some move (place a queen on
    /// vertex `i`, deleting `N[i]`) leaves the opponent a LOSS. ≤16-vertex children bottom out in
    /// the `u128` `wins_rec`; 17..19-vertex children recurse here. The empty graph is a loss.
    fn winsw_scalar(adj: &[u32; 20], alive: u32, tables: &[Box<[u64]>]) -> bool {
        let mut a = alive;
        while a != 0 {
            let i = a.trailing_zeros() as usize;
            a &= a - 1;
            let child = alive & !((1u32 << i) | adj[i]);
            let cpc = child.count_ones() as usize;
            let lost = if cpc <= 16 {
                let (ck, ccode) = projw_to_u128(adj, child);
                !wins_rec(ck, ccode, tables)
            } else {
                !winsw_scalar(adj, child, tables)
            };
            if lost {
                return true;
            }
        }
        false
    }

    /// `get17`..`get20` must match the scalar minimax over random graphs at a spread of densities.
    fn check_wide_layer(k: usize) {
        let w8 = DenseW8::build();
        let full = (1u32 << k) - 1;
        let edges = k * (k - 1) / 2;
        for x in 0..40_000u64 {
            let mut code = [0u64; 3];
            for (w, c) in code.iter_mut().enumerate() {
                let s = (x + 1).wrapping_mul(
                    0x9E37_79B9_7F4A_7C15 ^ (w as u64).wrapping_mul(0x1234_5678_9ABC_DEF1),
                );
                let mut v = s ^ s.rotate_left(31);
                if x % 3 == 1 {
                    v &= s.rotate_left(17); // ~25% density
                } else if x % 3 == 2 {
                    v &= s.rotate_left(17) & s.rotate_left(43); // ~12% density (deep recursion)
                }
                *c = v;
            }
            // Keep only the `edges` used bits (clear the high tail of word 2).
            if edges < 128 {
                code[2] = 0;
                code[1] &= if edges <= 64 {
                    0
                } else {
                    (1u64 << (edges - 64)) - 1
                };
                if edges <= 64 {
                    code[0] &= if edges == 64 { !0 } else { (1u64 << edges) - 1 };
                }
            } else {
                code[2] &= (1u64 << (edges - 128)) - 1;
            }
            let adj = adjw_from_code(&code, k);
            assert_eq!(
                w8.get_dyn_wide(k, &code),
                winsw_scalar(&adj, full, ref_tables()),
                "W{k} mismatch at code {code:?}"
            );
        }
    }

    #[test]
    fn direct_w17_matches_scalar_recurrence() {
        check_wide_layer(17);
    }

    #[test]
    fn direct_w18_matches_scalar_recurrence() {
        check_wide_layer(18);
    }

    #[test]
    fn direct_w19_matches_scalar_recurrence() {
        check_wide_layer(19);
    }

    #[test]
    fn direct_w20_matches_scalar_recurrence() {
        check_wide_layer(20);
    }
}
