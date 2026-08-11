//! Graph-isomorphism (WL / individualisation-refinement) keys over the
//! available-graph -- the freeze-time merge lever (#7). Measurement/spike keys.

use super::*;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Largest connected component the graph key resolves by direct degree-sequence
/// lookup instead of WL refinement (#18). For connected graphs on at most four
/// vertices the sorted degree sequence is a complete isomorphism invariant.
pub(crate) const TINY_MAX: usize = 4;
/// Sentinel "vertex" the padded WL neighbour lists fill unused slots with (#17).
/// It indexes a reserved scratch cell whose mixed colour is held at 0, so a padding
/// slot contributes nothing to the colour fold -- the fixed-stride loop stays
/// value-identical to the variable-trip one. One past the real square range
/// (squares are `0..MAXV`), so it never collides with a real vertex.
const DUMMY_VERT: usize = MAXV;

/// A 64-bit avalanche mix (the SplitMix64 finaliser) for the WL colour hashes in
/// [`Queens::iso_key`]. Cold path (measurement only).
#[inline]
pub(crate) fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Refine `colour` (indexed by square, seeded by the caller) by 1-WL colour
/// refinement until the partition stabilises (≤ |V| rounds): each vertex's next
/// colour mixes its own with the commutative fold of its neighbours' (so neighbour
/// order cannot matter). Returns the stable colouring. Cold measurement path.
fn wl_refine(verts: &[u32], nbrs: &[Bits], mut colour: Vec<u64>) -> Vec<u64> {
    let distinct = |c: &[u64]| {
        let mut v: Vec<u64> = verts.iter().map(|&s| c[s as usize]).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    let mut prev = 0usize;
    for _ in 0..verts.len() {
        let mut next = colour.clone();
        for (&s, nb) in verts.iter().zip(nbrs) {
            let mut h = colour[s as usize].wrapping_mul(0x100_0000_01B3);
            nb.each(|t| h = h.wrapping_add(mix64(colour[t as usize])));
            next[s as usize] = mix64(h);
        }
        colour = next;
        let classes = distinct(&colour);
        if classes == prev {
            break; // partition stable -- further rounds cannot refine
        }
        prev = classes;
    }
    colour
}

/// Hash the sorted multiset of vertex colours into one order-independent value.
fn hash_colours(verts: &[u32], colour: &[u64]) -> u64 {
    let mut c: Vec<u64> = verts.iter().map(|&s| colour[s as usize]).collect();
    c.sort_unstable();
    c.iter().fold(0x2545_F491_4F6C_DD1D, |h, &x| {
        mix64(h ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    })
}

/// Per-thread preallocated scratch for the allocation-free graph key
/// ([`Queens::iso_key_fast`]). Reused across every node on a rayon worker; every cell
/// is written before it is read (only the touched prefixes are read), so it needs no
/// per-call zeroing -- **zero heap allocation in the hot loop**. Boxed so the buffers
/// live on the heap once per thread, not on every call's stack.
///
/// `#[repr(C, align(64))]`: hot-struct discipline (CLAUDE.md). The `[u64; MAXV]` colour
/// arrays are 2048 B (a multiple of 64), so a 64-aligned struct start makes every one of
/// them cache-line-aligned -- the contiguous `col`/`nxt`/`mc` colour map (which LLVM
/// auto-vectorises to AVX-512) then loads/stores on line boundaries, no split loads.
#[repr(C, align(64))]
pub(crate) struct IsoScratch {
    col: [u64; MAXV],             // current colour per *local* vertex (lcol[0..k])
    nxt: [u64; MAXV],             // next-round colour per local vertex
    base: [u64; MAXV],            // degree-seeded local colour (restored before each individualise)
    sort: [u64; MAXV],            // scratch for class-count / colour-multiset sorts
    sigs: [u64; MAXV],            // per-vertex individualisation signatures
    comp_keys: [u64; MAXV],       // per-component canonical keys
    mc: [u64; MAXV + 1], // mix64(lcol) per local vertex, hoisted once/round; [MAXV]=0 dummy
    pub(crate) verts: [u8; MAXV], // local vertex -> square index
    loc: [u16; MAXV],    // square index -> local vertex (inverse of verts)
    order: [u8; MAXV],   // canonical *local* vertex order for the certificate
    nbr_pad: [u16; MAXV * MAXV], // fixed-stride neighbour *local* indices, DUMMY_VERT-padded (#17)
}

impl IsoScratch {
    pub(crate) fn new() -> Box<Self> {
        Box::new(IsoScratch {
            col: [0; MAXV],
            nxt: [0; MAXV],
            base: [0; MAXV],
            sort: [0; MAXV],
            sigs: [0; MAXV],
            comp_keys: [0; MAXV],
            mc: [0; MAXV + 1],
            verts: [0; MAXV],
            loc: [0; MAXV],
            order: [0; MAXV],
            nbr_pad: [0; MAXV * MAXV],
        })
    }
}

thread_local! {
    static ISO_SCRATCH: std::cell::RefCell<Box<IsoScratch>> =
        std::cell::RefCell::new(IsoScratch::new());
}

/// Log2 of the per-thread component-canon cache size (#19). 2^22 slots * 16 B = 64 MB
/// per worker -- the same flat fingerprint-slot shape as the main TT. Tuned at n=16: 2^20
/// is capacity-bound (2^22 is +3.7%), 2^23 ties 2^22; 64 MB/thread (~1.5 GB across 24
/// workers) fits comfortably under the n=16 TT budget.
const COMP_CACHE_BITS: u32 = 22;

/// Per-thread direct-mapped cache amortising [`Queens::comp_canon`] setup+WL (#19).
/// `comp_canon` is a pure function of `(component square-set, board geometry)`, and the
/// same component recurs across many nodes (the graph key is recomputed every node,
/// before the TT probe), so caching its canon skips the whole bit-scan + CSR build + WL
/// when a component repeats. Fingerprint-guarded like the TT: a slot collision with a
/// different component is a fingerprint mismatch (recompute), a same-fingerprint hit on a
/// different component is ~2^-64 (negligible; the search is already probabilistic at the
/// 55-bit TT slot and cross-checked vs Jenrich). The fingerprint folds in the board side
/// `n`, so entries never carry across different-`n` solves in one process.
struct CompCache {
    fp: Box<[u64]>,  // per-slot fingerprint (0 = empty)
    val: Box<[u64]>, // per-slot cached canon
}

impl CompCache {
    fn new() -> Self {
        let n = 1usize << COMP_CACHE_BITS;
        CompCache {
            fp: vec![0u64; n].into_boxed_slice(),
            val: vec![0u64; n].into_boxed_slice(),
        }
    }
    /// Slot index and (nonzero) fingerprint for component `comp` on an `n`-board.
    #[inline]
    fn probe(comp: Bits, n: u32) -> (usize, u64) {
        let w = comp.0;
        let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ n as u64;
        h = mix64(h ^ w[0]);
        h = mix64(h ^ w[1]);
        h = mix64(h ^ w[2]);
        h = mix64(h ^ w[3]);
        let slot = (mix64(h) >> (64 - COMP_CACHE_BITS)) as usize;
        (slot, h | 1) // fingerprint forced nonzero so 0 stays the empty marker
    }
}

thread_local! {
    static COMP_CACHE: std::cell::RefCell<CompCache> =
        std::cell::RefCell::new(CompCache::new());
}

/// Experimental k=8 direct-canon cache. The direct exact canon is too expensive to
/// run per key; this tests whether repeated labelled 8-vertex edge codes are common
/// enough to make an eventual packed dense table worth building. Thread-local to keep
/// the probe lock-free and off the shared-coherence path.
const K8_CACHE_BITS: u32 = 18;

struct K8CanonCache {
    fp: Box<[u32]>,
    val: Box<[u64]>,
}

impl K8CanonCache {
    fn new() -> Self {
        let n = 1usize << K8_CACHE_BITS;
        K8CanonCache {
            fp: vec![0u32; n].into_boxed_slice(),
            val: vec![0u64; n].into_boxed_slice(),
        }
    }

    #[inline]
    fn get_or_insert(&mut self, code: u32) -> u64 {
        let slot = (mix64(code as u64) >> (64 - K8_CACHE_BITS)) as usize;
        let fp = code.wrapping_add(1);
        if self.fp[slot] == fp {
            return self.val[slot];
        }
        let v = small_key_from_code(8, code);
        self.fp[slot] = fp;
        self.val[slot] = v;
        v
    }
}

thread_local! {
    static K8_CANON_CACHE: std::cell::RefCell<K8CanonCache> =
        std::cell::RefCell::new(K8CanonCache::new());
}

/// Combine a node's per-component canon keys into the single graph-iso key: the
/// **sorted multiset** hash. Sorting makes it order-independent (a multiset), so two
/// available-graphs with the same component classes hash identically regardless of how
/// the decomposition enumerated them. The seed/multiplier are fixed so this is
/// byte-identical to the inline fold [`Queens::iso_key_fast_in`] used to do -- the
/// incremental carry must produce the *same* key as a from-scratch decompose, or the TT
/// would split transpositions. Sorts `keys` in place (caller passes scratch, not the
/// component-aligned array).
#[inline]
pub(crate) fn fold_comp_keys(keys: &mut [u64]) -> u64 {
    if keys.is_empty() {
        return 0;
    }
    keys.sort_unstable();
    keys.iter().fold(0x515E_AF00_D515_E5A1, |h, &k| {
        mix64(h ^ k).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    })
}

const SMALL_CANON_MAX: usize = 7;
const SMALL_WORK_MAX: usize = 8;
const SMALL_CANON_OFF: [usize; SMALL_CANON_MAX + 2] = [
    0,       // k=0, 2^0
    1,       // k=1, 2^0
    2,       // k=2, 2^1
    4,       // k=3, 2^3
    12,      // k=4, 2^6
    76,      // k=5, 2^10
    1100,    // k=6, 2^15
    33868,   // k=7, 2^21
    2131020, // total
];

const SMALL_CANON_TAG: u64 = 0x71E7_1E55_7107_0007;

/// Slot count for a complete ≤7 win/loss table keyed by [`Queens::tiny_table_index`]
/// (one slot per labelled edge code across `k ≤ 7`). ~2.1 M slots; at one byte/slot a
/// flat, eviction-free 2 MB table — no fingerprint, no canon lookup on the query path.
pub(crate) const TINY_TABLE_SLOTS: usize = SMALL_CANON_OFF[SMALL_CANON_MAX + 1];

static SMALL_CANON: OnceLock<Box<[u64]>> = OnceLock::new();

pub(crate) fn small_canon_table() -> &'static [u64] {
    SMALL_CANON
        .get_or_init(|| {
            let mut table = vec![0u64; SMALL_CANON_OFF[SMALL_CANON_MAX + 1]].into_boxed_slice();
            for k in 1..=SMALL_CANON_MAX {
                let codes = 1usize << ((k * (k - 1)) / 2);
                for code in 0..codes {
                    table[SMALL_CANON_OFF[k] + code] = small_key_from_code(k, code as u32);
                }
            }
            table
        })
        .as_ref()
}

/// 1-WL refine `col` (square-indexed) to stability over the `k` component vertices.
/// `nbr_pad` is the fixed-stride neighbour table: row `i` holds vertex `i`'s neighbour
/// squares in `nbr_pad[i*stride .. i*stride+deg]`, the rest padded with [`DUMMY_VERT`].
/// All preallocated; no allocation (#17).
///
/// Two TMA-driven shapes vs the old variable-trip CSR walk, both value-identical:
/// - **`mix64` hoisted out of the per-edge loop** into `mcol` -- computed once per
///   vertex per round (`k` calls) instead of once per incident edge (`2|E|` calls).
/// - **fixed-trip inner loop** (`0..stride` every vertex) -- the loop-exit branch is
///   perfectly predicted, where the per-vertex variable trip mispredicted on exit.
///
/// `mc[DUMMY_VERT]` is held at 0, so padding slots add nothing: the accumulated `h`
/// is bit-identical to summing `mix64(lcol[t])` over the real neighbours only.
///
/// Colours are **compact local** (`lcol[0..k]`, vertex `i`'s colour), so the per-round
/// `mc[i] = mix64(lcol[i])` map is a contiguous load→mix→store with no gather/scatter --
/// LLVM auto-vectorises it to AVX-512 `vpmullq`/`vpsrlq`/`vpxorq` (8× u64) on znver5
/// (#17b). The fold's only gather (`mc[neighbour-local]`) hits a small `k`-element array.
fn wl_refine_in(
    k: usize,
    stride: usize,
    nbr_pad: &[u16],
    lcol: &mut [u64],
    nlcol: &mut [u64],
    mc: &mut [u64],
    sort: &mut [u64],
) {
    let mut prev = 0usize;
    mc[DUMMY_VERT] = 0; // padding contributes nothing; never overwritten below
    for _ in 0..k {
        for i in 0..k {
            mc[i] = mix64(lcol[i]); // contiguous map -> AVX-512 vectorised
        }
        for i in 0..k {
            let mut h = lcol[i].wrapping_mul(0x100_0000_01B3);
            let base = i * stride;
            for s in 0..stride {
                h = h.wrapping_add(mc[nbr_pad[base + s] as usize]);
            }
            nlcol[i] = mix64(h);
        }
        lcol[..k].copy_from_slice(&nlcol[..k]);
        let c = classes_in(k, lcol, sort);
        if c == prev {
            break;
        }
        prev = c;
    }
}

/// The number of distinct colours among the `k` (local) vertices (uses `sort`).
fn classes_in(k: usize, lcol: &[u64], sort: &mut [u64]) -> usize {
    sort[..k].copy_from_slice(&lcol[..k]);
    let s = &mut sort[..k];
    s.sort_unstable();
    let mut c = 0usize;
    let mut last = 0u64;
    for (i, &x) in s.iter().enumerate() {
        if i == 0 || x != last {
            c += 1;
            last = x;
        }
    }
    c
}

/// Hash the sorted colour multiset of the `k` (local) vertices (uses `sort`).
fn hash_colours_in(k: usize, lcol: &[u64], sort: &mut [u64]) -> u64 {
    sort[..k].copy_from_slice(&lcol[..k]);
    let s = &mut sort[..k];
    s.sort_unstable();
    s.iter().fold(0x2545_F491_4F6C_DD1D, |h, &x| {
        mix64(h ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    })
}

/// Hash the adjacency of a discrete-coloured component in canonical (colour) order --
/// a complete certificate. `order` holds *local* indices sorted by colour; `verts` maps
/// each back to its square for the adjacency test. Uses preallocated `order`.
fn cert_hash_in(
    attack: &[Bits],
    comp: Bits,
    k: usize,
    verts: &[u8],
    lcol: &[u64],
    order: &mut [u8],
) -> u64 {
    for (i, o) in order[..k].iter_mut().enumerate() {
        *o = i as u8; // local indices 0..k (discrete colouring ⇒ k <= MAXV)
    }
    order[..k].sort_unstable_by_key(|&li| lcol[li as usize]);
    let mut h = 0x0CA7_F00D_u64;
    for ii in 0..k {
        let vi = verts[order[ii] as usize]; // square
        let nbr = attack[vi as usize].and(comp);
        for jj in 0..k {
            let vj = verts[order[jj] as usize]; // square
            if vi != vj && nbr.get(vj as u32) {
                h = mix64(h ^ (jj as u64 + 1)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            }
        }
        h = mix64(h ^ 0xFFFF); // row separator
    }
    h
}

/// Can vertices `a` and `b` be swapped while fixing every other vertex in the
/// component? This cheap automorphism case covers true twins (including cliques) and
/// false twins: `a` and `b` have identical adjacency to every third vertex, so a
/// transposition of the pair preserves the component.
#[inline]
fn twin_vertices(attack: &[Bits], comp: Bits, a: u8, b: u8) -> bool {
    let a = a as usize;
    let b = b as usize;
    let mut an = attack[a].and(comp);
    let mut bn = attack[b].and(comp);
    an.0[a / 64] &= !(1u64 << (a % 64));
    an.0[b / 64] &= !(1u64 << (b % 64));
    bn.0[a / 64] &= !(1u64 << (a % 64));
    bn.0[b / 64] &= !(1u64 << (b % 64));
    for w in 0..WORDS {
        if an.0[w] != bn.0[w] {
            return false;
        }
    }
    true
}

/// Exact canonical key for a connected 5-vertex component. Among the 21 connected
/// unlabeled graphs on five vertices, `(sorted degree sequence, triangle count)` is a
/// complete invariant; this skips WL for the common k=5 bucket without a permutation
/// search.
fn canon5_key(attack: &[Bits], verts: &[u8]) -> u64 {
    let mut deg = [0u8; 5];
    let mut edge = [[false; 5]; 5];
    for i in 0..5 {
        let vi = verts[i] as usize;
        for (j, &vj) in verts.iter().take(5).enumerate() {
            if i != j && attack[vi].get(vj as u32) {
                edge[i][j] = true;
                deg[i] += 1;
            }
        }
    }
    deg.sort_unstable();
    let mut triangles = 0u64;
    for a in 0..3 {
        for b in a + 1..4 {
            for c in b + 1..5 {
                triangles += (edge[a][b] && edge[a][c] && edge[b][c]) as u64;
            }
        }
    }
    let mut packed = triangles;
    for &d in &deg {
        packed = (packed << 4) | d as u64;
    }
    mix64(packed ^ 0xC005_CAFE_5555_0005)
}

/// Exact canonical key for a connected 6-vertex component. Exhaustive enumeration of
/// all 112 connected unlabeled graphs on six vertices shows this tuple multiset is
/// complete: for each vertex, `(degree, incident triangle count, sum of neighbour
/// degrees)`, sorted across vertices.
fn canon6_key(attack: &[Bits], verts: &[u8]) -> u64 {
    let mut adj = [0u8; 6];
    let mut deg = [0u8; 6];
    for i in 0..6 {
        let vi = verts[i] as usize;
        for (j, &vj) in verts.iter().take(6).enumerate() {
            if i != j && attack[vi].get(vj as u32) {
                adj[i] |= 1u8 << j;
                deg[i] += 1;
            }
        }
    }
    let mut tri = [0u8; 6];
    for a in 0..4 {
        for b in a + 1..5 {
            for c in b + 1..6 {
                if (adj[a] & (1u8 << b)) != 0
                    && (adj[a] & (1u8 << c)) != 0
                    && (adj[b] & (1u8 << c)) != 0
                {
                    tri[a] += 1;
                    tri[b] += 1;
                    tri[c] += 1;
                }
            }
        }
    }
    let mut inv = [0u16; 6];
    for i in 0..6 {
        let mut nbr_deg = 0u8;
        for (j, &dj) in deg.iter().enumerate() {
            if (adj[i] & (1u8 << j)) != 0 {
                nbr_deg += dj;
            }
        }
        inv[i] = ((deg[i] as u16) << 9) | ((tri[i] as u16) << 5) | nbr_deg as u16;
    }
    inv.sort_unstable();
    let mut packed = 0u64;
    for &x in &inv {
        packed = (packed << 12) ^ x as u64;
        packed = mix64(packed);
    }
    mix64(packed ^ 0xC006_CAFE_6666_0006)
}

fn adj_from_edge_code(k: usize, code: u32) -> [u8; SMALL_WORK_MAX] {
    let mut adj = [0u8; SMALL_WORK_MAX];
    let mut bit = 0u32;
    for i in 0..k {
        for j in i + 1..k {
            if (code & (1u32 << bit)) != 0 {
                adj[i] |= 1u8 << j;
                adj[j] |= 1u8 << i;
            }
            bit += 1;
        }
    }
    adj
}

fn small_vertex_sigs(k: usize, adj: &[u8]) -> [u32; SMALL_WORK_MAX] {
    let mut deg = [0u8; SMALL_WORK_MAX];
    for i in 0..k {
        deg[i] = adj[i].count_ones() as u8;
    }
    let mut tri = [0u8; SMALL_WORK_MAX];
    for a in 0..k {
        for b in a + 1..k {
            for c in b + 1..k {
                if (adj[a] & (1u8 << b)) != 0
                    && (adj[a] & (1u8 << c)) != 0
                    && (adj[b] & (1u8 << c)) != 0
                {
                    tri[a] += 1;
                    tri[b] += 1;
                    tri[c] += 1;
                }
            }
        }
    }
    let mut sig = [0u32; SMALL_WORK_MAX];
    for i in 0..k {
        let mut nbr_deg = 0u8;
        let mut nbr_deg_multiset = [0u8; SMALL_WORK_MAX];
        for j in 0..k {
            if (adj[i] & (1u8 << j)) != 0 {
                nbr_deg += deg[j];
                nbr_deg_multiset[j] = deg[j];
            }
        }
        nbr_deg_multiset[..k].sort_unstable();
        let mut packed = 0u32;
        for &d in &nbr_deg_multiset[..k] {
            packed = (packed << 3) | d as u32;
        }
        sig[i] =
            ((deg[i] as u32) << 28) | ((tri[i] as u32) << 24) | ((nbr_deg as u32) << 18) | packed;
    }
    sig
}

fn permuted_code(k: usize, adj: &[u8], perm: &[u8; SMALL_WORK_MAX]) -> u32 {
    let mut code = 0u32;
    let mut bit = 0u32;
    for i in 0..k {
        let pi = perm[i] as usize;
        for &pj_raw in &perm[i + 1..k] {
            let pj = pj_raw as usize;
            if (adj[pi] & (1u8 << pj)) != 0 {
                code |= 1u32 << bit;
            }
            bit += 1;
        }
    }
    code
}

fn small_canon_code_rec(
    k: usize,
    adj: &[u8],
    candidates: &[u8; SMALL_WORK_MAX],
    classes: &[(usize, usize)],
    class_idx: usize,
    out: &mut [u8; SMALL_WORK_MAX],
    best: &mut u32,
) {
    if class_idx == classes.len() {
        *best = (*best).min(permuted_code(k, adj, out));
        return;
    }
    let (lo, hi) = classes[class_idx];
    // Recursive permutation-enumeration helper: the state it threads (graph, class
    // partition, output buffer, running best) is intrinsic, not worth a wrapper struct.
    #[allow(clippy::too_many_arguments)]
    fn fill_class(
        k: usize,
        adj: &[u8],
        candidates: &[u8; SMALL_WORK_MAX],
        classes: &[(usize, usize)],
        class_idx: usize,
        pos: usize,
        hi: usize,
        used: u8,
        out: &mut [u8; SMALL_WORK_MAX],
        best: &mut u32,
    ) {
        if pos == hi {
            small_canon_code_rec(k, adj, candidates, classes, class_idx + 1, out, best);
            return;
        }
        for v in classes[class_idx].0..hi {
            let cand = candidates[v];
            let bit = 1u8 << cand;
            if (used & bit) == 0 {
                out[pos] = cand;
                fill_class(
                    k,
                    adj,
                    candidates,
                    classes,
                    class_idx,
                    pos + 1,
                    hi,
                    used | bit,
                    out,
                    best,
                );
            }
        }
    }
    fill_class(k, adj, candidates, classes, class_idx, lo, hi, 0, out, best);
}

fn small_canon_code(k: usize, adj: &[u8]) -> u32 {
    let sig = small_vertex_sigs(k, adj);
    let mut order = [0u8; SMALL_WORK_MAX];
    for (i, o) in order[..k].iter_mut().enumerate() {
        *o = i as u8;
    }
    order[..k].sort_unstable_by_key(|&v| sig[v as usize]);
    let mut classes = [(0usize, 0usize); SMALL_WORK_MAX];
    let mut nc = 0usize;
    let mut lo = 0usize;
    while lo < k {
        let mut hi = lo + 1;
        while hi < k && sig[order[hi] as usize] == sig[order[lo] as usize] {
            hi += 1;
        }
        classes[nc] = (lo, hi);
        nc += 1;
        lo = hi;
    }
    let mut out = [0u8; SMALL_WORK_MAX];
    let mut best = u32::MAX;
    small_canon_code_rec(k, adj, &order, &classes[..nc], 0, &mut out, &mut best);
    best
}

#[inline]
fn small_key_from_code(k: usize, code: u32) -> u64 {
    let canon = if code == 0 {
        0
    } else {
        let all = if k >= 2 {
            (1u32 << ((k * (k - 1)) / 2)) - 1
        } else {
            0
        };
        if code == all {
            all
        } else {
            small_canon_code(k, &adj_from_edge_code(k, code))
        }
    };
    (((k as u64) << 32) | canon as u64) ^ SMALL_CANON_TAG
}

#[inline(always)]
fn edge_bit(row: Bits, v: u8) -> bool {
    let v = v as usize;
    (row.0[v >> 6] & (1u64 << (v & 63))) != 0
}

#[inline(always)]
fn tiny_edge_code<const K: usize>(attack: &[Bits], verts: &[u8; SMALL_CANON_MAX]) -> u32 {
    let mut code = 0u32;
    let mut bit = 0u32;
    let mut i = 0usize;
    while i < K {
        debug_assert!((verts[i] as usize) < attack.len());
        // SAFETY: vertices are extracted from a board mask, and `attack` has one row per
        // board square.
        let row = unsafe { *attack.get_unchecked(verts[i] as usize) };
        let mut j = i + 1;
        while j < K {
            // Branchless: OR in the edge bit unconditionally (0 or 1). The `if`-guarded
            // form is a data-dependent branch per edge — and `tiny_edge_code` is on the
            // always-run path of `band_entry`, the measured #1 branch-miss site. Same code
            // value (matches `w8_get`'s branchless edge-code build).
            code |= (edge_bit(row, verts[j]) as u32) << bit;
            bit += 1;
            j += 1;
        }
        i += 1;
    }
    code
}

/// The tiny-table iso key for an induced subgraph carried as a **local adjacency** +
/// `alive` bitmask, instead of a 256-bit board mask. `adj[i]` is the neighbour bitmask
/// (over local vertex labels `0..k0`, self bit clear) of vertex `i`; `alive` selects the
/// vertices still present. The result is **byte-identical** to
/// [`Queens::iso_key_tiny_table_pc`] on the corresponding board mask: the tiny-canon
/// table is relabelling-invariant (it maps every labelled edge code of a graph to one
/// canonical key), so the local q.order labelling here and the board-order extraction
/// there land on the same value. This is the iso tail's hot key: no board scan, no
/// 256-bit attack-row load, no vertex re-extraction — only `alive`-bit byte ops. `table`
/// is the prebuilt [`small_canon_table`].
#[inline]
pub(crate) fn tiny_key_from_adj(adj: &[u8; SMALL_WORK_MAX], alive: u8, table: &[u64]) -> u64 {
    let k = alive.count_ones() as usize;
    debug_assert!((1..=SMALL_CANON_MAX).contains(&k));
    if k == 1 {
        return (1u64 << 32) ^ SMALL_CANON_TAG;
    }
    // Collect the alive vertex labels (ascending = q.order), then pack the triangular
    // edge code in that order — the same `i<j` low-to-high convention `tiny_edge_code`
    // and `adj_from_edge_code` use, so the table index matches.
    let mut av = [0u8; SMALL_CANON_MAX];
    let mut n = 0usize;
    let mut a = alive;
    while a != 0 {
        av[n] = a.trailing_zeros() as u8;
        n += 1;
        a &= a - 1;
    }
    let mut code = 0u32;
    let mut bit = 0u32;
    for x in 0..k {
        let ax = adj[av[x] as usize];
        for &vy in av.iter().take(k).skip(x + 1) {
            code |= (((ax >> vy) & 1) as u32) << bit;
            bit += 1;
        }
    }
    if k == 2 {
        return ((2u64 << 32) | code as u64) ^ SMALL_CANON_TAG;
    }
    if k == 3 {
        let canon = (1u64 << code.count_ones()) - 1;
        return ((3u64 << 32) | canon) ^ SMALL_CANON_TAG;
    }
    let idx = SMALL_CANON_OFF[k] + code as usize;
    debug_assert!(idx < table.len());
    // SAFETY: `code` is a triangular edge code for exactly `k <= SMALL_CANON_MAX`
    // vertices, so `idx` is within this table's `[SMALL_CANON_OFF[k], OFF[k+1])`.
    unsafe { *table.get_unchecked(idx) }
}

/// Direct canonical key of a *tiny* connected component (`k <= TINY_MAX`), bypassing
/// CSR construction + WL refinement + certificate hashing (#18). For a connected graph
/// on at most four vertices the **sorted degree sequence is a complete isomorphism
/// invariant** -- the 1 / 1 / 2 / 6 connected graphs on 1..=4 vertices each carry a
/// distinct sorted degree sequence -- so we map straight from `(k, sorted degrees)` to a
/// constant. Deep in the search the available-graph fragments into overwhelmingly such
/// components (the isolated vertex and the edge dominate), and `comp_canon` was
/// recomputing their canon millions of times. The key shares the 64-bit space of the
/// full certificate hash; a collision with a `k >= 5` key is a ~2^-64 event (the search
/// already keys through a 55-bit slot fingerprint and is cross-checked vs Jenrich).
#[inline]
pub(crate) fn tiny_comp_key(attack: &[Bits], comp: Bits, k: usize, verts: &[u8]) -> u64 {
    let mut deg = [0u8; TINY_MAX];
    for i in 0..k {
        // attack[v] includes v itself, so the self-bit is one of the set bits.
        deg[i] = (attack[verts[i] as usize].and(comp).popcount() - 1) as u8;
    }
    deg[..k].sort_unstable();
    // Pack (k, sorted degree sequence) -- each degree is < k <= 4, so fits a byte --
    // into one integer and avalanche it. Distinct sorted degree sequence (and distinct
    // k) ⇒ distinct packed value ⇒ distinct key; the invariant is complete here.
    let mut packed = k as u64;
    for &d in &deg[..k] {
        packed = (packed << 8) | d as u64;
    }
    mix64(packed ^ 0x7111_C0DE_7111_C0DE)
}

/// Individualisation-refinement canonical certificate (see [`Queens::iso_key_canon`]).
/// `nbr_sq` is the square-indexed neighbour lookup; `coloring` is the current vertex
/// colouring (by square); `depth` gives each individualisation level a distinct tag so
/// nested individualisations cannot collide. Returns the canonical adjacency rows, or
/// `None` if the shared `budget` is exhausted.
fn canon_cert(
    verts: &[u32],
    nbrs: &[Bits],
    nbr_sq: &[Bits],
    coloring: Vec<u64>,
    budget: &mut i64,
    depth: u32,
) -> Option<Vec<Bits>> {
    *budget -= 1;
    if *budget < 0 {
        return None;
    }
    let coloring = wl_refine(verts, nbrs, coloring);
    // The target cell: the non-singleton colour class with the smallest colour value
    // (a canonical choice -- the same relative class in isomorphic graphs).
    let mut groups: HashMap<u64, Vec<u32>> = HashMap::new();
    for &s in verts {
        groups.entry(coloring[s as usize]).or_default().push(s);
    }
    let target = groups
        .iter()
        .filter(|(_, vs)| vs.len() > 1)
        .min_by_key(|(&c, _)| c)
        .map(|(_, vs)| vs.clone());
    match target {
        None => {
            // Discrete colouring ⇒ canonical vertex order ⇒ adjacency certificate.
            let mut order = verts.to_vec();
            order.sort_unstable_by_key(|&s| coloring[s as usize]);
            let cert: Vec<Bits> = order
                .iter()
                .map(|&vi| {
                    let mut row = Bits::ZERO;
                    for (j, &vj) in order.iter().enumerate() {
                        if nbr_sq[vi as usize].get(vj) {
                            row.set(j as u32);
                        }
                    }
                    row
                })
                .collect();
            Some(cert)
        }
        Some(cell) => {
            // Branch: individualise each cell vertex apart, recurse, keep the min cert.
            let tag = 0xF1F2_F3F4_0000_0000u64 ^ depth as u64;
            let mut best: Option<Vec<Bits>> = None;
            for &w in &cell {
                let mut c2 = coloring.clone();
                c2[w as usize] = tag;
                let cert = canon_cert(verts, nbrs, nbr_sq, c2, budget, depth + 1)?;
                if best.as_ref().is_none_or(|b| cert < *b) {
                    best = Some(cert);
                }
            }
            best
        }
    }
}

/// Output of [`Queens::module_profile`] — the Node-Kayles **modular-kernel** gate (probe #1 in
/// the lit-levers backlog). All fields are over the available-graph at one search node.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModuleStats {
    /// `popcount(mask)` — the node's piece count (available-square count).
    pub pc: u32,
    /// A vertex adjacent to all `pc-1` others (a move ⇒ instant win; O(K) short-circuit).
    pub has_universal: bool,
    /// Count of **clique modules** (closed-twin classes, size ≥ 2: pairwise-adjacent vertices
    /// sharing one closed neighbourhood).
    pub n_clique_modules: u32,
    /// Count of **independent modules** (open-twin classes, size ≥ 2: pairwise-non-adjacent
    /// vertices sharing one open neighbourhood).
    pub n_indep_modules: u32,
    /// Largest clique-module size (0 if none ≥ 2).
    pub max_clique_module: u32,
    /// Largest independent-module size (0 if none ≥ 2).
    pub max_indep_module: u32,
    /// `pc` after one nimber-preserving kernel pass (clique ≥ 3 → 1 rep; independent ≥ 3 → 2
    /// reps). If this drops to ≤ 12 the shape resolves in the paying W12 frontier — the gate.
    pub reduced_pc: u32,
}

impl Queens {
    /// A Weisfeiler–Leman (1-WL / colour-refinement) **invariant** of the
    /// *available-graph* of `mask`: vertices are the available squares, edges are
    /// attacking pairs (the game from here is Node Kayles on this graph, so any two
    /// positions with isomorphic available-graphs have identical game values and
    /// subtrees). Isomorphic graphs share this value, so counting distinct `iso_key`s
    /// over the working set MEASURES how many positions would merge under graph-
    /// isomorphism canonicalisation -- beyond the 8 board symmetries [`canon`] folds
    /// (the queen graph's automorphisms include D4 but small residual graphs often
    /// coincide up to iso without being board-symmetric, e.g. k mutually-non-attacking
    /// squares = k isolated vertices wherever they sit).
    ///
    /// It is an *invariant*, not a canonical form: non-isomorphic graphs can collide
    /// (1-WL failures -- and these queen available-graphs are WL-hard), so it
    /// over-counts merges. Measurement tool only (`count --iso`), not a TT key. See
    /// [`Queens::iso_key_ir`] for the stronger individualisation-refinement variant.
    pub fn iso_key(&self, mask: Bits) -> u64 {
        let (verts, nbrs, base) = self.avail_graph(mask);
        if verts.is_empty() {
            return 0;
        }
        hash_colours(&verts, &wl_refine(&verts, &nbrs, base))
    }

    /// A **stronger** available-graph invariant: 1-WL augmented by *individualisation*
    /// (the core of nauty/bliss). 1-WL alone is too weak on these regular/symmetric
    /// graphs, so when its colouring is non-discrete we individualise each vertex in
    /// turn (tag it apart, re-refine to stability) and combine the resulting per-vertex
    /// colour signatures into one order-independent invariant. Breaking the regularity
    /// 1-WL chokes on, this distinguishes far more non-isomorphic graphs -- so its
    /// distinct count is a much tighter (still conservative) estimate of the true
    /// graph-isomorphism class count. Still an invariant, not a full canonical form;
    /// O(|V|) refinements per non-discrete graph (cold measurement path only).
    pub fn iso_key_ir(&self, mask: Bits) -> u64 {
        let (verts, nbrs, base) = self.avail_graph(mask);
        if verts.is_empty() {
            return 0;
        }
        let stable = wl_refine(&verts, &nbrs, base.clone());
        // Already discrete ⇒ 1-WL pins every vertex, individualisation adds nothing.
        let distinct = {
            let mut c: Vec<u64> = verts.iter().map(|&s| stable[s as usize]).collect();
            c.sort_unstable();
            c.dedup();
            c.len()
        };
        if distinct == verts.len() {
            return hash_colours(&verts, &stable);
        }
        // Individualise each vertex with the same distinguished tag, refine, and fold
        // the per-vertex signatures (sorted ⇒ vertex order cannot matter).
        let mut sigs: Vec<u64> = verts
            .iter()
            .map(|&v| {
                let mut init = base.clone();
                init[v as usize] = 0xD15C_0DED_1111_2222; // tag distinct from any degree
                hash_colours(&verts, &wl_refine(&verts, &nbrs, init))
            })
            .collect();
        sigs.sort_unstable();
        sigs.iter().fold(0xABCD_1234_5678_9ABC, |h, &c| {
            mix64(h ^ c).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        })
    }

    /// A **true canonical form** of the available-graph (individualisation-refinement,
    /// nauty-style): refine; if the colouring is discrete, read off the adjacency in
    /// the colour-induced vertex order as a certificate; else branch on each vertex of
    /// the first non-singleton (smallest-colour) cell, individualising it apart, and
    /// take the lexicographically **minimum** certificate over the branches. Two graphs
    /// get the same certificate **iff** isomorphic — so distinct `iso_key_canon`s over
    /// the working set is the *exact* graph-isomorphism class count (the safe merge),
    /// and a correct canon can never make a win/loss-mixed class.
    ///
    /// Symmetric graphs branch widely, so a node budget caps the search; a capped graph
    /// falls back to the (sound, weaker) [`iso_key_ir`] invariant — which may over-merge
    /// and so surface as a mixed class, flagging that this graph wasn't fully canonised.
    pub fn iso_key_canon(&self, mask: Bits) -> u64 {
        let (verts, nbrs, base) = self.avail_graph(mask);
        if verts.is_empty() {
            return 0;
        }
        // Square-indexed neighbour lookup, for adjacency tests when serialising.
        let mut nbr_sq = vec![Bits::ZERO; (self.n * self.n) as usize];
        for (&s, &nb) in verts.iter().zip(&nbrs) {
            nbr_sq[s as usize] = nb;
        }
        let mut budget: i64 = 200_000;
        match canon_cert(&verts, &nbrs, &nbr_sq, base, &mut budget, 0) {
            Some(cert) => cert.iter().flat_map(|r| r.0).fold(0x0CA7_F00D_u64, |h, x| {
                mix64(h ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            }),
            None => self.iso_key_ir(mask), // budget exhausted: fall back to the invariant
        }
    }

    /// An **allocation-free** graph-isomorphism key (the production-candidate live key):
    /// component-decompose `mask` and canonicalise each component using only the
    /// per-thread preallocated [`IsoScratch`] buffers -- no `Vec`/`HashMap`, no heap
    /// allocation, no per-call zeroing (every buffer cell is written before it is read).
    /// Same merge as [`iso_key_canon`] (validated equal on n ≤ 14), far cheaper.
    pub fn iso_key_fast(&self, mask: Bits) -> u64 {
        ISO_SCRATCH.with(|s| {
            let mut g = s.borrow_mut();
            // Production: the `HIST = false` instantiation emits *no* component-size
            // tally at all -- not a per-component branch but a compile-time-eliminated
            // path -- so the hot loop's I-cache footprint is unchanged. The gate is a
            // const generic resolved at the call site, the way the project keeps
            // measurement toggles out of latency-bound loops (see the env-var rule in
            // CLAUDE.md). The measurement entry instantiates `HIST = true` instead.
            // `CACHE = true` keeps the per-thread component-canon cache (#19).
            self.iso_key_fast_in::<false, true>(mask, &mut g, &mut [])
        })
    }

    /// Measurement variant of [`iso_key_fast`] with the per-thread component-canon
    /// cache (#19) **bypassed**: every component above the [`tiny_comp_key`] shortcut
    /// recomputes its WL canon. In the live n=16 search the component working set
    /// dwarfs the 64 MB cache (its hit rate is low -- doubling the cache moves the n=16
    /// rate only ~3.7%), so this no-cache cost is the **live-representative** per-key
    /// cost and the right gate for the iso-key optimisation: a cache-warmed bench loop
    /// would hit on every repeat and hide every `comp_canon_full` change. Returns the
    /// same value as `iso_key_fast` -- only the cache lookup/store is skipped.
    /// `CACHE = false` is a compile-time-eliminated path, no per-component branch.
    pub fn iso_key_fast_nocache(&self, mask: Bits) -> u64 {
        ISO_SCRATCH.with(|s| {
            let mut g = s.borrow_mut();
            self.iso_key_fast_in::<false, false>(mask, &mut g, &mut [])
        })
    }

    /// Exact whole-graph canonical key for tiny available graphs (`popcount <= 7`),
    /// backed by an eagerly precomputed table indexed by the labelled adjacency code.
    /// `iso-burr` uses this instead of component decomposition/WL for its default
    /// selective `iso<=7` path.
    pub(crate) fn iso_key_tiny_table(&self, mask: Bits) -> u64 {
        self.iso_key_tiny_table_in(mask, small_canon_table())
    }

    /// Hot-path variant of [`Self::iso_key_tiny_table`] for callers that already carry the
    /// prebuilt tiny-canon table, avoiding the per-node [`OnceLock`] check.
    #[inline]
    pub(crate) fn iso_key_tiny_table_in(&self, mask: Bits, table: &[u64]) -> u64 {
        self.iso_key_tiny_table_pc(mask, mask.popcount(), table)
    }

    /// Hot-path variant of [`Self::iso_key_tiny_table_in`] for callers that already computed
    /// `mask.popcount()` while deciding whether the tiny key applies.
    #[inline]
    pub(crate) fn iso_key_tiny_table_pc(&self, mask: Bits, pc: u32, table: &[u64]) -> u64 {
        let k = pc as usize;
        debug_assert!(k <= SMALL_CANON_MAX);
        if k == 0 {
            return 0;
        }
        if k == 1 {
            return (1u64 << 32) ^ SMALL_CANON_TAG;
        }
        let mut verts = [0u8; SMALL_CANON_MAX];
        let mut n = 0usize;
        mask.each(|v| {
            verts[n] = v as u8;
            n += 1;
        });
        // Build the triangular edge code directly: one attack test per `i<j` pair, packed
        // low-to-high in `(i,j)` lexicographic order (the index convention `adj_from_edge_code`
        // inverts). Queen attacks are mutual, so the `i<j` direction is the whole edge — this is
        // bit-identical to building the full adjacency then re-scanning the upper triangle, at
        // half the attack tests and no adj-array writes.
        let code = match k {
            2 => tiny_edge_code::<2>(&self.attack, &verts),
            3 => tiny_edge_code::<3>(&self.attack, &verts),
            4 => tiny_edge_code::<4>(&self.attack, &verts),
            5 => tiny_edge_code::<5>(&self.attack, &verts),
            6 => tiny_edge_code::<6>(&self.attack, &verts),
            7 => tiny_edge_code::<7>(&self.attack, &verts),
            _ => unreachable!(),
        };
        if k == 2 {
            return ((2u64 << 32) | code as u64) ^ SMALL_CANON_TAG;
        }
        if k == 3 {
            let canon = (1u64 << code.count_ones()) - 1;
            return ((3u64 << 32) | canon) ^ SMALL_CANON_TAG;
        }
        let idx = SMALL_CANON_OFF[k] + code as usize;
        debug_assert!(idx < table.len());
        // SAFETY: `code` is a triangular edge code for exactly `k <= SMALL_CANON_MAX`
        // vertices, so `idx` is within this table's `[SMALL_CANON_OFF[k], OFF[k+1])`.
        unsafe { *table.get_unchecked(idx) }
    }

    /// The **labelled** dense slot `SMALL_CANON_OFF[k] + edge_code` of a ≤7 graph, computed
    /// like [`Self::iso_key_tiny_table_pc`] but **without the canon-table lookup** (and
    /// without canonicalising the `k==3` code). A ≤7 position's Node-Kayles win/loss is
    /// isomorphism-invariant, so a win/loss table keyed by this raw index is still correct
    /// — it just stores the value under every labelling instead of merging to the canonical
    /// form. That trades a little merge (cheap to recompute in an L1 memo) for skipping the
    /// 16 MB canon table, whose scattered L3/DRAM probe is ~22% of the n=16 search. The
    /// result is `< `[`TINY_TABLE_SLOTS`].
    #[inline]
    pub(crate) fn tiny_table_index(&self, mask: Bits, pc: u32) -> usize {
        let k = pc as usize;
        debug_assert!((1..=SMALL_CANON_MAX).contains(&k));
        if k == 1 {
            return SMALL_CANON_OFF[1];
        }
        let mut verts = [0u8; SMALL_CANON_MAX];
        let mut n = 0usize;
        mask.each(|v| {
            verts[n] = v as u8;
            n += 1;
        });
        let code = match k {
            2 => tiny_edge_code::<2>(&self.attack, &verts),
            3 => tiny_edge_code::<3>(&self.attack, &verts),
            4 => tiny_edge_code::<4>(&self.attack, &verts),
            5 => tiny_edge_code::<5>(&self.attack, &verts),
            6 => tiny_edge_code::<6>(&self.attack, &verts),
            7 => tiny_edge_code::<7>(&self.attack, &verts),
            _ => unreachable!(),
        };
        SMALL_CANON_OFF[k] + code as usize
    }

    /// Exact whole-graph canonical key for an 8-vertex available graph. This is the
    /// measurement/prototype twin of [`Self::iso_key_tiny_table`]: same edge-code canon,
    /// but computed on demand so we can test the k=8 payoff without allocating a large
    /// dense table or paying its startup build.
    pub(crate) fn iso_key8_direct(&self, mask: Bits) -> u64 {
        debug_assert_eq!(mask.popcount(), 8);
        let mut verts = [0u8; SMALL_WORK_MAX];
        let mut n = 0usize;
        mask.each(|v| {
            verts[n] = v as u8;
            n += 1;
        });
        let mut code = 0u32;
        let mut bit = 0u32;
        for i in 0..8 {
            let vi = verts[i] as usize;
            debug_assert!(vi < self.attack.len());
            // SAFETY: `verts` was extracted from `mask`, which only contains board squares.
            let row = unsafe { *self.attack.get_unchecked(vi) };
            for &vj in verts.iter().skip(i + 1) {
                code |= (edge_bit(row, vj) as u32) << bit; // branchless (see tiny_edge_code)
                bit += 1;
            }
        }
        K8_CANON_CACHE.with(|c| c.borrow_mut().get_or_insert(code))
    }

    /// Measurement entry for `count --comps`: run the *same* graph-key decomposition the
    /// live key runs, but with the connected-component-size tally monomorphised in
    /// (`HIST = true`). Each available-graph's component sizes are accumulated into
    /// `hist` (bucket `i` = components with `i` vertices; the final bucket catches the
    /// tail). Cold analysis only -- never reached from the search.
    pub fn tally_components(&self, mask: Bits, hist: &mut [u64]) {
        ISO_SCRATCH.with(|s| {
            let mut g = s.borrow_mut();
            self.iso_key_fast_in::<true, true>(mask, &mut g, hist);
        });
    }

    /// Cold analysis helper for selective graph-key experiments: largest connected
    /// component of the available-graph. This is intentionally separate from the hot
    /// key path; a production max-component gate should fuse this decomposition with
    /// component canon so selected positions do not decompose twice.
    pub fn iso_max_component_size(&self, mask: Bits) -> u32 {
        let mut remaining = mask;
        let mut max_k = 0u32;
        while let Some(start) = remaining.lowest() {
            let comp = self.component(start, mask);
            remaining = remaining.and_not(comp);
            max_k = max_k.max(comp.popcount());
        }
        max_k
    }

    /// Cold analysis: one decomposition pass over `mask` returning `(pc, max_comp,
    /// ncomp)` -- popcount, largest connected component size, and number of connected
    /// components of the available-graph. Fuses what [`Self::iso_max_component_size`]
    /// and a separate component-count loop would each do, so `count --comps`'s
    /// dense-nimber-table incremental-coverage tally decomposes the mask only once.
    pub fn component_profile(&self, mask: Bits) -> (u32, u32, u32) {
        let pc = mask.popcount();
        let mut remaining = mask;
        let mut max_k = 0u32;
        let mut ncomp = 0u32;
        while let Some(start) = remaining.lowest() {
            let comp = self.component(start, mask);
            remaining = remaining.and_not(comp);
            max_k = max_k.max(comp.popcount());
            ncomp += 1;
        }
        (pc, max_k, ncomp)
    }

    /// Cold analysis (`count --comps` slice-finder): one decomposition pass returning the
    /// fragmentation shape `(pc, ncomp, maxc, second, n_iso)` — popcount, component count,
    /// largest and second-largest component sizes, and the number of **isolated** available
    /// squares (degree 0 in the available-graph — each a size-1 component). `n_iso` is the
    /// *cheap* hot-path-computable predictor (one masked popcount per vertex, no BFS), so the
    /// slice-finder can check whether it identifies the high-fire (all-components-≤K) slice
    /// without a full decomposition gate.
    pub fn frag_profile(&self, mask: Bits) -> (u32, u32, u32, u32, u32) {
        let pc = mask.popcount();
        let mut remaining = mask;
        let (mut maxc, mut second, mut ncomp) = (0u32, 0u32, 0u32);
        while let Some(start) = remaining.lowest() {
            let comp = self.component(start, mask);
            remaining = remaining.and_not(comp);
            let sz = comp.popcount();
            if sz >= maxc {
                second = maxc;
                maxc = sz;
            } else if sz > second {
                second = sz;
            }
            ncomp += 1;
        }
        let mut n_iso = 0u32;
        mask.each(|v| {
            if self.attack[v as usize].and(mask).popcount() == 1 {
                n_iso += 1; // only itself in N[v]∩avail ⇒ no available attacker ⇒ isolated
            }
        });
        (pc, ncomp, maxc, second, n_iso)
    }

    /// Cold analysis (`count --comps`): decompose `mask` and hand each connected
    /// component's `(size, comp_canon)` to `f`. Uses the **measurement-exact** canon
    /// (`comp_canon::<false>` -- the complete WL/IR certificate, cache bypassed) so the
    /// distinct-key census is the true distinct-canonical-component count, not a cached
    /// approximation. Drives the C1 dense-component-value-table go/no-go: how many distinct
    /// canonical components arise per size (the table's cache footprint) and how often each
    /// recurs (the amortisation multiplicity).
    pub fn each_comp_canon(&self, mask: Bits, mut f: impl FnMut(u32, u64)) {
        ISO_SCRATCH.with(|s| {
            let mut g = s.borrow_mut();
            let mut remaining = mask;
            while let Some(start) = remaining.lowest() {
                let comp = self.component(start, mask);
                remaining = remaining.and_not(comp);
                let key = self.comp_canon::<false>(comp, &mut g);
                f(comp.popcount(), key);
            }
        });
    }

    /// Cold, read-only `count --roots` instrumentation: the per-root structural proxies
    /// for one symmetry-distinct first move `sq`, returned as
    /// `(centrality, avail_pop, frag, ncomp)`:
    ///   - `centrality` = attack degree of the first move (its forcing weight),
    ///   - `avail_pop`  = popcount of the residual available mask after placing `sq`,
    ///   - `frag`       = largest connected component of that available-graph,
    ///   - `ncomp`      = number of connected components of that available-graph.
    ///
    /// Lives beside [`Self::iso_max_component_size`] (same cold decomposition) so the bin
    /// crate need not reach the hot-path-private board/attack fields or `Bits` ops.
    pub fn root_proxies(&self, sq: u32) -> (u32, u32, u32, u32) {
        let available = self.board.and_not(self.place(Bits::empty(), sq));
        let centrality = self.attack[sq as usize].popcount();
        let avail_pop = available.popcount();
        let frag = self.iso_max_component_size(available);
        let mut remaining = available;
        let mut ncomp = 0u32;
        while let Some(start) = remaining.lowest() {
            let comp = self.component(start, available);
            remaining = remaining.and_not(comp);
            ncomp += 1;
        }
        (centrality, avail_pop, frag, ncomp)
    }

    /// Cold `count --comps` structural-incidence probe for the **Node-Kayles reductions** the
    /// literature offers as a cheaper-than-`pext` shortcut for the W_K layer. For the
    /// available-graph on `mask` returns `(has_universal, has_twin)`:
    ///   - **universal vertex** — a vertex adjacent to all `K-1` others; a move on it deletes
    ///     `N[v] =` everything, so the mover wins immediately (the position is an N-position).
    ///     Detectable in O(K), it would short-circuit the whole K-move child sweep.
    ///   - **twin pair** — two vertices `u,v` whose neighbourhoods agree outside `{u,v}` (true
    ///     or false twins); they are equivalent moves, so the sweep can branch on one and skip
    ///     the other (and a chain of twins is an independent/clique module that collapses `K`).
    ///
    /// Measures how often these fire on the queen subgraphs at pc==K, the gate for whether a
    /// reduction pre-pass would pay. O(K²) `Bits` ops; only called from `count --comps`.
    pub fn struct_profile(&self, mask: Bits) -> (bool, bool) {
        let k = mask.popcount() as usize;
        let mut verts = [0u32; 16];
        let mut nbhd = [Bits::ZERO; 16];
        if k == 0 || k > verts.len() {
            return (false, false);
        }
        let mut n = 0usize;
        mask.each(|v| {
            verts[n] = v;
            // open neighbourhood within the available set (attack excludes self)
            nbhd[n] = self.attack[v as usize].and(mask);
            n += 1;
        });
        let mut has_universal = false;
        for &nb in nbhd.iter().take(n) {
            if nb.popcount() as usize == k - 1 {
                has_universal = true;
                break;
            }
        }
        let mut has_twin = false;
        'outer: for i in 0..n {
            for j in (i + 1)..n {
                // u,v are twins iff N(u) and N(v) differ only (possibly) in each other's bit.
                let diff = nbhd[i].and_not(nbhd[j]).or(nbhd[j].and_not(nbhd[i]));
                let extra = diff.popcount() - diff.get(verts[i]) as u32 - diff.get(verts[j]) as u32;
                if extra == 0 {
                    has_twin = true;
                    break 'outer;
                }
            }
        }
        (has_universal, has_twin)
    }

    /// Probe #1 — the Node-Kayles **modular-reduction** gate (item A in the lit-levers backlog).
    /// Cold, O(K²) `Bits` ops; only called from the `count --comps` report (zero production cost).
    ///
    /// Partitions the available-graph on `mask` into **twin classes** and applies the
    /// Grundy-preserving modular kernel (Kobayashi, Node Kayles by modular-width):
    ///   - a **clique module** is a *closed*-twin class — vertices with identical closed
    ///     neighbourhood `N[v]` (hence pairwise adjacent). Kernel: size ≥ 3 → keep 1 rep.
    ///   - an **independent module** is an *open*-twin class — identical open neighbourhood
    ///     `N(v)` (hence pairwise non-adjacent). Kernel: size ≥ 3 → keep 2 reps.
    ///
    /// Equality of the (closed/open) neighbourhood is a genuine equivalence, so one rep-scan
    /// per partition is exact. Returns [`ModuleStats`] including `reduced_pc` after one pass —
    /// **if that drops to ≤ 12 the shape resolves in the paying W12 frontier** instead of a
    /// flat-TT recurse, which is exactly what a reduce-then-W12 evaluator (item A) would exploit.
    /// One pass only (no fixpoint) — a deliberate lower bound on the kernel for the prevalence gate.
    pub fn module_profile(&self, mask: Bits) -> ModuleStats {
        let k = mask.popcount();
        let mut out = ModuleStats {
            pc: k,
            reduced_pc: k,
            ..ModuleStats::default()
        };
        const CAP: usize = 24;
        let kk = k as usize;
        if kk == 0 || kk > CAP {
            return out; // pc > 24 won't occur in the probed bands; report unreduced.
        }
        let mut verts = [0u32; CAP];
        let mut open = [Bits::ZERO; CAP]; // N(v) ∩ mask
        let mut closed = [Bits::ZERO; CAP]; // N[v] ∩ mask = N(v) ∪ {v}
        let mut n = 0usize;
        mask.each(|v| {
            // `attack[v]` includes self, so `attack[v] ∩ mask` is the *closed* neighbourhood N[v];
            // strip v for the true open neighbourhood N(v). Open-vs-closed equality is exactly what
            // separates independent modules (false twins) from clique modules (true twins) —
            // conflating them silently drops every independent module.
            let nclosed = self.attack[v as usize].and(mask);
            verts[n] = v;
            closed[n] = nclosed;
            open[n] = nclosed.and_not(single(v));
            n += 1;
        });
        // Universal vertex: open-degree k-1 (adjacent to every other) ⇒ an N-position.
        for &nb in open.iter().take(n) {
            if nb.popcount() == k - 1 {
                out.has_universal = true;
                break;
            }
        }
        // Clique modules: group by equal closed neighbourhood; contract ≥3 → 1 rep.
        let mut removed = Bits::ZERO; // vertices contracted away (so the indep pass won't re-count)
        let mut seen = [false; CAP];
        for i in 0..n {
            if seen[i] {
                continue;
            }
            let mut members = [0usize; CAP];
            members[0] = i;
            let mut m = 1usize;
            for j in (i + 1)..n {
                if !seen[j] && closed[j] == closed[i] {
                    seen[j] = true;
                    members[m] = j;
                    m += 1;
                }
            }
            seen[i] = true;
            let sz = m as u32;
            if sz >= 2 {
                out.n_clique_modules += 1;
                out.max_clique_module = out.max_clique_module.max(sz);
            }
            if sz >= 3 {
                out.reduced_pc -= sz - 1;
                for &mi in members.iter().take(m).skip(1) {
                    removed.set(verts[mi]);
                }
            }
        }
        // Independent modules: group by equal open neighbourhood among not-yet-removed
        // vertices; contract ≥3 → 2 reps.
        let mut seen = [false; CAP];
        for i in 0..n {
            if seen[i] || removed.get(verts[i]) {
                continue;
            }
            let mut sz = 1u32;
            for j in (i + 1)..n {
                if !seen[j] && !removed.get(verts[j]) && open[j] == open[i] {
                    seen[j] = true;
                    sz += 1;
                }
            }
            seen[i] = true;
            if sz >= 2 {
                out.n_indep_modules += 1;
                out.max_indep_module = out.max_indep_module.max(sz);
            }
            if sz >= 3 {
                out.reduced_pc -= sz - 2;
            }
        }
        out
    }

    /// `HIST` selects, at monomorphisation time, whether to tally component sizes into
    /// `hist` -- `false` for the search's live key (the tally vanishes), `true` for the
    /// `count --comps` measurement. Keeping it a const generic (rather than a runtime
    /// flag) is the project rule for hot-path toggles: the disabled branch never enters
    /// the instruction stream, so it cannot pollute L1i or the frontend the graph key is
    /// already bound by.
    fn iso_key_fast_in<const HIST: bool, const CACHE: bool>(
        &self,
        mask: Bits,
        s: &mut IsoScratch,
        hist: &mut [u64],
    ) -> u64 {
        let mut remaining = mask;
        let mut nc = 0usize;
        while let Some(start) = remaining.lowest() {
            let comp = self.component(start, mask);
            remaining = remaining.and_not(comp);
            if HIST {
                let k = comp.popcount() as usize;
                hist[k.min(hist.len() - 1)] += 1;
            }
            let ck = self.comp_canon::<CACHE>(comp, s);
            s.comp_keys[nc] = ck;
            nc += 1;
        }
        fold_comp_keys(&mut s.comp_keys[..nc])
    }

    /// Canonical key of one connected component, scratch-only. 1-WL refine; if discrete,
    /// hash the adjacency certificate in canonical order (a complete canon); else fall
    /// back to the validated-equivalent individualisation invariant. Components are small
    /// (the graph fragments deep), so the fallback stays cheap.
    pub(crate) fn comp_canon<const CACHE: bool>(&self, comp: Bits, s: &mut IsoScratch) -> u64 {
        let mut k = 0usize;
        comp.each(|v| {
            s.verts[k] = v as u8;
            k += 1;
        });
        // Tiny components -- the deep majority -- resolve by sorted degree sequence
        // alone (a complete invariant for k <= 4), skipping all WL work (#18).
        if k <= TINY_MAX {
            return tiny_comp_key(&self.attack, comp, k, &s.verts);
        }
        // Measurement (`CACHE = false`): bypass the cache, always recompute -- the
        // live-representative cost (see `iso_key_fast_nocache`). Compile-time path.
        if !CACHE {
            return self.comp_canon_full(comp, k, s);
        }
        // #19: amortise the full canon (a pure function of `comp`) across recurring
        // components via the per-thread cache. Probe; on a fingerprint hit return it,
        // else compute and store. The borrow is dropped around `comp_canon_full` so the
        // (recursive-free) compute never holds the cache lock.
        let (slot, fp) = CompCache::probe(comp, self.n);
        if let Some(v) = COMP_CACHE.with(|c| {
            let c = c.borrow();
            (c.fp[slot] == fp).then(|| c.val[slot])
        }) {
            return v;
        }
        let v = self.comp_canon_full(comp, k, s);
        COMP_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            c.fp[slot] = fp;
            c.val[slot] = v;
        });
        v
    }

    /// The full Weisfeiler-Leman canon of a component whose vertices are already in
    /// `s.verts[..k]` -- 1-WL refine, then the adjacency certificate if discrete, else
    /// the individualisation invariant. Used for `k > TINY_MAX` (tiny components take
    /// the [`tiny_comp_key`] shortcut). Kept as a named entry so the test corpus can
    /// cross-check the shortcut against it on small components too.
    pub(crate) fn comp_canon_full(&self, comp: Bits, k: usize, s: &mut IsoScratch) -> u64 {
        // Stride = the component's max degree (one branchless popcount per vertex, no
        // bit-scan), so every padded neighbour row is the same fixed length.
        let mut stride = 0usize;
        let mut clique = true;
        for i in 0..k {
            let deg = self.attack[s.verts[i] as usize].and(comp).popcount() as usize - 1;
            clique &= deg == k - 1;
            if deg > stride {
                stride = deg;
            }
        }
        if clique {
            return mix64((k as u64) ^ 0xC11C_EC11_CEC1_1C1E);
        }
        if k == 5 {
            return canon5_key(&self.attack, &s.verts);
        }
        if k == 6 {
            return canon6_key(&self.attack, &s.verts);
        }
        // Invert verts so neighbour squares map to compact local indices 0..k.
        for i in 0..k {
            s.loc[s.verts[i] as usize] = i as u16;
        }
        // Build the fixed-stride neighbour table once (one bit-scan per vertex, not per
        // round): real neighbours as *local* indices, then DUMMY_VERT padding. Seed the
        // compact colour `lcol[i] = base[i]` by degree.
        for i in 0..k {
            let v = s.verts[i] as usize;
            let base = i * stride;
            let mut p = base;
            self.attack[v].and(comp).each(|t| {
                if t != v as u32 {
                    s.nbr_pad[p] = s.loc[t as usize];
                    p += 1;
                }
            });
            for q in p..base + stride {
                s.nbr_pad[q] = DUMMY_VERT as u16;
            }
            s.base[i] = ((p - base) as u64) | 0x9E37_79B9_0000_0000;
            s.col[i] = s.base[i];
        }
        wl_refine_in(
            k,
            stride,
            &s.nbr_pad,
            &mut s.col,
            &mut s.nxt,
            &mut s.mc,
            &mut s.sort,
        );
        if classes_in(k, &s.col, &mut s.sort) == k {
            return cert_hash_in(&self.attack, comp, k, &s.verts, &s.col, &mut s.order);
        }
        // Non-discrete: individualise only the vertices in *non-singleton* 1-WL classes
        // and combine the signatures. A vertex already alone in its colour class is
        // pinned by 1-WL -- individualising it re-runs WL to a colouring fully determined
        // by the stable partition, so its signature adds nothing the stable colouring
        // does not already fix. Isomorphisms preserve 1-WL colours (hence singleton-ness),
        // so the multiset of non-singleton signatures is still an iso-invariant -- the
        // same merge, far fewer WL re-runs (k -> the count of WL-indistinguishable
        // vertices, the only ones that matter).
        //
        // Fold in the base stable-colouring hash once so dropping the singleton sigs
        // cannot weaken the invariant: those sigs are a deterministic function of the
        // stable colouring, which this hash captures in full.
        let base_hash = hash_colours_in(k, &s.col, &mut s.sort);
        // Snapshot the non-singleton locals from the stable colouring into `s.order`
        // (free here -- `cert_hash_in`/`order` is the discrete path, already returned).
        let mut ns = 0usize;
        for i in 0..k {
            let ci = s.col[i];
            if (0..k).any(|j| j != i && s.col[j] == ci) {
                s.order[ns] = i as u8;
                ns += 1;
            }
        }
        // Collapse the cheap automorphism orbits inside each stable colour class:
        // if two local vertices are twins, individualising either one gives the same
        // signature. Compute the representative once and duplicate its signature so
        // the final multiset hash stays exactly equivalent to the old all-vertices
        // fold, just with fewer WL reruns.
        let mut group = [0u8; MAXV];
        let mut reps = [0u8; MAXV];
        let mut nr = 0usize;
        for (idx, &ord) in s.order[..ns].iter().enumerate() {
            let i = ord as usize;
            let mut g = nr;
            for (r, &rep) in reps[..nr].iter().enumerate() {
                let j = rep as usize;
                if s.col[i] == s.col[j] && twin_vertices(&self.attack, comp, s.verts[i], s.verts[j])
                {
                    g = r;
                    break;
                }
            }
            if g == nr {
                reps[nr] = i as u8;
                nr += 1;
            }
            group[idx] = g as u8;
        }
        for idx in 0..ns {
            let g = group[idx] as usize;
            if reps[g] != s.order[idx] {
                continue;
            }
            let i = reps[g] as usize;
            s.col[..k].copy_from_slice(&s.base[..k]);
            s.col[i] = 0xD15C_0DED_1111_2222;
            wl_refine_in(
                k,
                stride,
                &s.nbr_pad,
                &mut s.col,
                &mut s.nxt,
                &mut s.mc,
                &mut s.sort,
            );
            let sig = hash_colours_in(k, &s.col, &mut s.sort);
            for (j, &gg) in group[..ns].iter().enumerate() {
                if gg as usize == g {
                    s.sigs[j] = sig;
                }
            }
        }
        let sigs = &mut s.sigs[..ns];
        sigs.sort_unstable();
        sigs.iter()
            .fold(mix64(0xABCD_1234_5678_9ABC ^ base_hash), |h, &x| {
                mix64(h ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            })
    }

    /// The connected component of the available-graph containing `start` (flood-fill
    /// over attacking edges within `mask`).
    pub(crate) fn component(&self, start: u32, mask: Bits) -> Bits {
        let mut comp = single(start);
        let mut frontier = comp;
        loop {
            let mut next = Bits::ZERO;
            frontier.each(|v| next = next.or(self.attack[v as usize]));
            next = next.and(mask).and_not(comp);
            if next == Bits::ZERO {
                break;
            }
            comp = comp.or(next);
            frontier = next;
        }
        comp
    }

    /// A **cheaper** graph-isomorphism key: split the available-graph into connected
    /// components and canonicalise each independently, then combine the sorted multiset
    /// of component keys. Sound and complete (two graphs are isomorphic iff their
    /// components match up to iso), and far cheaper on the deep, *fragmented* graphs
    /// that dominate the search -- a whole-graph [`iso_key_canon`] over k isolated
    /// vertices blows its individualisation budget, whereas here each tiny component
    /// canonises instantly. Gives the **same merge** as `iso_key_canon`, faster.
    pub fn iso_key_components(&self, mask: Bits) -> u64 {
        let mut remaining = mask;
        let mut keys: Vec<u64> = Vec::new();
        while let Some(start) = remaining.lowest() {
            let comp = self.component(start, mask);
            remaining = remaining.and_not(comp);
            keys.push(self.iso_key_canon(comp));
        }
        if keys.is_empty() {
            return 0;
        }
        keys.sort_unstable();
        keys.iter().fold(0x515E_AF00_D515_E5A1, |h, &k| {
            mix64(h ^ k).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        })
    }

    /// The available-graph of `mask`: its vertices (set squares), each vertex's
    /// neighbour mask (attacking available squares), and the degree-seeded initial
    /// 1-WL colours indexed by square. Shared by [`iso_key`] and [`iso_key_ir`].
    fn avail_graph(&self, mask: Bits) -> (Vec<u32>, Vec<Bits>, Vec<u64>) {
        let mut verts: Vec<u32> = Vec::new();
        mask.each(|s| verts.push(s));
        let nbrs: Vec<Bits> = verts
            .iter()
            .map(|&s| self.attack[s as usize].and(mask).and_not(single(s)))
            .collect();
        let mut base = vec![0u64; (self.n * self.n) as usize];
        for (&s, nb) in verts.iter().zip(&nbrs) {
            base[s as usize] = nb.popcount() as u64 | 0x9E37_79B9_0000_0000;
        }
        (verts, nbrs, base)
    }

    /// A realistic deep corpus of *raw available* masks for the iso-key benchmark
    /// (`src/bin/iso_key_bench.rs`): a DFS over the real move geometry that dedups on
    /// the D4 canonical key (exactly the live search's TT key) and records each
    /// newly-seen position's raw `available = board & !blocked` mask once, capped at
    /// `cap` distinct positions. Because dedup is by `canon`, the returned masks have
    /// pairwise-distinct D4 keys, so `cap` (or fewer at exhaustion) D4 classes are
    /// present; recording the *raw* (pre-canon) mask -- one orbit member per class --
    /// gives the iso key the realistic mix of orientations it is called on per node in
    /// a live D4 search. Cold measurement path (allocates a dedup set); never reached
    /// from the search.
    pub fn iso_corpus(&self, cap: usize) -> Vec<Bits> {
        fn dfs(
            q: &Queens,
            blocked: Bits,
            seen: &mut HashSet<Bits>,
            out: &mut Vec<Bits>,
            cap: usize,
        ) {
            if out.len() >= cap {
                return;
            }
            let available = q.board.and_not(blocked);
            if !seen.insert(q.canon(available)) {
                return; // transposition -- the live TT would prune here
            }
            out.push(available);
            for &sq in &q.order {
                if out.len() >= cap {
                    return;
                }
                if !q.is_available(blocked, sq) {
                    continue;
                }
                let child = q.place(blocked, sq);
                if q.no_moves(child) {
                    continue; // terminal: nothing below to key
                }
                dfs(q, child, seen, out, cap);
            }
        }
        let mut seen = HashSet::with_capacity(cap * 2);
        let mut out = Vec::with_capacity(cap);
        dfs(self, Bits::ZERO, &mut seen, &mut out, cap);
        out
    }
}
