//! `Nimber` -- full Sprague-Grundy value via mex (no cutoff).

use super::*;
use crate::queens::dense::{DenseW8, GrundyW8};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU16, Ordering};

/// **Nimber** -- computes the full Sprague-Grundy value (not just win/loss) by
/// `mex` over the children's nimbers. Because `mex` needs the whole child set,
/// there is **no α-β cutoff**, so this is strictly more work than the win/loss
/// solvers; it earns its keep by yielding the exact game value (OEIS A344227) and
/// the machinery a future Grundy-decomposition / df-pn solver would build on. As
/// a [`Solver`] it reports `wins = (nimber != 0)`.
pub struct Nimber {
    tt: QueensTt,
    /// The nimber of the empty board once `wins` computes it (`u16::MAX` until
    /// then), so the solve summary can report the actual Sprague-Grundy value.
    root: AtomicU16,
}

impl Nimber {
    pub fn new(bits: u32) -> Self {
        Nimber {
            tt: QueensTt::new(bits),
            root: AtomicU16::new(u16::MAX),
        }
    }

    /// The Sprague-Grundy value (nimber) of the position. `0` is a P-position
    /// (loss for the player to move); any non-zero value is an N-position (win).
    /// Sequential: `mex` needs every child, so there is no cutoff.
    pub fn grundy(&self, q: &Queens, blocked: Bits) -> u8 {
        let key = q.pos_key(blocked);
        if let Some(v) = self.tt.get(key) {
            return v;
        }
        self.tt.bump();
        let mut seen = 0u64; // bitset of child nimbers (all are < n <= 16 < 64)
        for &sq in &q.order {
            if q.is_available(blocked, sq) {
                seen |= 1u64 << self.grundy(q, q.place(blocked, sq));
            }
        }
        let mex = (!seen).trailing_zeros() as u8; // smallest value not in `seen`
        self.tt.put(key, mex);
        mex
    }

    /// As `grundy`, but the top `par_levels` levels fan their children across
    /// rayon workers (sharing the table). Since `mex` has no cutoff, every child
    /// must be computed regardless, so this is pure speedup with no speculative
    /// waste -- unlike the win/loss search it needs no Young-Brothers-Wait guard.
    fn grundy_par(&self, q: &Queens, blocked: Bits, par_levels: u32) -> u8 {
        if par_levels == 0 {
            return self.grundy(q, blocked);
        }
        let key = q.pos_key(blocked);
        if let Some(v) = self.tt.get(key) {
            return v;
        }
        self.tt.bump();
        let kids: Vec<Bits> = q
            .order
            .iter()
            .filter(|&&sq| q.is_available(blocked, sq))
            .map(|&sq| q.place(blocked, sq))
            .collect();
        let seen = kids
            .par_iter()
            .map(|&c| 1u64 << self.grundy_par(q, c, par_levels - 1))
            .reduce(|| 0u64, |a, b| a | b);
        let mex = (!seen).trailing_zeros() as u8;
        self.tt.put(key, mex);
        mex
    }

    /// The nimber of the empty board, root-parallel. The root `mex` is over the
    /// children's nimbers, and symmetric first moves have equal nimbers, so the
    /// dihedral-distinct representatives suffice; they are independent subtrees,
    /// computed across rayon workers (and one level deeper, for load balance).
    pub fn nimber(&self, q: &Queens) -> u8 {
        let seen = q
            .distinct_first_moves()
            .par_iter()
            .map(|&sq| 1u64 << self.grundy_par(q, q.place(Bits::empty(), sq), PAR_LEVELS))
            .reduce(|| 0u64, |a, b| a | b);
        (!seen).trailing_zeros() as u8
    }
}

/// Levels below each root child that also fan out in parallel. The root itself
/// fans over the distinct first moves, so the top `1 + PAR_LEVELS` levels are
/// parallel; deeper nodes run sequentially under the shared table.
const PAR_LEVELS: u32 = 1;

impl Solver for Nimber {
    fn name(&self) -> &'static str {
        "nimber"
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        let g = self.grundy(q, blocked);
        if blocked == Bits::empty() {
            self.root.store(g as u16, Ordering::Relaxed); // capture the root nimber
        }
        g != 0
    }
    fn nodes(&self) -> u64 {
        self.tt.nodes()
    }
    fn cap_bytes(&self) -> u64 {
        self.tt.capacity().1
    }
    /// The Sprague-Grundy value is this solver's reason for being -- report it.
    fn stats(&self) -> String {
        let root = self.root.load(Ordering::Relaxed);
        let nimber = if root == u16::MAX {
            "nimber n/a".to_string()
        } else {
            format!("nimber *{root}")
        };
        format!(
            "{nimber} (full Sprague-Grundy, no cutoff) · {}",
            self.tt.summary()
        )
    }
}

// ================================ NimberSum ================================

/// Per-heap-size `(route, fp)` mixers for the sum-game TT key: the flat table stores
/// `win(avail, h)` under `hash128(canon(avail)) ⊕ HMIX[h]`. `h = 0` is the identity, so the
/// plain-queens subspace keys exactly as a normal solve. splitmix64 finalizer per lane.
const HMIX: [(u64, u64); 16] = {
    const fn mix(mut z: u64) -> u64 {
        z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        z
    }
    let mut out = [(0u64, 0u64); 16];
    let mut h = 1usize;
    while h < 16 {
        out[h] = (
            mix(h as u64 ^ 0x9E37_79B9_7F4A_7C15),
            mix((h as u64) << 32 ^ 0xD1B5_4A32_D192_ED03),
        );
        h += 1;
    }
    out
};

/// Levels below each root child that fan out across rayon workers (the root itself fans
/// over the distinct first moves). Win/loss children short-circuit via `any`, so deeper
/// fan-out is speculative; two levels saturate the box on the boards this engine targets.
const SUM_PAR_LEVELS: u32 = 2;

/// Move-buffer capacity: a node can have up to `MAX_N²` available squares.
const SUM_MAXV: usize = (crate::queens::MAX_N * crate::queens::MAX_N) as usize;

/// Dynamic-order key packing `(child_pc << SQ_BITS) | sq` in a `u32`: 9 square bits cover
/// boards past n=16 (a `u16` with 8 square bits silently corrupts keys once sq ≥ 256).
const SQ_BITS: u32 = 9;
const SQ_MASK: u32 = (1 << SQ_BITS) - 1;
const _: () = assert!(SUM_MAXV <= SQ_MASK as usize + 1);

/// **NimberSum** — the scalable Sprague-Grundy engine behind `queens nimber` (OEIS A344227).
///
/// The full-mex [`Nimber`] must expand the *entire* game DAG (mex admits no cutoff), which is
/// hopeless past n≈13. This engine instead uses the classical sum trick: `G(board) = k` ⟺ the
/// game sum *board + Nim-heap(k)* is a P-position, and win/loss of the sum IS α-β-searchable.
/// The driver solves `win(board, k)` for `k = 0, 1, 2, …` (one TT shared across rounds —
/// `win(avail, h)` is round-independent) until the first LOSS, which is the nimber.
///
/// State is `(avail, h)`: moves are queen placements (h unchanged) or heap reductions
/// (`h' < h`, avail unchanged). Three leaf/probe layers do the heavy lifting:
/// - `h == 0`, `pc ≤ bk` (default 20 — the 3-word code ceiling): the plain-queens boolean [`DenseW8`] leaf
///   (`win ⟺ W(avail)`; `u128` layers to 16, wide 3-word W17..W20 above) — probed first:
///   its child sweeps early-out, where a mex sweep must visit every child.
/// - `h > 0`, `pc ≤ gk` (default 16): the node is `win ⟺ G(avail) ≠ h`, with `G` from the
///   complete [`GrundyW8`] tables + nested mex sweeps `g9..g16` — no expansion, any `h`.
/// - deep: flat lockless TT keyed `hash128(D4-canon) ⊕ HMIX[h]`, heap moves probed first
///   (they re-enter the same `avail` at lower `h` — usually a TT hit or a dense leaf, and a
///   `G(avail)=0` position wins instantly by the h→0 move), then queen moves in dynamic
///   order (child popcount ascending — the production-proven most-forcing-first).
pub struct NimberSum {
    tt: QueensTt,
    dense: DenseW8,
    grundy: GrundyW8,
    /// Grundy-leaf ceiling (`QUEENS_NIMBER_GK`, 9..=16, default 16): a node with
    /// `pc ≤ gk` resolves by mex sweep instead of expansion. Resolved once here.
    /// Only reached at `h > 0` — the `h = 0` subspace takes the cheaper boolean
    /// early-out leaf first (win ⟺ W(avail), no full mex needed).
    gk: usize,
    /// Boolean `h = 0` leaf ceiling (`QUEENS_NIMBER_BK`, 16..=20, default 20): a
    /// `h = 0` node with `pc ≤ bk` resolves from the dense boolean layers — `pc ≤ 16`
    /// via the `u128` [`DenseW8::get_dyn`], 17..20 via the wide 3-word
    /// [`DenseW8::get_dyn_wide`] (the production W17..W20 evaluators). The k=0 round
    /// IS a plain-queens solve, so this is the production `dense_k` lever verbatim.
    bk: usize,
    /// `h = 0` TT-skip band `[lo, hi]` (`QUEENS_NIMBER_SKIP="lo,hi"`, default empty) —
    /// the production skip[18,25] analog: an in-band `h = 0` node skips ALL TT work
    /// (D4 canon + hash + probe + put). With `bk = 20`, pc==21's children are all
    /// boolean leaves (bounded recompute, the skip18 safety argument verbatim); deeper
    /// bands trade bounded re-expansion into the leaf floor for TT pressure relief —
    /// the lever that made the giant n=18 root converge on the 26 GB box.
    skip: (u32, u32),
}

impl NimberSum {
    pub fn new(bits: u32) -> Self {
        let gk = std::env::var("QUEENS_NIMBER_GK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16)
            .clamp(9, 16);
        let bk = std::env::var("QUEENS_NIMBER_BK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(20)
            .clamp(16, 20);
        if bk >= 17 {
            crate::queens::dense::warm_wide(bk);
        }
        // "lo,hi" (e.g. "21,25"); anything unparseable (or unset) = empty band (1, 0).
        let skip = std::env::var("QUEENS_NIMBER_SKIP")
            .ok()
            .and_then(|s| {
                let (lo, hi) = s.split_once(',')?;
                Some((lo.trim().parse().ok()?, hi.trim().parse().ok()?))
            })
            .filter(|&(lo, hi)| lo > bk as u32 && hi >= lo)
            .unwrap_or((1, 0));
        NimberSum {
            tt: QueensTt::new(bits),
            dense: DenseW8::build(),
            grundy: GrundyW8::build(),
            gk,
            bk,
            skip,
        }
    }

    /// The labelled upper-triangular edge code of the `pc ≤ 16`-vertex conflict graph on
    /// `avail` (ascending square order — the same labelling every dense layer uses).
    fn leaf_code(&self, q: &Queens, avail: Bits, pc: usize) -> u128 {
        debug_assert!(pc <= 16);
        let mut verts = [0u32; 16];
        let mut nv = 0usize;
        avail.each(|v| {
            verts[nv] = v;
            nv += 1;
        });
        debug_assert_eq!(nv, pc);
        let mut code = 0u128;
        let mut bit = 0u32;
        for i in 0..pc {
            let row = q.attack[verts[i] as usize];
            for &vj in verts.iter().take(pc).skip(i + 1) {
                code |= (row.get(vj) as u128) << bit;
                bit += 1;
            }
        }
        code
    }

    /// The 3-word labelled edge code for `17 ≤ pc ≤ 20` — the wide twin of
    /// [`leaf_code`](Self::leaf_code) (C(20,2) = 190 bits ≤ 192).
    fn leaf_code_wide(&self, q: &Queens, avail: Bits, pc: usize) -> [u64; 3] {
        debug_assert!((17..=20).contains(&pc));
        let mut verts = [0u32; 20];
        let mut nv = 0usize;
        avail.each(|v| {
            verts[nv] = v;
            nv += 1;
        });
        debug_assert_eq!(nv, pc);
        let mut words = [0u64; 3];
        let mut bit = 0u32;
        for i in 0..pc {
            let row = q.attack[verts[i] as usize];
            for &vj in verts.iter().take(pc).skip(i + 1) {
                words[(bit >> 6) as usize] |= (row.get(vj) as u64) << (bit & 63);
                bit += 1;
            }
        }
        words
    }

    /// `G(avail)` for `pc ≤ gk` — table lookup / nested mex sweep, no expansion.
    #[inline]
    fn grundy_leaf(&self, q: &Queens, avail: Bits, pc: usize) -> u8 {
        self.grundy.grundy_dyn(pc, self.leaf_code(q, avail, pc))
    }

    /// The boolean `h = 0` leaf: `win ⟺ W(avail)` from the dense layers, `pc ≤ bk`.
    #[inline]
    fn boolean_leaf(&self, q: &Queens, avail: Bits, pc: usize) -> bool {
        if pc <= 16 {
            self.dense.get_dyn(pc, self.leaf_code(q, avail, pc))
        } else {
            self.dense
                .get_dyn_wide(pc, &self.leaf_code_wide(q, avail, pc))
        }
    }

    /// Win/loss of the sum node `(avail = board ∖ blocked, heap h)`, sequential.
    /// Leaf order matters: at `h = 0` the boolean early-out leaf is strictly cheaper
    /// than a full mex sweep (win ⟺ W(avail) — the sweep's exact G is wasted work),
    /// so it is probed first; the Grundy leaf serves the `h > 0` layers.
    fn win(&self, q: &Queens, blocked: Bits, h: u8) -> bool {
        let avail = q.board.and_not(blocked);
        let pc = avail.popcount() as usize;
        if h == 0 && pc <= self.bk {
            return self.boolean_leaf(q, avail, pc);
        }
        if pc <= self.gk {
            return self.grundy_leaf(q, avail, pc) != h;
        }
        // The h=0 skip band: no canon, no probe, no put — bounded recompute instead of
        // TT pressure (see the `skip` field). The node still counts (`bump`).
        if h == 0 && (self.skip.0..=self.skip.1).contains(&(pc as u32)) {
            self.tt.bump();
            return self.expand_heap(q, blocked, h) || self.expand_queens(q, blocked, h);
        }
        let (r0, f0) = QueensTt::hash128(q.pos_key(blocked));
        let (r, f) = (r0 ^ HMIX[h as usize].0, f0 ^ HMIX[h as usize].1);
        if let Some(v) = self.tt.get_hashed(r, f) {
            return v != 0;
        }
        self.tt.bump();
        let win = self.expand_heap(q, blocked, h) || self.expand_queens(q, blocked, h);
        self.tt.put_hashed(r, f, win as u8);
        win
    }

    /// The heap arm: any `h' < h` with `(avail, h')` a loss wins. `h' = 0` first — it is a
    /// dense-leaf/TT probe and fires whenever `G(avail) = 0`.
    #[inline]
    fn expand_heap(&self, q: &Queens, blocked: Bits, h: u8) -> bool {
        (0..h).any(|hp| !self.win(q, blocked, hp))
    }

    /// The queens arm: children in dynamic order (child popcount ascending = most-forcing
    /// first — the production ordering win), any losing child wins.
    fn expand_queens(&self, q: &Queens, blocked: Bits, h: u8) -> bool {
        let mut kids = [0u32; SUM_MAXV];
        let nk = self.gather_kids(q, blocked, &mut kids);
        kids[..nk].iter().any(|&kk| {
            let sq = kk & SQ_MASK;
            !self.win(q, q.place(blocked, sq), h)
        })
    }

    /// Collect the available moves as `(child_pc << SQ_BITS) | sq` keys, sorted ascending —
    /// the dynamic move order. Returns the count.
    fn gather_kids(&self, q: &Queens, blocked: Bits, kids: &mut [u32; SUM_MAXV]) -> usize {
        let avail = q.board.and_not(blocked);
        let mut nk = 0usize;
        avail.each(|sq| {
            let cpc = avail.and_not(q.attack[sq as usize]).popcount();
            kids[nk] = (cpc << SQ_BITS) | sq;
            nk += 1;
        });
        kids[..nk].sort_unstable();
        nk
    }

    /// As [`win`](Self::win), but the top `levels` plies fan the queen children across rayon
    /// workers (shared TT). `any` short-circuits once a losing child lands.
    fn win_par(&self, q: &Queens, blocked: Bits, h: u8, levels: u32) -> bool {
        if levels == 0 {
            return self.win(q, blocked, h);
        }
        let avail = q.board.and_not(blocked);
        let pc = avail.popcount() as usize;
        if h == 0 && pc <= self.bk {
            return self.boolean_leaf(q, avail, pc);
        }
        if pc <= self.gk {
            return self.grundy_leaf(q, avail, pc) != h;
        }
        // The near-root parallel plies sit far above any sensible skip band, but keep the
        // semantics identical to `win` in case a wide band reaches them.
        if h == 0 && (self.skip.0..=self.skip.1).contains(&(pc as u32)) {
            self.tt.bump();
            let mut kids = [0u32; SUM_MAXV];
            let nk = self.gather_kids(q, blocked, &mut kids);
            return self.expand_heap(q, blocked, h)
                || kids[..nk].par_iter().any(|&kk| {
                    let sq = kk & SQ_MASK;
                    !self.win_par(q, q.place(blocked, sq), h, levels - 1)
                });
        }
        let (r0, f0) = QueensTt::hash128(q.pos_key(blocked));
        let (r, f) = (r0 ^ HMIX[h as usize].0, f0 ^ HMIX[h as usize].1);
        if let Some(v) = self.tt.get_hashed(r, f) {
            return v != 0;
        }
        self.tt.bump();
        let win = self.expand_heap(q, blocked, h) || {
            let mut kids = [0u32; SUM_MAXV];
            let nk = self.gather_kids(q, blocked, &mut kids);
            kids[..nk].par_iter().any(|&kk| {
                let sq = kk & SQ_MASK;
                !self.win_par(q, q.place(blocked, sq), h, levels - 1)
            })
        };
        self.tt.put_hashed(r, f, win as u8);
        win
    }

    /// One round of the driver: does the first player win *board + Nim-heap(k)*? Root fans
    /// over the D4-distinct first moves in parallel. **Contract: call with ascending `k`** —
    /// the root's heap moves reach `(board, k' < k)`, all already proven first-player wins
    /// by the earlier rounds (else the driver would have stopped there), so they are skipped.
    pub fn round_win(&self, q: &Queens, k: u8) -> bool {
        q.distinct_first_moves()
            .par_iter()
            .any(|&sq| !self.win_par(q, q.place(Bits::empty(), sq), k, SUM_PAR_LEVELS))
    }

    /// The Sprague-Grundy value of the empty board: the first `k ≤ max_k` whose heap-sum
    /// round is a LOSS. `None` if every round through `max_k` is a win (G > max_k).
    pub fn nimber(&self, q: &Queens, max_k: u8) -> Option<u8> {
        (0..=max_k.min(15)).find(|&k| !self.round_win(q, k))
    }

    pub fn nodes(&self) -> u64 {
        self.tt.drain_all(); // fold the workers' thread-local tallies in first
        self.tt.nodes()
    }

    pub fn table_summary(&self) -> String {
        let skip = if self.skip.0 <= self.skip.1 {
            format!(" · h=0 TT-skip pc {}..={}", self.skip.0, self.skip.1)
        } else {
            String::new()
        };
        format!(
            "{} · G≤8 tables {} MB · Grundy-leaf pc≤{} · boolean h=0 leaf pc≤{}{skip}",
            self.tt.summary(),
            self.grundy.bytes() / (1 << 20),
            self.gk,
            self.bk,
        )
    }
}
