//! `Incremental` -- the A3 inner-loop kernel (handoff
//! `2026-06-16-queens-inner-loop-rewrite.md`, Step 3). Instead of recomputing the
//! D4 canonical key per node by scattering set bits through a permutation
//! ([`Queens::canon`], the ~250x "fat"), it **carries the 8 dihedral orientations
//! of `available` live down the DFS stack**. Placing a queen on `sq` updates each
//! orientation by one `and-not` against that move's attack mask in that
//! orientation's frame (`att[sq][t] = perm_t(attack[sq])`); the canonical key is
//! the lexicographic min of the 8 orientations (`lex_min8`), byte-identical to
//! `pos_key`. Measured at ~62 cyc/canon vs the scatter's ~574 (`canon_bench` A3).
//!
//! It is otherwise the production [`Parallel`] solver: parity-aware rayon root
//! parallelism (even/prove-a-loss plies fan all children with no speculation; odd/
//! prove-a-win plies stay sequential so the cutoff survives) plus the odd-board
//! O(1) theorem. D4-only -- the graph-isomorphism keys are a freeze-time concern,
//! not this hot path. Because the key per node equals `pos_key` exactly, the node
//! count, distinct working set, re-expansion, and verdict are identical to
//! `Parallel`; only the per-node canonicalisation cost changes.

use super::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

/// All 8 dihedral orientations of `available`: `orient[t] = perm_t(available)`.
/// The one-time recompute at a search *entry* (never per node -- the recursion
/// carries these incrementally). Uses the same scatter as [`Queens::canon`].
#[inline]
pub(crate) fn orient_of(q: &Queens, available: Bits) -> [Bits; 8] {
    std::array::from_fn(|t| {
        let perm = &q.sym[t];
        let mut img = Bits::ZERO;
        available.each(|s| img.set(perm[s as usize]));
        img
    })
}

/// The canonical key of a position from its 8 orientations: the lexicographically
/// smallest image. Equals `Queens::canon(available)` / `pos_key` exactly (same 8
/// images, same `Bits` order), so the TT merges identically. Serial early-out fold
/// -- the A3-validated form (beats branchless `lex_lt` and tree reductions; most
/// image pairs differ in word 0, so the compare exits after one limb).
// NOTE (2026-06-18, session --6): an AVX-512 gather + reduce-min-cascade lex_min8 was
// built and A/B'd on n=16 — MEASURED LOSS (CPI 1.110→1.275, M/s 41.0→37.4, −9%) despite
// −12% instructions: the strided gathers are CPI-expensive on znver5, and even a
// gather-free SoA variant can't beat the scalar because the scalar early-exits on word 0
// (most orientations differ there) while any branchless all-4-limb reduction processes
// all limbs unconditionally. Kept scalar. Don't re-attempt the cascade form.
// RE-CONFIRMED (2026-06-21, session --19): a branch-mispredict audit found this `cand < best`
// the #1 mispredict source post-W17 (~34% of `wins_inc`'s misses; ~coin-flip). A *scalar* branchless
// blend — `lt` mask from two `u128`-half compares (`(chi<bhi)|((chi==bhi)&(clo<blo))`), then
// `best = blend(best,cand,lt)`, no gather — was built + A/B'd at the **default 8 GB TT**: **+2.0%
// cyc/node (5-round, every B round above every A)**. So the early-out wins even with the cheapest
// branchless form and even with the mispredict isolated: the all-4-limb ALU exceeds the saved
// mispredict. The de-branch lever for `lex_min8` is DEAD; the residual `wins_inc` mispredicts are the
// irreducible α-β cutoffs (`if lost` / empty-child) which can't be de-branched byte-identically.
#[inline]
pub(crate) fn lex_min8(o: &[Bits; 8]) -> Bits {
    let mut best = o[0];
    for &cand in &o[1..] {
        if cand < best {
            best = cand;
        }
    }
    best
}

/// The per-square, per-orientation attack table: `att[sq][t] = perm_t(attack[sq])`,
/// the move's attack mask in orientation `t`'s frame (8 * n^2 `Bits`; 64 KB at
/// n=16, L1/L2-resident). Built once per solve from the board geometry.
pub(crate) fn build_att(q: &Queens) -> Box<[[Bits; 8]]> {
    let nn = (q.n * q.n) as usize;
    (0..nn)
        .map(|sq| {
            std::array::from_fn(|t| {
                let perm = &q.sym[t];
                let mut img = Bits::ZERO;
                q.attack[sq].each(|s| img.set(perm[s as usize]));
                img
            })
        })
        .collect()
}

/// The child's 8 orientations after placing the move whose per-orientation attack
/// masks are `a`: `child[t] = parent[t] & !a[t] = perm_t(available & !attack[sq])`
/// (perm distributes over `&`/`!`, so the incremental update is exact). `child0`
/// (the identity image, already computed for the terminal test) is reused. Shared
/// with the `burr` solver, which reuses the exact A3 key path.
#[inline]
pub(crate) fn child_orient(parent: &[Bits; 8], a: &[Bits; 8], child0: Bits) -> [Bits; 8] {
    [
        child0,
        parent[1].and_not(a[1]),
        parent[2].and_not(a[2]),
        parent[3].and_not(a[3]),
        parent[4].and_not(a[4]),
        parent[5].and_not(a[5]),
        parent[6].and_not(a[6]),
        parent[7].and_not(a[7]),
    ]
}

/// **Incremental** -- the A3 DFS-resident solver. See the module docs.
pub struct Incremental {
    tt: QueensTt,
    /// `att[sq][t] = perm_t(attack[sq])`, built lazily on the first solve (the board
    /// side is fixed per solve, so the table is built once and threaded through the
    /// recursion -- never rebuilt or looked up via a per-node `OnceLock::get`).
    att: OnceLock<Box<[[Bits; 8]]>>,
    par_depth: u32,
    par_min_avail: Option<u32>,
    eff_min_avail: AtomicU32,
    root_done: AtomicU64,
    root_total: AtomicU64,
}

impl Incremental {
    pub fn new(bits: u32) -> Self {
        Incremental {
            tt: QueensTt::new(bits),
            att: OnceLock::new(),
            par_depth: par_depth(),
            par_min_avail: par_min_avail_override(),
            eff_min_avail: AtomicU32::new(u32::MAX),
            root_done: AtomicU64::new(0),
            root_total: AtomicU64::new(0),
        }
    }

    /// As [`Incremental::new`], but counting the distinct positions visited (see
    /// [`Tt::new_counting`]). The HyperLogLog folds every key the TT is queried for,
    /// and the incremental key equals `pos_key`, so `--distinct` matches `parallel`.
    pub fn new_counting(bits: u32, hll_p: u32) -> Self {
        Incremental {
            tt: QueensTt::new_counting(bits, hll_p, false),
            att: OnceLock::new(),
            par_depth: par_depth(),
            par_min_avail: par_min_avail_override(),
            eff_min_avail: AtomicU32::new(u32::MAX),
            root_done: AtomicU64::new(0),
            root_total: AtomicU64::new(0),
        }
    }

    /// Wrap a reloaded checkpoint table for a warm resume (see [`Tt::from_tt`]). The
    /// dump must have been produced under the D4 key (the incremental key is D4).
    pub fn from_tt(tt: QueensTt) -> Self {
        Incremental {
            tt,
            att: OnceLock::new(),
            par_depth: par_depth(),
            par_min_avail: par_min_avail_override(),
            eff_min_avail: AtomicU32::new(u32::MAX),
            root_done: AtomicU64::new(0),
            root_total: AtomicU64::new(0),
        }
    }

    /// The per-square attack table, built once per solve and threaded through the
    /// recursion. The board side is fixed per solve, so the `get_or_init` resolves
    /// to a build exactly once; callers hold the `&[[Bits; 8]]` for the whole search.
    #[inline]
    fn att(&self, q: &Queens) -> &[[Bits; 8]] {
        self.att.get_or_init(|| build_att(q))
    }

    /// Sequential cutoff search with the node's 8 orientations and canonical key in
    /// hand. Mirrors [`Tt::wins_keyed`] exactly, but keys each child by the
    /// incremental `lex_min8` (8 `and-not`s) instead of recomputing `pos_key`.
    fn wins_inc(&self, q: &Queens, att: &[[Bits; 8]], orient: &[Bits; 8], key: Bits) -> bool {
        if let Some(w) = self.tt.get(key) {
            return w != 0;
        }
        self.tt.bump();
        let avail = orient[0]; // identity image == the available mask
        let mut result = false;
        for &sq in &q.order {
            if !avail.get(sq) {
                continue;
            }
            let a = &att[sq as usize];
            // Child available (identity frame): a terminal child empties the board,
            // so the opponent cannot move and we win at once -- skip its 7 other
            // and-nots and the recursive probe (the `Tt::wins_keyed` fast path).
            let child0 = avail.and_not(a[0]);
            if child0 == Bits::ZERO {
                result = true;
                break;
            }
            let child = child_orient(orient, a, child0);
            let ckey = lex_min8(&child);
            self.tt.prefetch(ckey);
            if !self.wins_inc(q, att, &child, ckey) {
                result = true;
                break;
            }
        }
        self.tt.put(key, result as u8);
        result
    }

    /// Recursive parallel cutoff search (the incremental twin of [`Parallel::par_wins`]):
    /// for the top [`par_depth`] plies (and while a deep prove-a-loss node is still
    /// large) it fans children across rayon, else drops to [`Self::wins_inc`]. Parity
    /// is the trick -- even/prove-a-loss plies fan **all** children (no cutoff to lose,
    /// zero speculation); odd/prove-a-win plies stay sequential so the cutoff survives.
    fn par_wins_inc(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        orient: &[Bits; 8],
        key: Bits,
        depth: u32,
        min_avail: u32,
    ) -> bool {
        if let Some(w) = self.tt.get(key) {
            return w != 0;
        }
        let avail = orient[0];
        if depth >= self.par_depth && avail.popcount() <= min_avail {
            return self.wins_inc(q, att, orient, key);
        }
        self.tt.bump();
        // Gather non-terminal child moves (square indices); a terminal child wins now.
        let mut moves: [u32; MAXV] = [0; MAXV];
        let mut nc = 0usize;
        for &sq in &q.order {
            if !avail.get(sq) {
                continue;
            }
            if avail.and_not(att[sq as usize][0]) == Bits::ZERO {
                self.tt.put(key, 1);
                return true;
            }
            moves[nc] = sq;
            nc += 1;
        }
        let kids = &moves[..nc];
        // Each child recomputes its 8 orientations from the parent's (held by ref,
        // register/L1-resident) -- the 8-and-not incremental update, on its own task.
        let recurse = |&sq: &u32| {
            let a = &att[sq as usize];
            let child0 = avail.and_not(a[0]);
            let child = child_orient(orient, a, child0);
            let ckey = lex_min8(&child);
            !self.par_wins_inc(q, att, &child, ckey, depth + 1, min_avail)
        };
        let won = if depth.is_multiple_of(2) {
            // Even / prove-a-loss: fan all children (no cutoff to lose, no elder-first
            // lead). `any` still short-circuits a mis-parity winning child.
            kids.par_iter().any(recurse)
        } else {
            // Odd / prove-a-win: keep the α-β cutoff -- sequential, recursing into the
            // parallel even children below.
            kids.iter().any(recurse)
        };
        self.tt.put(key, won as u8);
        won
    }
}

impl Solver for Incremental {
    fn name(&self) -> &'static str {
        "incremental"
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        // Compute the 8 orientations of this position's available mask once, then
        // recurse incrementally (the lineage cross-check + any non-root query).
        let att = self.att(q);
        let orient = orient_of(q, q.board.and_not(blocked));
        let key = lex_min8(&orient);
        self.wins_inc(q, att, &orient, key)
    }
    /// Odd boards are the centre + 180°-mirror theorem (as [`Parallel`]); even boards
    /// search the symmetry-distinct first moves with the parity-aware parallel solver.
    fn first_player_wins(&self, q: &Queens) -> bool {
        if q.is_odd() {
            return true;
        }
        let att = self.att(q);
        let min_avail = min_avail_for(self.par_min_avail, q.n);
        self.eff_min_avail.store(min_avail, Ordering::Relaxed);
        let moves = q.distinct_first_moves();
        self.root_total.store(moves.len() as u64, Ordering::Relaxed);
        self.root_done.store(0, Ordering::Relaxed);
        // Root available = the full board; its 8 orientations seed the recursion.
        let root = orient_of(q, q.board);
        // Pre-resolve roots already decided in a warm (resumed) TT: a hit on a first
        // move's child key means that move's value is already known. Count those toward
        // the progress (so a resume shows the snapshot's roots at once, not 0/N, and we
        // skip re-searching solved roots), and a decided root where the second player
        // *loses* means the first player wins outright. A cold run hits none, so the
        // pending list is every move in order and the behaviour is unchanged.
        let mut pending: Vec<([Bits; 8], Bits)> = Vec::with_capacity(moves.len());
        for &sq in &moves {
            let a = &att[sq as usize];
            let co = child_orient(&root, a, q.board.and_not(a[0]));
            let ckey = lex_min8(&co);
            match self.tt.get(ckey) {
                Some(0) => {
                    self.root_done.fetch_add(1, Ordering::Relaxed);
                    return true; // second player loses from this move ⇒ first player wins
                }
                Some(_) => {
                    self.root_done.fetch_add(1, Ordering::Relaxed); // refuted; count it done
                }
                None => pending.push((co, ckey)),
            }
        }
        if pending.is_empty() {
            return false; // every first move refuted ⇒ second player wins
        }
        let resolve = |co: &[Bits; 8], ckey: Bits| {
            let wins = !self.par_wins_inc(q, att, co, ckey, 1, min_avail);
            self.root_done.fetch_add(1, Ordering::Relaxed);
            wins
        };
        let (first, rest) = pending.split_first().unwrap();
        // Elder brother in parallel (not single-core), then fan the siblings.
        if resolve(&first.0, first.1) {
            return true; // best move already wins -- no speculation
        }
        rest.par_iter().any(|(co, ckey)| resolve(co, *ckey))
    }
    fn drain(&self) {
        self.tt.drain_all();
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
    fn working_set(&self) -> Option<Vec<(Bits, u8)>> {
        self.tt.working_set()
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
        format!(
            "{} rayon workers, {done}/{total} root moves, par-depth {}/min-avail {ma} · incremental canon · {}",
            rayon::current_num_threads(),
            self.par_depth,
            self.tt.summary(),
        )
    }
    fn tt(&self) -> Option<&QueensTt> {
        Some(&self.tt)
    }
}
