//! `Parallel` -- the production solver: parity-aware rayon root parallelism
//! (Young-Brothers-Wait) plus the odd-board O(1) theorem.

use super::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// **Parallel** -- the production solver. Sequential search is [`Tt`] with
/// canonical keys; `first_player_wins` adds the odd-board O(1) theorem and rayon
/// root parallelism with a Young-Brothers-Wait guard.
pub struct Parallel {
    inner: Tt,
    /// Plies from the root searched in parallel (see [`par_depth`]); below this a
    /// node may *still* split if it is large (see `par_min_avail`), else it drops to
    /// the sequential cutoff search.
    par_depth: u32,
    /// `QUEENS_PAR_MIN_AVAIL` override (`None` = auto by board size); the size-based
    /// split that divides the deep stragglers so idle cores can steal them (#20). See
    /// [`min_avail_for`].
    par_min_avail: Option<u32>,
    /// The effective size-split threshold for the current solve (set at
    /// `first_player_wins` from the board size), captured for the stats line.
    eff_min_avail: AtomicU32,
    /// Root moves resolved / to resolve (for a progress indicator). A
    /// second-player win must refute *every* distinct first move, so `done`
    /// climbs to `total`; a first-player win short-circuits earlier.
    root_done: AtomicU64,
    root_total: AtomicU64,
}

impl Parallel {
    pub fn new(bits: u32) -> Self {
        Parallel {
            inner: Tt::new(bits, true),
            par_depth: par_depth(),
            par_min_avail: par_min_avail_override(),
            eff_min_avail: AtomicU32::new(u32::MAX),
            root_done: AtomicU64::new(0),
            root_total: AtomicU64::new(0),
        }
    }

    /// As [`Parallel::new`], but counting the distinct positions visited (see
    /// [`Tt::new_counting`]). The HyperLogLog is lock-free, so it works under the
    /// root parallelism; the `exact` hash set is not (it would serialise every
    /// worker), so counting `exact` requires the sequential [`Tt`] solver instead.
    pub fn new_counting(bits: u32, hll_p: u32) -> Self {
        Parallel {
            inner: Tt::new_counting(bits, true, hll_p, false),
            par_depth: par_depth(),
            par_min_avail: par_min_avail_override(),
            eff_min_avail: AtomicU32::new(u32::MAX),
            root_done: AtomicU64::new(0),
            root_total: AtomicU64::new(0),
        }
    }

    /// Wrap a reloaded checkpoint table for a warm resume (see [`Tt::from_tt`]).
    /// Re-running `first_player_wins` fast-forwards already-solved root subtrees
    /// (instant TT hits) and continues the unsolved ones -- the TT *is* the progress.
    pub fn from_tt(tt: QueensTt) -> Self {
        Parallel {
            inner: Tt::from_tt(tt, true),
            par_depth: par_depth(),
            par_min_avail: par_min_avail_override(),
            eff_min_avail: AtomicU32::new(u32::MAX),
            root_done: AtomicU64::new(0),
            root_total: AtomicU64::new(0),
        }
    }

    /// Recursive parallel cutoff search of `blocked` (canonical key in hand). For the
    /// top [`par_depth`](Self::par_depth) plies it fans children across rayon; below
    /// that it drops to the sequential [`Tt::wins_keyed`]. Parity is the trick: for a
    /// second-player win the tree alternates "prove-a-loss" nodes (every child must be
    /// searched -- no α-β cutoff to lose) with "prove-a-win" nodes (one winner suffices
    /// -- cutoff). The root (refute every first move) is prove-a-loss, so the EVEN plies
    /// below it are too: fan those for free; keep the ODD (prove-a-win) plies sequential
    /// so their cutoff survives (elder child first, which well-ordered usually cuts at
    /// once). This keeps the dominant root-0 subtree off a single core at n=16 while
    /// confining speculation to mis-ordered OR nodes.
    fn par_wins(&self, q: &Queens, blocked: Bits, key: Bits, depth: u32, min_avail: u32) -> bool {
        if let Some(w) = self.inner.tt.get(key) {
            return w != 0;
        }
        // Drop to the sequential cutoff search once we are both below the `par_depth`
        // floor *and* the subtree is small (available count ≤ `min_avail`). Big deep
        // nodes keep splitting so an idle core can steal a straggler -- the #20 tail
        // fix -- with rayon paying the split cost only on an actual steal.
        if depth >= self.par_depth && q.board.and_not(blocked).popcount() <= min_avail {
            return self.inner.wins_keyed(q, blocked, key);
        }
        self.inner.tt.bump();
        // Gather children; a terminal child means the opponent cannot move, so this
        // node wins at once (the [`Tt::wins_keyed`] terminal fast path).
        let mut children: [Bits; MAXV] = [Bits::ZERO; MAXV];
        let mut nc = 0usize;
        for &sq in &q.order {
            if !q.is_available(blocked, sq) {
                continue;
            }
            let child = q.place(blocked, sq);
            if q.no_moves(child) {
                self.inner.tt.put(key, 1);
                return true;
            }
            children[nc] = child;
            nc += 1;
        }
        let kids = &children[..nc];
        let won = if depth.is_multiple_of(2) {
            // Even / prove-a-loss: no α-β cutoff to lose, so fan *all* children at once
            // (no elder-first lead -- that would grind one huge child single-core, the
            // n=16 failure mode). `any` still short-circuits if some child unexpectedly
            // wins (a mis-parity node), but for a true prove-a-loss node all are searched.
            kids.par_iter().any(|&child| {
                let ckey = self.inner.node_key(q, child);
                !self.par_wins(q, child, ckey, depth + 1, min_avail)
            })
        } else {
            // Odd / prove-a-win: keep the α-β cutoff -- sequential, recursing into the
            // parallel even children below.
            let mut w = false;
            for &child in kids {
                let ckey = self.inner.node_key(q, child);
                if !self.par_wins(q, child, ckey, depth + 1, min_avail) {
                    w = true;
                    break;
                }
            }
            w
        };
        self.inner.tt.put(key, won as u8);
        won
    }
}

impl Solver for Parallel {
    fn name(&self) -> &'static str {
        "parallel"
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        self.inner.wins(q, blocked)
    }
    /// Odd boards are a theorem, not a search: the first player takes the centre,
    /// then mirrors every reply by 180° rotation. The centre attacks all four
    /// lines through it, so a legal reply `s` is off those lines -- exactly the
    /// condition for `s` not to attack its mirror `s'` -- and by symmetry `s'` is
    /// free, so the first player always has the pairing response and the second
    /// player runs out first. (Impartial normal-play ⇒ Sprague-Grundy N-position.)
    ///
    /// Even boards search. A Young-Brothers-Wait guard searches the best-ordered
    /// first move sequentially -- returning at once if it wins (no speculation) --
    /// and only then fans the siblings across rayon workers sharing the table,
    /// where `any` short-circuits on the first winning one. A naïve
    /// `par_iter().any()` over *all* first moves regresses first-player wins
    /// badly (~40× on 13×13) by speculatively searching whole losing subtrees the
    /// cutoff would skip; the guard keeps the cutoff while still parallelising the
    /// must-refute-everything case of a second-player win.
    fn first_player_wins(&self, q: &Queens) -> bool {
        if q.is_odd() {
            return true; // centre + 180° mirror strategy
        }
        // Resolve the size-split threshold once for this solve (auto by board size,
        // env-overridable) -- never per node -- and thread it through the recursion.
        let min_avail = min_avail_for(self.par_min_avail, q.n);
        self.eff_min_avail.store(min_avail, Ordering::Relaxed);
        let moves = q.distinct_first_moves();
        self.root_total.store(moves.len() as u64, Ordering::Relaxed);
        self.root_done.store(0, Ordering::Relaxed);
        match moves.split_first() {
            None => false,
            Some((&first, rest)) => {
                // Elder brother (root move 0): search its subtree *in parallel* (not on a
                // single core), so the dominant first move uses all workers from the start
                // -- the n=16 fix -- while still warming the shared TT before the younger
                // brothers fan out.
                let fc = q.place(Bits::ZERO, first);
                let wins = !self.par_wins(q, fc, self.inner.node_key(q, fc), 1, min_avail);
                self.root_done.fetch_add(1, Ordering::Relaxed);
                if wins {
                    return true; // best move already wins -- no speculation
                }
                rest.par_iter().any(|&sq| {
                    let c = q.place(Bits::ZERO, sq);
                    let wins = !self.par_wins(q, c, self.inner.node_key(q, c), 1, min_avail);
                    self.root_done.fetch_add(1, Ordering::Relaxed);
                    wins
                })
            }
        }
    }
    fn drain(&self) {
        self.inner.tt.drain_all();
    }
    fn nodes(&self) -> u64 {
        self.inner.nodes()
    }
    fn cap_bytes(&self) -> u64 {
        self.inner.cap_bytes()
    }
    fn report(&self) -> Option<CountReport> {
        self.inner.report()
    }
    fn working_set(&self) -> Option<Vec<(Bits, u8)>> {
        self.inner.working_set()
    }
    fn root_progress(&self) -> Option<(u64, u64)> {
        let total = self.root_total.load(Ordering::Relaxed);
        (total > 0).then(|| (self.root_done.load(Ordering::Relaxed).min(total), total))
    }
    /// Root parallelism is this solver's whole point, so report the worker count
    /// and root-move fan-out alongside the shared table's fill.
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
            "{} rayon workers, {done}/{total} root moves, par-depth {}/min-avail {ma} · {}",
            rayon::current_num_threads(),
            self.par_depth,
            self.inner.stats(),
        )
    }
    fn tt(&self) -> Option<&QueensTt> {
        Some(&self.inner.tt)
    }
}
