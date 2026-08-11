//! `Tt` -- the cutoff search backed by the transposition table (memo /
//! symmetry), plus the `count --branching` tally.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// **Memo** (`canon=false`) / **Symmetry** (`canon=true`) -- the cutoff search
/// backed by a fixed-size transposition table. With `canon` the key is the
/// position's dihedral-canonical image, so all 8 symmetric states share an entry.
/// Per-node branching/cutoff tally for the `count --branching` measurement. Lives on
/// [`Tt`] but is only ever touched on the `wins_keyed_in::<true>` monomorphisation
/// (selected once at the root by `branching`); production (`::<false>`) never emits a
/// reference to it, so it is zero-cost. Single-threaded (the branching measurement uses
/// the sequential solver), so the atomics never contend -- they are atomics only because
/// [`Solver`] is `Sync`.
#[derive(Default)]
struct Tally {
    /// Total `node_key` (canon) calls = expanded edges. `edges / distinct` = b̄, the
    /// per-distinct-node canonicalisation multiplier the theoretical floor turns on.
    edges: AtomicU64,
    /// Nodes that found a winning move (returned `true` after expansion).
    win_nodes: AtomicU64,
    /// Nodes that refuted every move (returned `false`): the prove-a-loss nodes, which
    /// have no cutoff to lose and are *required* by the proof DAG.
    loss_nodes: AtomicU64,
    /// Σ over win nodes of the available moves tried before the cutoff fired (1 = the
    /// first available move won). Mean cutoff = `win_tried_sum / win_nodes`.
    win_tried_sum: AtomicU64,
    /// Histogram of the cutoff position at win nodes: index k = cut on the (k+1)-th
    /// available move; index 7 = 8th-or-later. The move-ordering-quality shape.
    win_cut: [AtomicU64; 8],
}

/// A snapshot of [`Tally`] for reporting (see [`Solver::branching_stats`]).
pub struct BranchingStats {
    pub edges: u64,
    pub win_nodes: u64,
    pub loss_nodes: u64,
    pub win_tried_sum: u64,
    pub win_cut: [u64; 8],
}

pub struct Tt {
    pub(crate) tt: QueensTt,
    pub(crate) canon: bool,
    pub(crate) key: KeyMode,
    pub(crate) max_avail: u32,
    /// Selects the counting monomorphisation at the root (`count --branching`); off in
    /// production. Resolved once at construction, never read per node.
    branching: bool,
    tally: Tally,
}

impl Tt {
    pub fn new(bits: u32, canon: bool) -> Self {
        Tt {
            tt: QueensTt::new(bits),
            canon,
            key: key_mode(),
            max_avail: key_max_avail(),
            branching: false,
            tally: Tally::default(),
        }
    }

    /// Wrap an already-built table (e.g. a reloaded checkpoint image) so a search
    /// resumes warm. The key mode / selective-keying threshold are resolved from the
    /// environment as in [`Tt::new`]; the caller must use the *same* `QUEENS_KEY` the
    /// dump was produced under, or stored keys won't match.
    pub fn from_tt(tt: QueensTt, canon: bool) -> Self {
        Tt {
            tt,
            canon,
            key: key_mode(),
            max_avail: key_max_avail(),
            branching: false,
            tally: Tally::default(),
        }
    }

    /// As [`Tt::new`], but the table also folds every position it is queried for
    /// into a HyperLogLog (and, with `exact`, a hash set) so the search reports
    /// the number of *distinct* positions it visited -- its true working set.
    pub fn new_counting(bits: u32, canon: bool, hll_p: u32, exact: bool) -> Self {
        Tt {
            tt: QueensTt::new_counting(bits, hll_p, exact),
            canon,
            key: key_mode(),
            max_avail: key_max_avail(),
            branching: false,
            tally: Tally::default(),
        }
    }

    /// Enable the `count --branching` tally: the next `wins`/`first_player_wins` selects
    /// the counting monomorphisation (`wins_keyed_in::<true>`) at the root. Build-time
    /// only -- resolved once here, never per node. Use the **sequential** solver (this
    /// is on [`Tt`]); the measurement is single-threaded by construction.
    pub fn with_branching(mut self) -> Self {
        self.branching = true;
        self
    }

    /// The transposition key for the position with this `blocked` mask, per the
    /// configured [`KeyMode`]. The graph keys canonicalise the *available-graph* up
    /// to isomorphism (merging far more than the 8 board symmetries).
    #[inline]
    pub(crate) fn node_key(&self, q: &Queens, blocked: Bits) -> Bits {
        if !self.canon {
            return blocked; // memo: raw mask
        }
        if self.key == KeyMode::D4 {
            return q.pos_key(blocked);
        }
        // Selective keying: only graph-key positions whose available-graph is small
        // enough to be cheap (and where the iso-merge is densest); larger graphs keep
        // the cheap D4 key. Strictly intra-ply transpositions ⇒ no merges are lost.
        let available = q.board.and_not(blocked);
        if available.popcount() > self.max_avail {
            return d4_bits(q.pos_key(blocked));
        }
        match self.key {
            KeyMode::GraphIr => graph_bits(q.iso_key_ir(available)),
            KeyMode::GraphCanon => graph_bits(q.iso_key_canon(available)),
            KeyMode::GraphComp => graph_bits(q.iso_key_components(available)),
            KeyMode::GraphFast => graph_bits(q.iso_key_fast(available)),
            KeyMode::D4 => unreachable!(),
        }
    }

    /// The cutoff search with `blocked`'s canonical key already in hand. The caller
    /// prefetched the matching slot before recursing, so this entry `get` -- the
    /// first thing every node does -- is typically warm (Session 5, L1 cluster).
    pub(crate) fn wins_keyed(&self, q: &Queens, blocked: Bits, key: Bits) -> bool {
        self.wins_keyed_in::<false>(q, blocked, key)
    }

    /// The cutoff search, monomorphised on `COUNT`. Production is `::<false>` -- the
    /// `COUNT` blocks are compile-time eliminated, so the [`Tally`] is never referenced
    /// and the hot path is byte-identical to before. `::<true>` (selected once at the
    /// root by `branching`) tallies b̄ (canons per node) and the win-node cutoff
    /// distribution for `count --branching`. The const threads down the recursion so the
    /// single runtime decision happens once, at the top -- per the hot-path-toggle rule.
    fn wins_keyed_in<const COUNT: bool>(&self, q: &Queens, blocked: Bits, key: Bits) -> bool {
        if let Some(w) = self.tt.get(key) {
            return w != 0;
        }
        self.tt.bump();
        let mut result = false;
        let mut tried = 0u32;
        for &sq in &q.order {
            if !q.is_available(blocked, sq) {
                continue;
            }
            tried += 1;
            let child = q.place(blocked, sq);
            // Terminal-child fast path: the opponent then cannot move, so we win at
            // once -- skip the recursive probe. Every terminal canonicalises to the
            // same `ZERO` key (`pos_key` folds `available`; empty ⇒ `Bits::ZERO`), so
            // the elided probes would all hammer one hot atomic slot. (Raw-key `memo`
            // keys each terminal by its own `blocked`, so for it `--distinct` drops
            // every terminal, not one key.) A terminal *is* a winning move, so it counts
            // toward the cutoff position (`tried`) but pays no `node_key` (no canon).
            if q.no_moves(child) {
                result = true;
                break;
            }
            let ckey = self.node_key(q, child);
            if COUNT {
                self.tally.edges.fetch_add(1, Ordering::Relaxed);
            }
            // Prefetch the child's slot now; its recursion will probe it first thing.
            self.tt.prefetch(ckey);
            if !self.wins_keyed_in::<COUNT>(q, child, ckey) {
                result = true;
                break;
            }
        }
        if COUNT {
            if result {
                self.tally.win_nodes.fetch_add(1, Ordering::Relaxed);
                self.tally
                    .win_tried_sum
                    .fetch_add(tried as u64, Ordering::Relaxed);
                self.tally.win_cut[(tried as usize - 1).min(7)].fetch_add(1, Ordering::Relaxed);
            } else {
                self.tally.loss_nodes.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.tt.put(key, result as u8);
        result
    }
}

impl Solver for Tt {
    fn name(&self) -> &'static str {
        if self.canon {
            "symmetry"
        } else {
            "memo"
        }
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        let key = self.node_key(q, blocked);
        // The single runtime decision: select the counting or production monomorphisation
        // once, at the root; the const threads down the recursion (no per-node branch).
        if self.branching {
            self.wins_keyed_in::<true>(q, blocked, key)
        } else {
            self.wins_keyed_in::<false>(q, blocked, key)
        }
    }
    fn drain(&self) {
        self.tt.drain_all();
    }
    fn nodes(&self) -> u64 {
        self.tt.nodes()
    }
    fn branching_stats(&self) -> Option<BranchingStats> {
        self.branching.then(|| BranchingStats {
            edges: self.tally.edges.load(Ordering::Relaxed),
            win_nodes: self.tally.win_nodes.load(Ordering::Relaxed),
            loss_nodes: self.tally.loss_nodes.load(Ordering::Relaxed),
            win_tried_sum: self.tally.win_tried_sum.load(Ordering::Relaxed),
            win_cut: std::array::from_fn(|i| self.tally.win_cut[i].load(Ordering::Relaxed)),
        })
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
    fn stats(&self) -> String {
        self.tt.summary()
    }
    fn tt(&self) -> Option<&QueensTt> {
        Some(&self.tt)
    }
}
