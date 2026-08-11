//! `Fused` -- the 20-minute lever: [`IsoBurr`](super::IsoBurr)'s wins (the
//! graph-iso merge + the eviction-free BuRR store) fused with
//! [`Incremental`](super::Incremental)'s nodes/sec, by **keying each node with
//! exactly one key** instead of two.
//!
//! `iso-burr` probes the cheap D4 key *and then* the iso key on every miss, and
//! stores both -- so the dominant deep/small node population (≈74% of nodes are
//! first-visit misses) pays **two** latency-bound store probes where
//! `incremental`'s flat table pays one, and the duplicated D4 entries bloat the
//! store (more freezes → a longer segment cascade, and the byte cap is reached
//! sooner → higher re-expansion). The D4 pre-probe never adds a *merge*: it only
//! saves recomputing the (cheap, L1-resident) tiny-table iso key on an exact-D4
//! revisit, at the cost of a DRAM probe on every first visit.
//!
//! `Fused` drops it. Each node computes **one** key -- the tiny-table graph-iso
//! key when the available graph is small (`avail.popcount() <= iso_max`, the
//! transposition-rich deep nodes), else the incremental D4 `lex_min8` -- in a
//! tagged namespace, threaded through the recursion exactly like
//! [`Burr`](super::Burr). The merge is identical to `iso-burr` (the same iso
//! classes), but the store sees ~one key per node: fewer probes, fewer segments,
//! lower re-expansion. Sound because the key choice is a pure function of the
//! position (its available popcount) and transpositions are strictly intra-ply.

use super::incremental::{build_att, child_orient, lex_min8, orient_of};
use super::*;
use crate::queens::BurrStore;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

/// **Fused** -- the A3 DFS-resident kernel + BuRR store + single selective iso/D4 key.
pub struct Fused {
    store: BurrStore,
    att: OnceLock<Box<[[Bits; 8]]>>,
    par_depth: u32,
    par_min_avail: Option<u32>,
    iso_max_avail: u32,
    eff_min_avail: AtomicU32,
    root_done: AtomicU64,
    root_total: AtomicU64,
}

impl Fused {
    pub fn new(bits: u32) -> Self {
        Self::from_store(BurrStore::new(bits))
    }

    /// As [`Fused::new`], but counting the distinct (tagged iso/D4) keys visited.
    pub fn new_counting(bits: u32, hll_p: u32) -> Self {
        Self::from_store(BurrStore::new_counting(bits, hll_p))
    }

    fn from_store(store: BurrStore) -> Self {
        Fused {
            store,
            att: OnceLock::new(),
            par_depth: par_depth(),
            par_min_avail: par_min_avail_override(),
            iso_max_avail: iso_burr_key_max_avail(),
            eff_min_avail: AtomicU32::new(u32::MAX),
            root_done: AtomicU64::new(0),
            root_total: AtomicU64::new(0),
        }
    }

    #[inline]
    fn att(&self, q: &Queens) -> &[[Bits; 8]] {
        self.att.get_or_init(|| build_att(q))
    }

    /// The single canonical key for a node from its 8 orientations: the tiny-table
    /// graph-iso key (tagged) when the available graph is small enough to merge
    /// cheaply, else the incremental D4 key (tagged into a disjoint namespace).
    /// One key per node -- no D4-and-iso double probe.
    #[inline]
    fn node_key(&self, q: &Queens, orient: &[Bits; 8]) -> Bits {
        let avail = orient[0];
        if avail.popcount() <= self.iso_max_avail {
            // `iso_key_tiny_table` is exact for popcount <= SMALL_CANON_MAX (7); above
            // that (only with a raised `QUEENS_KEY_MAX`) fall back to the WL key.
            let h = if avail.popcount() <= 7 {
                q.iso_key_tiny_table(avail)
            } else {
                q.iso_key_fast(avail)
            };
            graph_bits(h)
        } else {
            d4_bits(lex_min8(orient))
        }
    }

    /// Sequential cutoff search (the [`Burr::wins_inc`](super::Burr) twin, single-keyed).
    fn wins_inc(&self, q: &Queens, att: &[[Bits; 8]], orient: &[Bits; 8], key: Bits) -> bool {
        if let Some(w) = self.store.get(key) {
            return w != 0;
        }
        self.store.bump();
        let avail = orient[0];
        let mut result = false;
        for &sq in &q.order {
            if !avail.get(sq) {
                continue;
            }
            let a = &att[sq as usize];
            let child0 = avail.and_not(a[0]);
            if child0 == Bits::ZERO {
                result = true;
                break;
            }
            let child = child_orient(orient, a, child0);
            let ckey = self.node_key(q, &child);
            self.store.prefetch(ckey);
            if !self.wins_inc(q, att, &child, ckey) {
                result = true;
                break;
            }
        }
        self.store.put(key, result as u8);
        result
    }

    /// Recursive parallel cutoff search (the [`Burr::par_wins_inc`](super::Burr) twin).
    fn par_wins_inc(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        orient: &[Bits; 8],
        key: Bits,
        depth: u32,
        min_avail: u32,
    ) -> bool {
        if let Some(w) = self.store.get(key) {
            return w != 0;
        }
        let avail = orient[0];
        if depth >= self.par_depth && avail.popcount() <= min_avail {
            return self.wins_inc(q, att, orient, key);
        }
        self.store.bump();
        let mut moves: [u32; MAXV] = [0; MAXV];
        let mut nc = 0usize;
        for &sq in &q.order {
            if !avail.get(sq) {
                continue;
            }
            if avail.and_not(att[sq as usize][0]) == Bits::ZERO {
                self.store.put(key, 1);
                return true;
            }
            moves[nc] = sq;
            nc += 1;
        }
        let kids = &moves[..nc];
        let recurse = |&sq: &u32| {
            let a = &att[sq as usize];
            let child0 = avail.and_not(a[0]);
            let child = child_orient(orient, a, child0);
            let ckey = self.node_key(q, &child);
            !self.par_wins_inc(q, att, &child, ckey, depth + 1, min_avail)
        };
        let won = if depth.is_multiple_of(2) {
            kids.par_iter().any(recurse)
        } else {
            kids.iter().any(recurse)
        };
        self.store.put(key, won as u8);
        won
    }
}

impl Solver for Fused {
    fn name(&self) -> &'static str {
        "fused"
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        let att = self.att(q);
        let orient = orient_of(q, q.board.and_not(blocked));
        let key = self.node_key(q, &orient);
        let won = self.wins_inc(q, att, &orient, key);
        self.store.drain_local(); // sequential path: only this thread accumulated
        won
    }
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
        let root = orient_of(q, q.board);
        let mut pending: Vec<([Bits; 8], Bits)> = Vec::with_capacity(moves.len());
        for &sq in &moves {
            let a = &att[sq as usize];
            let co = child_orient(&root, a, q.board.and_not(a[0]));
            let ckey = self.node_key(q, &co);
            pending.push((co, ckey));
        }
        let resolve = |co: &[Bits; 8], ckey: Bits| {
            let wins = !self.par_wins_inc(q, att, co, ckey, 1, min_avail);
            self.root_done.fetch_add(1, Ordering::Relaxed);
            wins
        };
        let (first, rest) = pending.split_first().unwrap();
        let won =
            resolve(&first.0, first.1) || rest.par_iter().any(|(co, ckey)| resolve(co, *ckey));
        self.store.drain_all(); // fold every worker's tail tally into the shared totals
        won
    }
    fn nodes(&self) -> u64 {
        self.store.nodes()
    }
    fn cap_bytes(&self) -> u64 {
        self.store.cap_bytes()
    }
    fn report(&self) -> Option<CountReport> {
        self.store.report()
    }
    fn working_set(&self) -> Option<Vec<(Bits, u8)>> {
        self.store.working_set()
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
            "{} rayon workers, {done}/{total} root moves, par-depth {}/min-avail {ma}, iso<= {} (single-key) · {}",
            rayon::current_num_threads(),
            self.par_depth,
            self.iso_max_avail,
            self.store.summary(),
        )
    }
}
