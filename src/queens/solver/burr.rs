//! `Burr` -- the [`Incremental`](super::Incremental) A3 kernel over a
//! [`BurrStore`](crate::queens::BurrStore) instead of a flat [`QueensTt`] (Chunk 4,
//! "BuRR live"). Same per-node key path (the 8 dihedral orientations carried down the
//! DFS, `lex_min8` ≡ `pos_key`), same parity-aware rayon root parallelism, same
//! odd-board O(1) theorem -- so the node count, distinct working set, and verdict are
//! identical to `incremental`. **What changes is the table:** the store is
//! log-structured (a small mutable memtable that freezes into eviction-free BuRR
//! segments), so the working set that fits without re-search is `memtable ∪ segments`
//! ≈ the whole distinct set -- collapsing n=16's capacity re-expansion instead of a
//! faster node. Reuses the validated A3 helpers (`orient_of`/`lex_min8`/`build_att`/
//! `child_orient`) so the key is byte-identical to `incremental`.

use super::incremental::{build_att, child_orient, lex_min8, orient_of};
use super::*;
use crate::queens::BurrStore;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

/// **Burr** -- the A3 DFS-resident solver over the log-structured BuRR store.
pub struct Burr {
    store: BurrStore,
    att: OnceLock<Box<[[Bits; 8]]>>,
    par_depth: u32,
    par_min_avail: Option<u32>,
    eff_min_avail: AtomicU32,
    root_done: AtomicU64,
    root_total: AtomicU64,
}

impl Burr {
    pub fn new(bits: u32) -> Self {
        Self::from_store(BurrStore::new(bits))
    }

    /// As [`Burr::new`], but counting the distinct positions visited (`--distinct`).
    pub fn new_counting(bits: u32, hll_p: u32) -> Self {
        Self::from_store(BurrStore::new_counting(bits, hll_p))
    }

    /// A store with a forced freeze threshold -- the test path that exercises the LSM
    /// round-trip on a brute-forceable board.
    pub fn with_freeze_at(bits: u32, freeze_at: u64) -> Self {
        Self::from_store(BurrStore::with_freeze_at(bits, freeze_at))
    }

    fn from_store(store: BurrStore) -> Self {
        Burr {
            store,
            att: OnceLock::new(),
            par_depth: par_depth(),
            par_min_avail: par_min_avail_override(),
            eff_min_avail: AtomicU32::new(u32::MAX),
            root_done: AtomicU64::new(0),
            root_total: AtomicU64::new(0),
        }
    }

    #[inline]
    fn att(&self, q: &Queens) -> &[[Bits; 8]] {
        self.att.get_or_init(|| build_att(q))
    }

    /// Sequential cutoff search (the `Incremental::wins_inc` twin over the store).
    fn wins_inc(&self, q: &Queens, att: &[[Bits; 8]], orient: &[Bits; 8], key: Bits) -> bool {
        if let Some(w) = self.store.get(key) {
            return w != 0;
        }
        self.store.bump();
        let avail = orient[0]; // identity image == the available mask
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
            let ckey = lex_min8(&child);
            self.store.prefetch(ckey);
            if !self.wins_inc(q, att, &child, ckey) {
                result = true;
                break;
            }
        }
        self.store.put(key, result as u8);
        result
    }

    /// Recursive parallel cutoff search (the `Incremental::par_wins_inc` twin).
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
            let ckey = lex_min8(&child);
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

impl Solver for Burr {
    fn name(&self) -> &'static str {
        "burr"
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        let att = self.att(q);
        let orient = orient_of(q, q.board.and_not(blocked));
        let key = lex_min8(&orient);
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
        // Cold store: no warm pre-seeding (unlike a resumed `Incremental`); every
        // root is pending in order.
        let mut pending: Vec<([Bits; 8], Bits)> = Vec::with_capacity(moves.len());
        for &sq in &moves {
            let a = &att[sq as usize];
            let co = child_orient(&root, a, q.board.and_not(a[0]));
            let ckey = lex_min8(&co);
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
            "{} rayon workers, {done}/{total} root moves, par-depth {}/min-avail {ma} · {}",
            rayon::current_num_threads(),
            self.par_depth,
            self.store.summary(),
        )
    }
    // No `tt()` -- the LSM store has no single dumpable table yet (checkpoint/resume
    // for `burr` is a later lever); checkpointing is disabled for it in the CLI.
}

/// **IsoBurr** -- the BuRR store with a selective graph-isomorphism key over the same
/// A3 DFS-resident orientation kernel as [`Burr`]/[`Incremental`]. Large positions use
/// the cheap incremental D4 canon (`lex_min8`) in a tagged namespace; small fragmented
/// positions use `iso_key_fast`, defaulting to `available <= 7` from the n=16 blend
/// table (`QUEENS_KEY_MAX` overrides).
pub struct IsoBurr {
    store: BurrStore,
    att: OnceLock<Box<[[Bits; 8]]>>,
    par_depth: u32,
    par_min_avail: Option<u32>,
    iso_max_avail: u32,
    eff_min_avail: AtomicU32,
    root_done: AtomicU64,
    root_total: AtomicU64,
}

#[derive(Clone, Copy)]
struct IsoBurrKeys {
    d4: Bits,
    iso: Option<Bits>,
}

impl IsoBurr {
    pub fn new(bits: u32) -> Self {
        Self::from_store(BurrStore::new(bits))
    }

    /// As [`IsoBurr::new`], but counting the distinct tagged D4/iso keys visited.
    pub fn new_counting(bits: u32, hll_p: u32) -> Self {
        Self::from_store(BurrStore::new_counting(bits, hll_p))
    }

    fn from_store(store: BurrStore) -> Self {
        IsoBurr {
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

    #[inline]
    fn d4_key(&self, orient: &[Bits; 8]) -> Bits {
        d4_bits(lex_min8(orient))
    }

    #[inline]
    fn iso_key(&self, q: &Queens, avail: Bits) -> Bits {
        let h = if avail.popcount() <= 7 {
            q.iso_key_tiny_table(avail)
        } else {
            q.iso_key_fast(avail)
        };
        graph_bits(h)
    }

    fn probe(&self, q: &Queens, orient: &[Bits; 8]) -> Result<bool, IsoBurrKeys> {
        let d4 = self.d4_key(orient);
        if let Some(w) = self.store.get(d4) {
            return Ok(w != 0);
        }
        let avail = orient[0];
        let iso = if avail.popcount() <= self.iso_max_avail {
            let iso = self.iso_key(q, avail);
            if let Some(w) = self.store.get(iso) {
                self.store.put(d4, w);
                return Ok(w != 0);
            }
            Some(iso)
        } else {
            None
        };
        Err(IsoBurrKeys { d4, iso })
    }

    #[inline]
    fn put_keys(&self, keys: IsoBurrKeys, val: u8) {
        self.store.put(keys.d4, val);
        if let Some(iso) = keys.iso {
            self.store.put(iso, val);
        }
    }

    fn wins_inc(&self, q: &Queens, att: &[[Bits; 8]], orient: &[Bits; 8]) -> bool {
        let keys = match self.probe(q, orient) {
            Ok(w) => return w,
            Err(keys) => keys,
        };
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
            self.store.prefetch(self.d4_key(&child));
            if !self.wins_inc(q, att, &child) {
                result = true;
                break;
            }
        }
        self.put_keys(keys, result as u8);
        result
    }

    fn par_wins_inc(
        &self,
        q: &Queens,
        att: &[[Bits; 8]],
        orient: &[Bits; 8],
        depth: u32,
        min_avail: u32,
    ) -> bool {
        let keys = match self.probe(q, orient) {
            Ok(w) => return w,
            Err(keys) => keys,
        };
        let avail = orient[0];
        if depth >= self.par_depth && avail.popcount() <= min_avail {
            return self.wins_inc(q, att, orient);
        }
        self.store.bump();
        let mut moves: [u32; MAXV] = [0; MAXV];
        let mut nc = 0usize;
        for &sq in &q.order {
            if !avail.get(sq) {
                continue;
            }
            if avail.and_not(att[sq as usize][0]) == Bits::ZERO {
                self.put_keys(keys, 1);
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
            !self.par_wins_inc(q, att, &child, depth + 1, min_avail)
        };
        let won = if depth.is_multiple_of(2) {
            kids.par_iter().any(recurse)
        } else {
            kids.iter().any(recurse)
        };
        self.put_keys(keys, won as u8);
        won
    }
}

impl Solver for IsoBurr {
    fn name(&self) -> &'static str {
        "iso-burr"
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        let att = self.att(q);
        let orient = orient_of(q, q.board.and_not(blocked));
        let won = self.wins_inc(q, att, &orient);
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
        let mut pending: Vec<[Bits; 8]> = Vec::with_capacity(moves.len());
        for &sq in &moves {
            let a = &att[sq as usize];
            let co = child_orient(&root, a, q.board.and_not(a[0]));
            pending.push(co);
        }
        let resolve = |co: &[Bits; 8]| {
            let wins = !self.par_wins_inc(q, att, co, 1, min_avail);
            self.root_done.fetch_add(1, Ordering::Relaxed);
            wins
        };
        let (first, rest) = pending.split_first().unwrap();
        let won = resolve(first) || rest.par_iter().any(resolve);
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
            "{} rayon workers, {done}/{total} root moves, par-depth {}/min-avail {ma}, iso<= {} · {}",
            rayon::current_num_threads(),
            self.par_depth,
            self.iso_max_avail,
            self.store.summary(),
        )
    }
}
