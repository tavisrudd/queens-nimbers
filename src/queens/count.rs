//! Distinct-position instrumentation: HyperLogLog + optional exact set.

use super::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

/// The result of a distinct-position measurement (see [`Tt::new_counting`]).
#[derive(Clone, Copy, Debug)]
pub struct CountReport {
    /// HyperLogLog estimate of the distinct positions the search visited.
    pub estimate: f64,
    /// Exact distinct count, if a hash set was kept (`--exact`, small boards).
    pub exact: Option<u64>,
    /// HyperLogLog register count (`2^p`), for reporting the estimator's error.
    pub registers: u64,
}

/// The instrumentation a counting [`QueensTt`] folds each visited key into: a
/// HyperLogLog of every looked-up key (the distinct estimate) and, optionally, an
/// exact key→value map (small boards only). The map is populated at `put` -- where
/// the win/loss value is known and exact -- not by peeking the lossy fingerprint TT,
/// whose index collisions would return stale values and pollute the `--iso`
/// win/loss-consistency check.
pub(crate) struct Counter {
    pub(crate) hll: Hll,
    pub(crate) exact: Option<Mutex<HashMap<Bits, u8>>>,
}

impl Counter {
    /// Record a solved key's exact value (called from `put`).
    #[inline]
    pub(crate) fn record(&self, key: Bits, val: u8) {
        if let Some(map) = &self.exact {
            map.lock().unwrap().insert(key, val);
        }
    }
}

/// A dense **HyperLogLog** (Flajolet, Fusy, Gandouet & Meunier, 2007) cardinality
/// estimator over 64-bit key hashes. Lock-free under the parallel solver: each of
/// the `2^p` registers is an [`AtomicU8`] updated with `fetch_max`. State is `2^p`
/// bytes (p=16 ⇒ 64 KB) for a standard error of ≈ `1.04/√(2^p)` (p=16 ⇒ ~0.4%) --
/// ample to size a transposition table. Pure instrumentation: never affects the
/// search result.
pub struct Hll {
    pub(crate) p: u32,
    pub(crate) registers: Vec<AtomicU8>,
}

impl Hll {
    /// A HyperLogLog with `2^p` registers. `p` in `4..=18` is sensible.
    pub fn new(p: u32) -> Self {
        Hll {
            p,
            registers: (0..1u64 << p).map(|_| AtomicU8::new(0)).collect(),
        }
    }

    /// The register index and ρ contribution of a key. ρ = 1 + (leading zeros of the
    /// remaining 64-p bits); the sentinel bit at position p-1 caps ρ at 64-p+1 when
    /// those bits are all zero. Shared by the atomic [`add`](Self::add) and the
    /// thread-local [`add_local`](Self::add_local).
    #[inline]
    fn idx_rho(&self, key: Bits) -> (usize, u8) {
        let h = Self::hash(key);
        let idx = (h >> (64 - self.p)) as usize; // top p bits index the register
        let rho = ((h << self.p) | (1u64 << (self.p - 1))).leading_zeros() as u8 + 1;
        (idx, rho)
    }

    /// Fold a board key in, hashed with a mixer independent of [`QueensTt::hash128`]
    /// (so the estimate is not coupled to the table's slot mapping).
    #[inline]
    pub fn add(&self, key: Bits) {
        let (idx, rho) = self.idx_rho(key);
        self.registers[idx].fetch_max(rho, Ordering::Relaxed);
    }

    /// Number of registers (`2^p`) -- the width a thread-local register slice must match.
    #[inline]
    pub(crate) fn register_count(&self) -> usize {
        self.registers.len()
    }

    /// Fold a key into a **thread-local** register slice (`2^p` bytes): a plain byte
    /// max, no atomics, no cross-core coherence -- the hot-loop form. The slice is
    /// merged into the shared registers off the hot path by [`merge_from`](Self::merge_from).
    #[inline]
    pub(crate) fn add_local(&self, key: Bits, local: &mut [u8]) {
        let (idx, rho) = self.idx_rho(key);
        if rho > local[idx] {
            local[idx] = rho;
        }
    }

    /// Merge a thread-local register slice into the shared registers (register-wise
    /// max). Called ~once a second + at drain, so the `fetch_max` traffic is off the
    /// hot loop entirely. Idempotent (re-merging the same slice changes nothing), so
    /// the caller need not reset its slice between merges -- only at solve end.
    pub(crate) fn merge_from(&self, local: &[u8]) {
        for (shared, &v) in self.registers.iter().zip(local) {
            if v != 0 {
                shared.fetch_max(v, Ordering::Relaxed);
            }
        }
    }

    /// The estimated number of distinct keys folded in, with the standard
    /// small-range (linear-counting) correction; the large-range correction is
    /// unnecessary with a 64-bit hash.
    pub fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let alpha = 0.7213 / (1.0 + 1.079 / m); // valid for p >= 7; we use p >= 14
        let mut sum = 0.0f64;
        let mut zeros = 0u64;
        for r in &self.registers {
            let v = r.load(Ordering::Relaxed);
            sum += 1.0 / (1u64 << v) as f64; // 2^-v
            zeros += (v == 0) as u64;
        }
        let raw = alpha * m * m / sum;
        if raw <= 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln() // linear counting for small cardinalities
        } else {
            raw
        }
    }

    /// A high-avalanche 64-bit hash of the key (FNV-1a mix + splitmix64 finalizer),
    /// deliberately distinct from [`QueensTt::hash128`] so estimator accuracy does
    /// not depend on the table's hashing.
    #[inline]
    fn hash(key: Bits) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64; // FNV-1a offset basis
        for &w in &key.0 {
            h ^= w;
            h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
        }
        h ^= h >> 30;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        h
    }
}
