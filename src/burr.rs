//! BuRR -- Bumped Ribbon Retrieval (Dillinger, Hübschle-Schneider, Sanders,
//! Walzer 2022, <https://arxiv.org/abs/2109.01892>).
//!
//! A *static retrieval* structure: built once from a known set of `(key, value)`
//! pairs, it returns the `r`-bit value for any key in the set in ~`1.0..1.1 * r`
//! bits/key -- far below a hash table's `1/load_factor * slot_width`. This is the
//! Chunk-4 lever for the Queens n=16 roadmap: freeze the *solved* positions
//! (proven win/loss) into an immutable layer that never evicts, so queries it
//! serves cost ~1.0x re-expansion (vs the live TT's 1.36x).
//!
//! # How a single ribbon works
//!
//! We solve a sparse linear system over GF(2). Each key contributes one equation:
//! a `W`-bit **coefficient** band placed at a hashed **start** column, with
//! right-hand side = the key's value. We solve for `Z` (one `r`-bit row per
//! column) such that, for every key, `XOR over the band of Z[start+j] == value`.
//! Because the band is narrow (`W = 64`, one `u64`), Gaussian elimination is a
//! handful of `u64` XORs per key -- "on-the-fly" incremental GE, no dense matrix.
//!
//! A query recomputes `(start, coeff)` from the key and XORs the `Z` rows the
//! coefficient selects -- no search structure, just `popcount(coeff)` array reads.
//!
//! # Bumping (the "BuRR" part)
//!
//! A key whose equation reduces to `0 == nonzero` (linearly dependent and
//! inconsistent) can't be placed -- it is **bumped** to a fallback layer built
//! over only the failures with a fresh seed. A few layers at high load drive the
//! total overhead to ~1.05-1.1x while never failing to build.
//!
//! # Membership (the [`Archive`] wrapper)
//!
//! A bare ribbon returns *garbage* for keys not in the set (it stores no key
//! material). [`Archive`] stores `fp_bits` of key fingerprint alongside the value;
//! a query accepts a layer's answer only if the retrieved fingerprint matches, so
//! a non-member is rejected with probability `1 - layers * 2^-fp_bits`. This is
//! what makes a cascade usable as a TT tier -- and its cost (the fingerprint must
//! be wide enough that `FP_rate * out-of-set-queries << 1`) is the key design
//! finding for the live integration (see the roadmap's BuRR note).

use std::io::{self, Read, Write};

/// Ribbon width: the coefficient band is one `u64`, so elimination is `u64` XORs.
const W: usize = 64;

/// SplitMix64 finalizer -- a strong 1:1 bit mixer for deriving hashes from a key.
#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// Lemire `fastrange`: map a 64-bit hash uniformly into `[0, n)` with one widening
/// multiply -- a power-of-two-free `hash % n`.
#[inline]
pub(crate) fn fastrange(h: u64, n: u64) -> u64 {
    ((h as u128).wrapping_mul(n as u128) >> 64) as u64
}

/// `(start_column, coefficient)` for `key` under `seed`. `start` indexes a pivot
/// slot in `[0, m)`; the band then spans columns `[start, start+W-1]`. The low bit
/// of `coeff` is forced set so `start` is always a valid pivot to begin reduction.
/// Two independent mixes (different seed offsets) give start and coeff so they
/// don't share bits.
#[inline]
fn band(seed: u64, key: u64, m: u64) -> (usize, u64) {
    let h_start = mix64(key.wrapping_add(seed));
    let h_coeff = mix64(key ^ seed ^ 0x9E37_79B9_7F4A_7C15);
    let start = fastrange(h_start, m) as usize;
    (start, h_coeff | 1)
}

/// A single ribbon layer: the solved GF(2) system as `r`-bit rows `z`, plus the
/// `seed` and pivot-slot count `m` needed to recompute a key's band on query.
///
/// There are `cols = m + W` columns: pivots live in `[0, m)`, and the trailing `W`
/// columns absorb the band overflow of a pivot near `m-1` (its band reaches column
/// `m+W-2`), so every column access is in-bounds by construction -- no boundary
/// masking. The `r`-bit rows are **bit-packed** into `z` (column `i` occupies bits
/// `[i*r, i*r+r)`), so resident memory is `cols * r` bits -- the actual ~`1.1*r`
/// bits/key density. (One trailing pad word lets the last column's two-word read run
/// off the end harmlessly.)
pub struct Ribbon {
    seed: u64,
    m: u64,
    r: u32,
    cols: u64,
    z: Box<[u64]>,
}

/// Low-`r`-bit mask (`r` in `1..=64`).
#[inline]
fn rmask(r: u32) -> u64 {
    if r >= 64 {
        u64::MAX
    } else {
        (1u64 << r) - 1
    }
}

/// Read the `r`-bit field for column `i` from a packed bit-vector (`z`). The field
/// can straddle two words; the trailing pad word makes the `+1` read in-bounds.
#[inline]
fn read_packed(z: &[u64], r: u32, i: usize) -> u64 {
    let bit = i as u64 * r as u64;
    let w = (bit >> 6) as usize;
    let off = (bit & 63) as u32;
    let lo = z[w] >> off;
    // off + r > 64 implies off > 0 (since r <= 64), so 64 - off is in 1..=63.
    let hi = if off + r > 64 {
        z[w + 1] << (64 - off)
    } else {
        0
    };
    (lo | hi) & rmask(r)
}

/// Write the `r`-bit `val` into column `i` of a packed bit-vector (`z`), assuming
/// the field starts cleared (back-substitution fills each column once).
#[inline]
fn write_packed(z: &mut [u64], r: u32, i: usize, val: u64) {
    let v = val & rmask(r);
    let bit = i as u64 * r as u64;
    let w = (bit >> 6) as usize;
    let off = (bit & 63) as u32;
    z[w] |= v << off;
    if off + r > 64 {
        z[w + 1] |= v >> (64 - off);
    }
}

/// Outcome of inserting one key's equation into the on-the-fly GE state.
enum Insert {
    /// Placed at a fresh pivot, or absorbed as a (consistent) dependent row.
    Ok,
    /// Reduced to `0 == nonzero`, or ran past the array -- bump to a fallback.
    Bumped,
}

impl Ribbon {
    /// Build a ribbon over `pairs` (`(key, value)`, value in the low `r` bits) at
    /// pivot-slot count `m`. Returns the layer plus the keys that had to be bumped
    /// (to be handed to a fallback layer). `m` should exceed `pairs.len()` by the
    /// chosen load slack; `r <= 64`.
    fn build(seed: u64, m: u64, r: u32, pairs: &[(u64, u64)]) -> (Ribbon, Vec<(u64, u64)>) {
        debug_assert!(r <= 64);
        let cols = m as usize + W; // pivots in [0,m); +W absorbs band overflow
                                   // On-the-fly Gaussian elimination state: per column, the stored row's
                                   // coefficient (bit 0 = this column = the pivot) and its r-bit rhs.
        let mut coeff = vec![0u64; cols];
        let mut rhs = vec![0u64; cols];
        let vmask = if r == 64 { u64::MAX } else { (1u64 << r) - 1 };
        let mut bumped = Vec::new();
        for &(key, val) in pairs {
            if let Insert::Bumped = Self::insert(seed, m, &mut coeff, &mut rhs, key, val & vmask) {
                bumped.push((key, val));
            }
        }
        // Back-substitute high column -> low into an unpacked scratch (one u64 per
        // column): a pivot row fixes its column from the rhs and the already-solved
        // higher columns its coefficient selects. Free columns (empty pivot) stay 0.
        let mut zfull = vec![0u64; cols];
        for i in (0..cols).rev() {
            let c = coeff[i];
            if c == 0 {
                continue;
            }
            let mut acc = rhs[i];
            let mut hi = c & !1u64; // non-pivot band bits
            while hi != 0 {
                let k = hi.trailing_zeros() as usize;
                acc ^= zfull[i + k];
                hi &= hi - 1;
            }
            zfull[i] = acc;
        }
        // Bit-pack the solution to r bits/column (the resident form). +1 pad word so
        // the last column's straddling read never runs off the end.
        let words = (cols as u64 * r as u64).div_ceil(64) as usize + 1;
        let mut z = vec![0u64; words];
        for (i, &v) in zfull.iter().enumerate() {
            write_packed(&mut z, r, i, v);
        }
        (
            Ribbon {
                seed,
                m,
                r,
                cols: cols as u64,
                z: z.into_boxed_slice(),
            },
            bumped,
        )
    }

    /// Reduce one key's equation against the current GE state, placing it at the
    /// first free pivot it reaches. Walking the pivot rightward stays in-bounds:
    /// the band is always `<= W` wide and anchored at the current pivot, and a
    /// pivot at `>= m` (the overflow region) is treated as a bump.
    #[inline]
    fn insert(seed: u64, m: u64, coeff: &mut [u64], rhs: &mut [u64], key: u64, val: u64) -> Insert {
        let (start, c0) = band(seed, key, m);
        let mut i = start;
        let mut co = c0; // bit 0 == column i
        let mut v = val;
        loop {
            if co == 0 {
                // Fully reduced: consistent iff rhs also zeroed out. Checked *before* the
                // shift below so a fully-eliminated `co` (whose `trailing_zeros()` is 64)
                // never runs an overflowing `co >> 64` -- a debug panic (release masks the
                // count to `>> 0`, so this was a debug-only false failure).
                return if v == 0 { Insert::Ok } else { Insert::Bumped };
            }
            // Align so the lowest set bit (next pivot) is at column i.
            let tz = co.trailing_zeros() as usize;
            i += tz;
            co >>= tz;
            if i >= m as usize {
                return Insert::Bumped; // ran into the overflow region
            }
            if coeff[i] == 0 {
                coeff[i] = co;
                rhs[i] = v;
                return Insert::Ok;
            }
            co ^= coeff[i]; // eliminate this pivot (both have bit 0 set -> clears)
            v ^= rhs[i];
        }
    }

    /// The stored `r`-bit value for `key` (garbage if `key` was not built in).
    #[inline]
    pub fn get(&self, key: u64) -> u64 {
        let (start, coeff) = band(self.seed, key, self.m);
        let mut acc = 0u64;
        let mut bits = coeff;
        while bits != 0 {
            let k = bits.trailing_zeros() as usize;
            acc ^= read_packed(&self.z, self.r, start + k);
            bits &= bits - 1;
        }
        acc & rmask(self.r)
    }

    /// Resident bits = `cols * r` (the packed `z`). Bumping keeps `m / n` near 1,
    /// so bits/key approaches `r`.
    fn bits(&self) -> u64 {
        self.cols * self.r as u64
    }

    fn write_to<Wr: Write>(&self, w: &mut Wr) -> io::Result<()> {
        w.write_all(&self.seed.to_le_bytes())?;
        w.write_all(&self.m.to_le_bytes())?;
        w.write_all(&self.cols.to_le_bytes())?;
        w.write_all(&self.r.to_le_bytes())?;
        w.write_all(&(self.z.len() as u64).to_le_bytes())?;
        for &word in self.z.iter() {
            w.write_all(&word.to_le_bytes())?;
        }
        Ok(())
    }

    fn read_from<Rd: Read>(r: &mut Rd) -> io::Result<Ribbon> {
        let seed = read_u64(r)?;
        let m = read_u64(r)?;
        let cols = read_u64(r)?;
        let rbits = read_u32(r)?;
        let zlen = read_u64(r)? as usize;
        let mut z = vec![0u64; zlen];
        for word in z.iter_mut() {
            *word = read_u64(r)?;
        }
        Ok(Ribbon {
            seed,
            m,
            r: rbits,
            cols,
            z: z.into_boxed_slice(),
        })
    }
}

/// Magic + version for a serialized [`Archive`].
const ARCHIVE_MAGIC: [u8; 8] = *b"QNSBURR\0";
const ARCHIVE_VERSION: u32 = 1;

/// A bumped-ribbon **retrieval archive with membership**: a cascade of [`Ribbon`]
/// layers each storing `fp_bits` of key fingerprint above `val_bits` of value.
///
/// A query walks the layers in order and accepts the first whose retrieved
/// fingerprint matches the key's -- so a key in the set returns its value, and a
/// key *not* in the set is rejected with false-positive probability
/// `~ layers * 2^-fp_bits` (a false positive returns a wrong value, so `fp_bits`
/// must be sized against the expected number of out-of-set queries).
pub struct Archive {
    val_bits: u32,
    fp_bits: u32,
    n_keys: u64,
    layers: Vec<Ribbon>,
}

/// A key's membership fingerprint at the given width (independent of the band
/// hashes; folds a distinct constant). Forced into the high `fp_bits` of the row.
#[inline]
fn fingerprint(key: u64, fp_bits: u32) -> u64 {
    if fp_bits == 0 {
        return 0;
    }
    let mask = if fp_bits == 64 {
        u64::MAX
    } else {
        (1u64 << fp_bits) - 1
    };
    mix64(key ^ 0xD6E8_FEB8_6659_FD93) & mask
}

impl Archive {
    /// Build an archive over `pairs` (`(key, value)`), storing `val_bits` of value
    /// and `fp_bits` of fingerprint per key. `load` is the per-layer fill target
    /// (e.g. 0.90); lower = fewer bumps + more memory. Bumped keys cascade into
    /// fallback layers until none remain.
    ///
    /// `fp_bits` doubles as the cascade's layer-routing signal, so a multi-layer
    /// archive needs `fp_bits > 0` to be sound (a bumped key is recognised at its
    /// layer by the fingerprint). True value-only (`fp_bits = 0`, ~`1.0..1.1`
    /// bits/key) is correct only when the build fits a *single* layer -- the
    /// ply-windowed use where membership is known a priori. The cascade route used
    /// as a TT tier must pay the fingerprint.
    pub fn build(pairs: &[(u64, u64)], val_bits: u32, fp_bits: u32, load: f64) -> Archive {
        assert!(val_bits + fp_bits <= 64, "row width must fit a u64");
        assert!((0.1..1.0).contains(&load), "load must be in (0.1, 1.0)");
        let r = val_bits + fp_bits;
        let vmask = if val_bits == 64 {
            u64::MAX
        } else {
            (1u64 << val_bits) - 1
        };
        // Row payload = fingerprint above value.
        let mut remaining: Vec<(u64, u64)> = pairs
            .iter()
            .map(|&(k, v)| (k, (fingerprint(k, fp_bits) << val_bits) | (v & vmask)))
            .collect();
        let n_keys = remaining.len() as u64;
        let mut layers = Vec::new();
        let mut seed = 0x5DEE_CE66_D3B7_1A2Fu64;
        while !remaining.is_empty() {
            // m: enough slots for the load target, with a floor so a tiny last
            // layer is sparse enough to (almost) always succeed.
            let m = (((remaining.len() as f64) / load).ceil() as u64)
                .max(remaining.len() as u64 + 2 * W as u64);
            let (ribbon, bumped) = Ribbon::build(seed, m, r, &remaining);
            layers.push(ribbon);
            if bumped.len() == remaining.len() {
                // No progress (vanishingly unlikely below load 1.0): perturb the
                // seed and retry the same set rather than spin.
                layers.pop();
            }
            remaining = bumped;
            seed = mix64(seed);
            assert!(
                layers.len() < 64,
                "ribbon cascade exceeded 64 layers -- load too high?"
            );
        }
        Archive {
            val_bits,
            fp_bits,
            n_keys,
            layers,
        }
    }

    /// The stored value for `key`, or `None` if `key` is (almost certainly) not in
    /// the archive. Walks layers; the first fingerprint match wins.
    #[inline]
    pub fn get(&self, key: u64) -> Option<u64> {
        let want = fingerprint(key, self.fp_bits);
        let vmask = if self.val_bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.val_bits) - 1
        };
        for layer in &self.layers {
            let row = layer.get(key);
            if (row >> self.val_bits) == want {
                return Some(row & vmask);
            }
        }
        None
    }

    /// Total resident bits across all layers.
    pub fn bits(&self) -> u64 {
        self.layers.iter().map(Ribbon::bits).sum()
    }

    /// Resident bits per built-in key -- the headline density (target ~`1.1 * r`).
    pub fn bits_per_key(&self) -> f64 {
        if self.n_keys == 0 {
            return 0.0;
        }
        self.bits() as f64 / self.n_keys as f64
    }

    pub fn n_keys(&self) -> u64 {
        self.n_keys
    }
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
    pub fn val_bits(&self) -> u32 {
        self.val_bits
    }
    pub fn fp_bits(&self) -> u32 {
        self.fp_bits
    }

    /// Serialize to `w` (header + each layer). Wrap `w` in a zstd encoder at the
    /// call site if compression is wanted (the layers are high-entropy, so it
    /// barely helps -- unlike the mostly-zero TT image).
    pub fn write_to<Wr: Write>(&self, w: &mut Wr) -> io::Result<()> {
        w.write_all(&ARCHIVE_MAGIC)?;
        w.write_all(&ARCHIVE_VERSION.to_le_bytes())?;
        w.write_all(&self.val_bits.to_le_bytes())?;
        w.write_all(&self.fp_bits.to_le_bytes())?;
        w.write_all(&self.n_keys.to_le_bytes())?;
        w.write_all(&(self.layers.len() as u64).to_le_bytes())?;
        for layer in &self.layers {
            layer.write_to(w)?;
        }
        Ok(())
    }

    /// Read an archive written by [`write_to`](Self::write_to), hard-erroring on a
    /// bad magic or version.
    pub fn read_from<Rd: Read>(r: &mut Rd) -> io::Result<Archive> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if magic != ARCHIVE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a queens BuRR archive (bad magic)",
            ));
        }
        let version = read_u32(r)?;
        if version != ARCHIVE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("archive version {version}, this build {ARCHIVE_VERSION}"),
            ));
        }
        let val_bits = read_u32(r)?;
        let fp_bits = read_u32(r)?;
        let n_keys = read_u64(r)?;
        let n_layers = read_u64(r)? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(Ribbon::read_from(r)?);
        }
        Ok(Archive {
            val_bits,
            fp_bits,
            n_keys,
            layers,
        })
    }
}

/// Magic + version for a serialized [`ShardedArchive`].
const SHARDED_MAGIC: [u8; 8] = *b"QNSBSHRD";
const SHARDED_VERSION: u32 = 1;

/// A key-partitioned set of independent [`Archive`]s. Sharding lets the freeze
/// build n=16 (billions of keys) in bounded RAM: shard `k` is built from a pass
/// over the dump that keeps only the keys hashing to `k`, so the in-flight GE
/// state is `1/shards` of the whole. A query routes by the same hash, so it touches
/// exactly one shard. (`shards == 1` is just a single [`Archive`] with a wrapper.)
pub struct ShardedArchive {
    salt: u64,
    shards: Vec<Archive>,
}

/// Salt for the shard-routing hash, kept distinct from the band/fingerprint mixes.
const SHARD_SALT: u64 = 0x51F0_9C2B_77A3_E6D1;

impl ShardedArchive {
    /// Which shard a key belongs to, for a given shard count.
    #[inline]
    pub fn shard_of(n_shards: usize, key: u64) -> usize {
        fastrange(mix64(key ^ SHARD_SALT), n_shards as u64) as usize
    }

    /// Assemble from per-shard archives (shard `k` must have been built from
    /// exactly the keys with `shard_of(shards.len(), key) == k`).
    pub fn from_shards(shards: Vec<Archive>) -> ShardedArchive {
        ShardedArchive {
            salt: SHARD_SALT,
            shards,
        }
    }

    /// The stored value for `key`, or `None` if not present -- routes to one shard.
    #[inline]
    pub fn get(&self, key: u64) -> Option<u64> {
        let s = fastrange(mix64(key ^ self.salt), self.shards.len() as u64) as usize;
        self.shards[s].get(key)
    }

    pub fn n_shards(&self) -> usize {
        self.shards.len()
    }
    pub fn n_keys(&self) -> u64 {
        self.shards.iter().map(Archive::n_keys).sum()
    }
    pub fn bits(&self) -> u64 {
        self.shards.iter().map(Archive::bits).sum()
    }
    pub fn bits_per_key(&self) -> f64 {
        let n = self.n_keys();
        if n == 0 {
            0.0
        } else {
            self.bits() as f64 / n as f64
        }
    }

    pub fn write_to<Wr: Write>(&self, w: &mut Wr) -> io::Result<()> {
        w.write_all(&SHARDED_MAGIC)?;
        w.write_all(&SHARDED_VERSION.to_le_bytes())?;
        w.write_all(&self.salt.to_le_bytes())?;
        w.write_all(&(self.shards.len() as u64).to_le_bytes())?;
        for a in &self.shards {
            a.write_to(w)?;
        }
        Ok(())
    }

    pub fn read_from<Rd: Read>(r: &mut Rd) -> io::Result<ShardedArchive> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if magic != SHARDED_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a queens sharded BuRR archive (bad magic)",
            ));
        }
        let version = read_u32(r)?;
        if version != SHARDED_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("sharded archive version {version}, this build {SHARDED_VERSION}"),
            ));
        }
        let salt = read_u64(r)?;
        let n = read_u64(r)? as usize;
        let mut shards = Vec::with_capacity(n);
        for _ in 0..n {
            shards.push(Archive::read_from(r)?);
        }
        Ok(ShardedArchive { salt, shards })
    }
}

#[inline]
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[inline]
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random keys (no `rand` dep; `Math.random` is banned in
    /// the harness and we want reproducible tests anyway).
    fn keys(n: usize, salt: u64) -> Vec<u64> {
        (0..n as u64)
            .map(|i| mix64(i.wrapping_add(salt) ^ 0xABCD))
            .collect()
    }

    /// Every built-in key retrieves its exact value, across value widths and sizes.
    /// A 32-bit fingerprint routes bumped keys past the layers they're absent from
    /// (a member is mis-answered only if an earlier layer's garbage fingerprint
    /// collides -- `~layers * 2^-32`, astronomically unlikely for these inputs).
    #[test]
    fn ribbon_round_trips_all_values() {
        for (n, vb) in [
            (1usize, 1u32),
            (10, 1),
            (1000, 1),
            (10_000, 8),
            (50_000, 16),
        ] {
            let vmask = (1u64 << vb) - 1;
            let pairs: Vec<(u64, u64)> = keys(n, vb as u64)
                .into_iter()
                .map(|k| (k, mix64(k) & vmask))
                .collect();
            let arch = Archive::build(&pairs, vb, 32, 0.90);
            for &(k, v) in &pairs {
                assert_eq!(arch.get(k), Some(v), "n={n} vb={vb} key={k:#x}");
            }
        }
    }

    /// Membership: built-in keys return their value; absent keys are rejected at
    /// the expected false-positive rate for the fingerprint width.
    #[test]
    fn archive_membership_and_fp_rate() {
        let n = 200_000;
        let fp_bits = 12;
        let pairs: Vec<(u64, u64)> = keys(n, 7).into_iter().map(|k| (k, k & 1)).collect();
        let arch = Archive::build(&pairs, 1, fp_bits, 0.90);
        // All members hit with the right value.
        for &(k, v) in &pairs {
            assert_eq!(arch.get(k), Some(v), "member key={k:#x}");
        }
        // Disjoint probe set: measured FP rate near layers * 2^-fp_bits.
        let probes = keys(n, 0xDEAD_BEEF); // different salt -> disjoint w.h.p.
        let fp = probes.iter().filter(|&&k| arch.get(k).is_some()).count();
        let rate = fp as f64 / n as f64;
        let bound = arch.n_layers() as f64 * 2f64.powi(-(fp_bits as i32)) * 4.0; // generous
        assert!(
            rate < bound,
            "FP rate {rate:.5} exceeded bound {bound:.5} ({} layers)",
            arch.n_layers()
        );
    }

    /// Serialization round-trips bit-for-bit (values + membership preserved).
    #[test]
    fn archive_serialization_round_trips() {
        let pairs: Vec<(u64, u64)> = keys(20_000, 3).into_iter().map(|k| (k, k & 1)).collect();
        let arch = Archive::build(&pairs, 1, 16, 0.90);
        let mut buf = Vec::new();
        arch.write_to(&mut buf).unwrap();
        let back = Archive::read_from(&mut buf.as_slice()).unwrap();
        assert_eq!(back.n_keys(), arch.n_keys());
        assert_eq!(back.n_layers(), arch.n_layers());
        for &(k, v) in &pairs {
            assert_eq!(back.get(k), Some(v));
        }
    }

    /// The cascade always terminates (well under the 64-layer guard) even at an
    /// aggressive load, and round-trips exactly with a wide enough fingerprint to
    /// route the larger bumped fraction a high load produces.
    #[test]
    fn cascade_terminates_at_high_load() {
        let pairs: Vec<(u64, u64)> = keys(100_000, 9).into_iter().map(|k| (k, k & 1)).collect();
        let arch = Archive::build(&pairs, 1, 32, 0.98);
        assert!((1..64).contains(&arch.n_layers()));
        for &(k, v) in &pairs {
            assert_eq!(arch.get(k), Some(v));
        }
    }

    /// A sharded archive built shard-by-shard (the freeze's bounded-RAM path)
    /// retrieves every key and survives serialization.
    #[test]
    fn sharded_archive_round_trips() {
        let n_shards = 8;
        let pairs: Vec<(u64, u64)> = keys(80_000, 11).into_iter().map(|k| (k, k & 1)).collect();
        // Build each shard from exactly its routed keys, as the freeze passes do.
        let shards: Vec<Archive> = (0..n_shards)
            .map(|s| {
                let sub: Vec<(u64, u64)> = pairs
                    .iter()
                    .copied()
                    .filter(|&(k, _)| ShardedArchive::shard_of(n_shards, k) == s)
                    .collect();
                Archive::build(&sub, 1, 24, 0.90)
            })
            .collect();
        let arch = ShardedArchive::from_shards(shards);
        assert_eq!(arch.n_keys(), pairs.len() as u64);
        let mut buf = Vec::new();
        arch.write_to(&mut buf).unwrap();
        let back = ShardedArchive::read_from(&mut buf.as_slice()).unwrap();
        for &(k, v) in &pairs {
            assert_eq!(back.get(k), Some(v), "key={k:#x}");
        }
    }
}
