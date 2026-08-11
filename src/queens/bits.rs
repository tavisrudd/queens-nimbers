//! The fixed-width board bitset.

use super::WORDS;

/// A fixed-width board bitset (`WORDS*64` bits). `Ord`/`Hash` are the derived
/// lexicographic order on the words -- a total order, all we need to pick a
/// canonical representative and to key the memo table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Debug)]
#[repr(transparent)] // hot-struct discipline (CLAUDE.md #4): explicit layout = the inner `[u64; WORDS]`.
pub struct Bits(pub(crate) [u64; WORDS]);

// #7: a size/align regression (e.g. WORDS bumped, or a field snuck in) fails the build.
const _: () =
    assert!(std::mem::size_of::<Bits>() == WORDS * 8 && std::mem::align_of::<Bits>() == 8);

impl Bits {
    pub(crate) const ZERO: Bits = Bits([0; WORDS]);

    /// An empty bitset (no bits set).
    #[inline]
    pub fn empty() -> Bits {
        Bits::ZERO
    }
    /// Set bit `i`.
    #[inline]
    pub fn set(&mut self, i: u32) {
        self.0[(i / 64) as usize] |= 1u64 << (i % 64);
    }
    /// Is bit `i` set?
    #[inline]
    pub fn get(self, i: u32) -> bool {
        self.0[(i / 64) as usize] & (1u64 << (i % 64)) != 0
    }
    #[inline]
    pub(crate) fn or(self, o: Bits) -> Bits {
        let mut r = self.0;
        for (rk, &ok) in r.iter_mut().zip(o.0.iter()) {
            *rk |= ok;
        }
        Bits(r)
    }
    #[inline]
    pub(crate) fn and_not(self, o: Bits) -> Bits {
        let mut r = self.0;
        for (rk, &ok) in r.iter_mut().zip(o.0.iter()) {
            *rk &= !ok;
        }
        Bits(r)
    }
    #[inline]
    pub(crate) fn and(self, o: Bits) -> Bits {
        let mut r = self.0;
        for (rk, &ok) in r.iter_mut().zip(o.0.iter()) {
            *rk &= ok;
        }
        Bits(r)
    }
    /// The number of set bits (board squares).
    #[inline]
    pub fn popcount(self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
    /// The lowest set bit index, or `None` if empty.
    #[inline]
    pub(crate) fn lowest(self) -> Option<u32> {
        self.0
            .iter()
            .enumerate()
            .find(|(_, &w)| w != 0)
            .map(|(k, &w)| k as u32 * 64 + w.trailing_zeros())
    }
    /// Call `f` with each set bit index (ascending).
    // `inline(always)`: the hot callers (`wK_get`/`tiny_table_index` vert-scatter, on the getK +
    // band majority of nodes) pass tiny closures, but left to the compiler's discretion `each` was
    // outlined into a shared `FnMut::call_mut` body (~4.8% of n=16 search cycles in the profile);
    // forcing inline cut cyc/node ~1.4% (n=16 A/B).
    #[inline(always)]
    pub(crate) fn each<F: FnMut(u32)>(self, mut f: F) {
        for (k, &w) in self.0.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = w.trailing_zeros();
                f(k as u32 * 64 + b);
                w &= w - 1;
            }
        }
    }
}

/// A bitset with exactly bit `i` set.
#[inline]
pub(crate) fn single(i: u32) -> Bits {
    let mut b = Bits::ZERO;
    b.set(i);
    b
}
