//! `Naive` -- the memo-less ground-truth negamax solver.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// **Naive** -- plain negamax win/loss with the α-β cutoff and *no* memo. The
/// ground truth: slowest, but the reference every other solver is checked against.
#[derive(Default)]
pub struct Naive {
    nodes: AtomicU64,
}

impl Naive {
    pub fn new() -> Self {
        Naive::default()
    }
}

impl Solver for Naive {
    fn name(&self) -> &'static str {
        "naive"
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        self.nodes.fetch_add(1, Ordering::Relaxed);
        let mut result = false;
        for &sq in &q.order {
            if q.is_available(blocked, sq) && !self.wins(q, q.place(blocked, sq)) {
                result = true;
                break;
            }
        }
        result
    }
    fn nodes(&self) -> u64 {
        self.nodes.load(Ordering::Relaxed)
    }
}
