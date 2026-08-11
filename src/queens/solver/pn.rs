//! `Pn` -- df-pn proof-number search (instructive negative; tiny boards).

use super::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// **Pn** -- depth-first proof-number search (Nagai's df-pn). Instead of a fixed
/// move order it always descends the *most-proving* line, guided by proof and
/// disproof numbers, so it focuses effort on the narrowest path to a verdict --
/// the published state of the art for boolean games (Allis 1994; Nagai 2002).
/// Negamax form on the win/loss tree: a node's proof `φ = min` over children of
/// their disproof `δ`, and `δ = Σ` of their `φ`; a terminal (mover stuck) is a
/// proven loss (`φ = ∞, δ = 0`). The table stores `(φ, δ)` per canonical position.
///
/// **Instructive negative for *this* game.** Verdicts are correct, but plain
/// df-pn hits the well-known df-pn + transposition (graph-history-interaction)
/// pathology: the Non-Attacking Queens game is extraordinarily transposition-
/// dense, so positions solved via one path get re-expanded under reset thresholds
/// when reached via another, and the search explodes past tiny boards (n=8 is
/// fine; n≥9 is impractical). The straightforward `parallel` α-β + TT + symmetry
/// solver dominates it here. Making df-pn competitive needs *careful* DAG-aware
/// proof-number search (Kishimoto on df-pn+transpositions; Čížek-Balko-Schmid
/// 2026) -- kept here as a documented, correct-but-not-competitive experiment.
pub struct Pn {
    tt: PnTt,
    /// The root's `(φ, δ)` once `wins` solves it, so the summary can report the
    /// proof/disproof numbers (`φ=0` ⇒ proven win, `δ=0` ⇒ proven loss).
    root_phi: AtomicU32,
    root_delta: AtomicU32,
}

/// Proof/disproof "infinity" -- a finite sentinel so the arithmetic saturates.
const PN_INF: u32 = u32::MAX;

impl Pn {
    pub fn new(bits: u32) -> Self {
        Pn {
            tt: PnTt::new(bits),
            root_phi: AtomicU32::new(PN_INF),
            root_delta: AtomicU32::new(PN_INF),
        }
    }

    /// The `(φ, δ)` for a *non-terminal* child given its precomputed canonical
    /// `key`: the table entry if present, else a unit leaf `(1, 1)`. Terminal
    /// children never reach here -- `mid` proves the node and returns the moment it
    /// collects one (a terminal child means the opponent cannot move), so this need
    /// not test `no_moves` or canonicalise a key for them.
    #[inline]
    fn child_pd(&self, key: Bits) -> (u32, u32) {
        self.tt.get(key).unwrap_or((1, 1))
    }

    /// df-pn `mid`: expand `blocked` until its `φ ≥ th_phi` or `δ ≥ th_delta`,
    /// always recursing into the child with the smallest disproof number.
    fn mid(&self, q: &Queens, blocked: Bits, th_phi: u32, th_delta: u32) {
        let key = q.pos_key(blocked);
        // Standard df-pn entry check: if the stored numbers already meet the
        // thresholds (in particular a solved node, φ=0/δ=∞ or φ=∞/δ=0), return at
        // once. Without this, a subtree solved via one path is re-expanded every
        // time it recurs through another -- fatal on a transposition-dense game.
        if let Some((phi, delta)) = self.tt.get(key) {
            if phi >= th_phi || delta >= th_delta {
                return;
            }
        }
        self.tt.bump();
        // Collect each non-terminal child with its canonical key *once*. The df-pn
        // loop below revisits this list every time the thresholds tighten, and a
        // child's key is a fixed function of the child, so caching it avoids
        // re-running `canon` (an 8-fold symmetry fold) on every pass. A *terminal*
        // child is decisive on sight -- the opponent then cannot move, so this node
        // is proven (φ=0, δ=∞); return before keying that child or any later one
        // (and before the loop), which also keeps terminals out of `kids` entirely.
        let mut kids: Vec<(Bits, Bits)> =
            Vec::with_capacity(q.board.and_not(blocked).popcount() as usize);
        for &sq in &q.order {
            if q.is_available(blocked, sq) {
                let child = q.place(blocked, sq);
                if q.no_moves(child) {
                    self.tt.put(key, 0, PN_INF); // terminal child ⇒ this node is won
                    return;
                }
                kids.push((child, q.pos_key(child)));
            }
        }
        if kids.is_empty() {
            self.tt.put(key, PN_INF, 0); // terminal node: mover here cannot move ⇒ loses
            return;
        }
        loop {
            // φ(n) = min_c δ(c); δ(n) = Σ_c φ(c). Track the two smallest δ(c).
            let mut phi_n = PN_INF;
            let mut delta_n = 0u32;
            let (mut best, mut best_phi, mut delta1, mut delta2) = (0usize, 1u32, PN_INF, PN_INF);
            let mut proven = false;
            for (i, &(_, ckey)) in kids.iter().enumerate() {
                let (cphi, cdelta) = self.child_pd(ckey);
                if cdelta == 0 {
                    // A non-terminal child the table already proves losing (its mover
                    // loses ⇒ δ(c)=0, φ(c)=∞) proves this node outright: φ(n)=min δ=0,
                    // δ(n)=Σ φ saturates to ∞. No need to scan the remaining children.
                    proven = true;
                    break;
                }
                delta_n = delta_n.saturating_add(cphi);
                if cdelta < phi_n {
                    phi_n = cdelta;
                }
                if cdelta < delta1 {
                    delta2 = delta1;
                    delta1 = cdelta;
                    best = i;
                    best_phi = cphi;
                } else if cdelta < delta2 {
                    delta2 = cdelta;
                }
            }
            if proven {
                self.tt.put(key, 0, PN_INF);
                return;
            }
            if phi_n >= th_phi || delta_n >= th_delta {
                self.tt.put(key, phi_n, delta_n);
                return;
            }
            // Thresholds for the most-proving child (Nagai df-pn).
            let th_phi_c = if th_delta == PN_INF {
                PN_INF
            } else {
                (th_delta - delta_n).saturating_add(best_phi)
            };
            let th_delta_c = th_phi.min(delta2.saturating_add(1));
            self.mid(q, kids[best].0, th_phi_c, th_delta_c);
        }
    }
}

impl Solver for Pn {
    fn name(&self) -> &'static str {
        "pn"
    }
    fn wins(&self, q: &Queens, blocked: Bits) -> bool {
        self.mid(q, blocked, PN_INF, PN_INF); // solve fully
        let pd = self.tt.get(q.pos_key(blocked));
        if blocked == Bits::empty() {
            if let Some((phi, delta)) = pd {
                self.root_phi.store(phi, Ordering::Relaxed);
                self.root_delta.store(delta, Ordering::Relaxed);
            }
        }
        // Solved ⇒ φ = 0 (proven win) or φ = ∞ (disproven ⇒ loss).
        pd.map(|(p, _)| p == 0).unwrap_or(false)
    }
    fn nodes(&self) -> u64 {
        self.tt.nodes()
    }
    fn cap_bytes(&self) -> u64 {
        self.tt.capacity().1
    }
    /// Proof and disproof numbers are df-pn's currency -- report the root's.
    fn stats(&self) -> String {
        let pf = |x: u32| {
            if x == PN_INF {
                "∞".to_string()
            } else {
                x.to_string()
            }
        };
        format!(
            "root proof φ={} disproof δ={} · {}",
            pf(self.root_phi.load(Ordering::Relaxed)),
            pf(self.root_delta.load(Ordering::Relaxed)),
            self.tt.summary(),
        )
    }
}
