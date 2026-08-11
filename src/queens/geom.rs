//! Board geometry: the `Queens` struct, attack masks, symmetry permutations,
//! placement, the D4 canonical key, and PV extraction.

use super::*;
use std::collections::HashSet;

/// An `n×n` Non-Attacking Queens game: board geometry, per-square attack masks,
/// a forcing static move order, and the board's 8 symmetry permutations. This is
/// pure geometry -- the search lives in the [`Solver`] implementations.
pub struct Queens {
    pub n: u32,
    pub(crate) board: Bits,
    pub(crate) attack: Vec<Bits>, // attack[s] = s plus its row/col/diagonals (self-blocking)
    pub(crate) order: Vec<u32>,   // squares by descending attack degree (forcing moves first)
    pub(crate) sym: Vec<Vec<u32>>, // sym[t][s] = image of square s under symmetry t (t=0 identity)
}

/// The 8 symmetries of the square applied to `(row, col)` on an `n×n` board.
#[inline]
fn symmetry(t: usize, r: u32, c: u32, n: u32) -> (u32, u32) {
    let (m1, m2) = (n - 1 - r, n - 1 - c);
    match t {
        0 => (r, c),   // identity
        1 => (c, m1),  // rotate 90°
        2 => (m1, m2), // rotate 180°
        3 => (m2, r),  // rotate 270°
        4 => (r, m2),  // flip horizontally
        5 => (m1, c),  // flip vertically
        6 => (c, r),   // transpose (main diagonal)
        _ => (m2, m1), // anti-transpose
    }
}

impl Queens {
    /// Build the geometry for an `n×n` board (`1 <= n <= MAX_N`).
    pub fn new(n: u32) -> Self {
        assert!(
            (1..=MAX_N).contains(&n),
            "board side must be 1..={MAX_N} (n*n must fit in {} bits)",
            WORDS * 64
        );
        let mut board = Bits::ZERO;
        for s in 0..n * n {
            board.set(s);
        }
        let mut attack = vec![Bits::ZERO; (n * n) as usize];
        for r in 0..n {
            for c in 0..n {
                let mut mask = Bits::ZERO;
                for rr in 0..n {
                    for cc in 0..n {
                        // share a row, column, or either diagonal (includes self)
                        if rr == r
                            || cc == c
                            || rr as i32 - cc as i32 == r as i32 - c as i32
                            || rr + cc == r + c
                        {
                            mask.set(rr * n + cc);
                        }
                    }
                }
                attack[(r * n + c) as usize] = mask;
            }
        }
        // Forcing move order: most-blocking squares first ⇒ winning moves (which
        // tend to slam the board shut) surface early, so the α-β cutoff fires.
        let mut order: Vec<u32> = (0..n * n).collect();
        order.sort_by_key(|&s| std::cmp::Reverse(attack[s as usize].popcount()));
        // Symmetry permutations on square indices.
        let sym: Vec<Vec<u32>> = (0..8)
            .map(|t| {
                (0..n * n)
                    .map(|s| {
                        let (r2, c2) = symmetry(t, s / n, s % n, n);
                        r2 * n + c2
                    })
                    .collect()
            })
            .collect();
        Queens {
            n,
            board,
            attack,
            order,
            sym,
        }
    }

    /// Square index from `(row, col)`, both `0`-indexed.
    #[inline]
    pub fn square(&self, row: u32, col: u32) -> u32 {
        row * self.n + col
    }

    /// Is `sq` available (on the board and not yet blocked)?
    #[inline]
    pub fn is_available(&self, blocked: Bits, sq: u32) -> bool {
        self.board.get(sq) && !blocked.get(sq)
    }

    /// Are there no legal moves left for this blocked mask?
    #[inline]
    pub fn no_moves(&self, blocked: Bits) -> bool {
        self.board.or(blocked) == blocked // board ⊆ blocked ⇒ nothing available
    }

    /// Place a queen on `sq`, returning the new blocked mask.
    #[inline]
    pub fn place(&self, blocked: Bits, sq: u32) -> Bits {
        blocked.or(self.attack[sq as usize])
    }

    /// Does the board have a centre square (odd side)?
    #[inline]
    pub fn is_odd(&self) -> bool {
        self.n % 2 == 1
    }

    /// The centre square -- only odd boards have one.
    #[inline]
    pub fn center(&self) -> Option<u32> {
        self.is_odd()
            .then(|| self.square((self.n - 1) / 2, (self.n - 1) / 2))
    }

    /// The 180° rotation of `sq` (point reflection through the board centre).
    #[inline]
    pub fn mirror(&self, sq: u32) -> u32 {
        self.sym[2][sq as usize]
    }

    /// **#9 free-involution loss certificate.** True if `available` (a canonical
    /// available-mask, as stored in the TT) proves the mover *loses* with no search:
    /// it is **180°-symmetric** (`available == rot180(available)`) **and** no set
    /// square lies on a centre diagonal (`r == c` or `r + c == n-1`). Then the
    /// responder mirrors every move by 180° rotation: a square attacks its own 180°
    /// image *only* when it is on a centre diagonal (shares the row/col only through
    /// the centre, which exists for odd n alone; shares a diagonal exactly on
    /// `r==c`/`r+c==n-1`), so off-diagonal the mirror stays available and the pairing
    /// strategy carries to the end ⇒ second player (responder) wins. Both conditions
    /// are invariant under the 8 board symmetries (rot180 is central in D4; the
    /// symmetries permute the two centre diagonals among themselves), so the test is
    /// exact on the *canonical* key. (Measurement: `count --psym`; lever #9.)
    pub fn is_free_involution_loss(&self, available: Bits) -> bool {
        // 180°-symmetric under sym[2].
        let mut rot = Bits::ZERO;
        available.each(|s| rot.set(self.sym[2][s as usize]));
        if rot != available {
            return false;
        }
        // No set square on either centre diagonal.
        let mut on_diag = false;
        available.each(|s| {
            let (r, c) = (s / self.n, s % self.n);
            on_diag |= r == c || r + c == self.n - 1;
        });
        !on_diag
    }

    /// The first available square in forcing order (for the losing side, or to
    /// drive the symmetry line).
    #[inline]
    pub fn first_available(&self, blocked: Bits) -> Option<u32> {
        self.order
            .iter()
            .copied()
            .find(|&s| self.is_available(blocked, s))
    }

    /// The canonical (lexicographically smallest) image of `mask` under the
    /// board's 8 symmetries.
    pub(crate) fn canon(&self, mask: Bits) -> Bits {
        let mut best = mask;
        for t in 1..8 {
            let perm = &self.sym[t];
            let mut img = Bits::ZERO;
            mask.each(|s| img.set(perm[s as usize]));
            if img < best {
                best = img;
            }
        }
        best
    }

    /// The canonical transposition key for the position with this `blocked` mask.
    ///
    /// We canonicalise the **available** squares (`board & !blocked`), not
    /// `blocked` itself. Available is a pure function of `blocked`, so this merges
    /// the *identical* equivalence classes (same transpositions, same symmetry
    /// folding) -- but for the deep majority of nodes most squares are blocked, so
    /// `available` has far fewer set bits than `blocked` and `canon` does
    /// proportionally less work. Pure speedup, no change to which states share.
    #[inline]
    pub(crate) fn pos_key(&self, blocked: Bits) -> Bits {
        self.canon(self.board.and_not(blocked))
    }
    /// The symmetry-distinct first moves from the empty board: one representative
    /// per orbit of the board's 8 symmetries. Cuts the root branching ~8×.
    pub fn distinct_first_moves(&self) -> Vec<u32> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for &sq in &self.order {
            if self.board.get(sq) && seen.insert(self.pos_key(self.place(Bits::ZERO, sq))) {
                out.push(sq);
            }
        }
        out
    }

    /// An optimal move for the side to move and whether it wins, using `solver`:
    /// a winning move if one exists, else the first available move (all lose).
    /// `None` if no move exists.
    pub fn best_move(&self, blocked: Bits, solver: &dyn Solver) -> Option<(u32, bool)> {
        let mut first = None;
        for &sq in &self.order {
            if !self.is_available(blocked, sq) {
                continue;
            }
            first.get_or_insert(sq);
            if !solver.wins(self, self.place(blocked, sq)) {
                return Some((sq, true)); // a winning move
            }
        }
        first.map(|sq| (sq, false)) // losing: any legal move
    }

    /// An optimal line from the empty board, given the known root verdict
    /// `root_wins` (does the first player win?). Odd boards take the O(1) centre +
    /// mirror line and ignore `root_wins`.
    ///
    /// For even boards the optimal line's value **strictly alternates** down the
    /// plies: a *loss* node (player to move loses) has *every* child winning, so any
    /// move is optimal and the child is a win; a *win* node has a move to a *losing*
    /// child, and that child is a loss for the next mover. So we thread the value
    /// from `root_wins` and **never search a loss ply** -- we take the first legal
    /// move with no search -- while a win ply searches (`best_move`'s α-β cutoff over
    /// the warm TT) for a move to a losing child. This avoids re-confirming the
    /// verdict by re-searching every root subtree single-core (the post-solve PV
    /// grind, backlog #21): for a second-player win the root is a loss, so the whole
    /// 36-subtree root re-search is replaced by one `first_available`.
    pub fn principal_variation(&self, solver: &dyn Solver, root_wins: bool) -> Vec<u32> {
        if self.is_odd() {
            return self.mirror_line();
        }
        let mut blocked = Bits::ZERO;
        let mut line = Vec::new();
        let mut node_wins = root_wins; // value for the player to move at this ply
        loop {
            let next = if node_wins {
                // Win node: a move to a losing child exists. `best_move` returns it
                // first, stopping at the first child the cutoff proves a loss.
                match self.best_move(blocked, solver) {
                    Some((sq, won)) => {
                        debug_assert!(won, "win-node PV ply must have a winning move");
                        Some(sq)
                    }
                    None => None,
                }
            } else {
                // Loss node: every move loses, so the first legal one is optimal --
                // no search. (Exactly the square a loss `best_move` would return.)
                self.first_available(blocked)
            };
            match next {
                Some(sq) => {
                    line.push(sq);
                    blocked = self.place(blocked, sq);
                    node_wins = !node_wins; // value strictly alternates down the line
                }
                None => break,
            }
        }
        line
    }

    /// The first player's winning line on an odd board, with no search: centre,
    /// then mirror each (here arbitrary) reply by the losing side. The mirror is
    /// always legal (see `Solver::first_player_wins`), so this terminates with the
    /// second player stuck and the first player having made the last move.
    pub fn mirror_line(&self) -> Vec<u32> {
        let mut blocked = Bits::ZERO;
        let mut line = Vec::new();
        let c = self.center().expect("odd board has a centre");
        line.push(c);
        blocked = self.place(blocked, c);
        while let Some(s) = self.first_available(blocked) {
            line.push(s); // losing side: any legal reply
            blocked = self.place(blocked, s);
            let m = self.mirror(s); // first player's pairing response
            debug_assert!(self.is_available(blocked, m), "mirror must stay legal");
            line.push(m);
            blocked = self.place(blocked, m);
        }
        line
    }
}
