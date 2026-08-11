//! The adversarial **Non-Attacking Queens** game (Noon & Van Brummelen, 2006).
//!
//! Two players alternately place a queen on an `n×n` board so that no two queens
//! attack each other (no shared row, column, or diagonal). A player who cannot
//! move loses -- equivalently, the player who places the queen that leaves every
//! remaining square attacked wins. It is a *combinatorial* game: perfect
//! information, no chance, normal play (last to move wins). Formally it is
//! **Node Kayles on the n-queens graph** (deciding the winner is PSPACE-complete,
//! Schaefer 1978), and its Sprague-Grundy value is OEIS A344227.
//!
//! The game is **impartial** -- a queen is colourless, so the legal moves depend
//! only on the position, captured entirely by the **blocked mask** (squares
//! occupied or attacked): placing a queen on `s` always adds the same `attack(s)`,
//! so move orders reaching the same mask are identical for all future play.
//!
//! **Odd boards need no search.** The first player wins by a pairing strategy:
//! take the centre, then answer every reply with its 180° rotation (see
//! `Solver::first_player_wins`). Only *even* boards are searched.
//!
//! ## Solver lineage (mirrors the Othello engine ladder)
//!
//! [`Queens`] is pure geometry; the search is a ladder of [`Solver`]s, each step
//! adding one idea, all computing the *same* win/loss so the simpler ones are
//! kept as ground truth (cross-checked in the tests):
//!
//! | solver       | adds                                                        |
//! |--------------|-------------------------------------------------------------|
//! | [`Naive`]    | plain negamax win/loss with an α-β cutoff, no memo (truth)  |
//! | [`Memo`]     | a fixed-size transposition table keyed on the raw mask      |
//! | [`Symmetry`] | + dihedral (8-fold) canonical keys, merging symmetric states|
//! | [`Parallel`] | + rayon root parallelism (Young-Brothers-Wait) + odd O(1)   |
//!
//! Bit `r*n + c` is the square at row `r`, column `c` (`0`-indexed). The bitset is
//! `WORDS` × 64 bits, so boards up to `16×16` (256 bits) fit.

/// 64-bit words backing the board bitset. 4 words = 256 bits ⇒ up to `n = 16`.
const WORDS: usize = 4;
/// Largest board side the bitset can hold (`n*n <= WORDS*64`).
pub const MAX_N: u32 = 16;
/// Largest vertex count of an available-graph (one per square) -- sizes the
/// preallocated graph-key scratch buffers.
const MAXV: usize = (MAX_N * MAX_N) as usize;

mod bits;
mod count;
mod dense;
mod geom;
mod graph;
mod solver;
mod store;
mod tt;

pub use bits::Bits;
pub use count::{CountReport, Hll};
pub use geom::Queens;
pub use graph::ModuleStats;
pub use solver::{
    make_solver, run_ranklab, BranchingStats, Burr, Fused, Incremental, IsoBurr, IsoFlat, Naive,
    Nimber, NimberSum, Parallel, Pn, Solver, StealReport, Tt, SOLVER_NAMES,
};
pub use store::BurrStore;
pub use tt::{archive_key_of, for_each_image_entry, QueensTt, TtHeader};

// Internal cross-module items (not part of the crate-facing API): shared by the
// sibling modules and the tests via `use super::*` / `use crate::queens::*`.
pub(crate) use bits::single;
pub(crate) use count::Counter;
pub(crate) use dense::{warm_wide, DenseW8};
pub(crate) use graph::mix64;
pub(crate) use tt::PnTt;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queens::graph::{tiny_comp_key, IsoScratch, TINY_MAX};
    use crate::queens::tt::{Slot, TT_HEADER_LEN};
    use std::collections::HashMap;
    use std::io;

    /// On tiny boards the first move already attacks the whole board, so the
    /// first player wins in a single ply.
    #[test]
    fn small_boards_first_player_wins_in_one() {
        for n in 1..=3 {
            let q = Queens::new(n);
            let s = Parallel::new(14);
            assert!(s.first_player_wins(&q), "n={n}: first player should win");
            assert_eq!(
                q.principal_variation(&s, true).len(),
                1,
                "n={n}: one placement clears it"
            );
        }
    }

    /// A queen attacks its whole row, column, and both diagonals.
    #[test]
    fn attack_mask_covers_lines() {
        let q = Queens::new(8);
        let d4 = q.square(3, 3);
        let a = q.attack[d4 as usize];
        assert!(a.get(q.square(3, 7)), "same row");
        assert!(a.get(q.square(0, 3)), "same column");
        assert!(a.get(q.square(0, 0)), "main diagonal");
        assert!(a.get(q.square(6, 0)), "anti-diagonal");
        assert!(!a.get(q.square(1, 0)), "a knight's move away is safe");
    }

    /// The whole solver lineage computes the same win/loss: the memo, symmetry,
    /// and parallel solvers all agree with the memo-less ground-truth `Naive` on
    /// every board small enough to brute-force.
    #[test]
    fn solver_lineage_agrees() {
        for n in 1..=9 {
            let q = Queens::new(n);
            let truth = Naive::new().first_player_wins(&q);
            assert_eq!(
                Tt::new(16, false).first_player_wins(&q),
                truth,
                "memo n={n}"
            );
            assert_eq!(
                Tt::new(16, true).first_player_wins(&q),
                truth,
                "symmetry n={n}"
            );
            assert_eq!(
                Parallel::new(16).first_player_wins(&q),
                truth,
                "parallel n={n}"
            );
            assert_eq!(
                Incremental::new(16).first_player_wins(&q),
                truth,
                "incremental n={n}"
            );
            assert_eq!(Burr::new(16).first_player_wins(&q), truth, "burr n={n}");
            assert_eq!(
                IsoBurr::new(16).first_player_wins(&q),
                truth,
                "iso-burr n={n}"
            );
            assert_eq!(Fused::new(16).first_player_wins(&q), truth, "fused n={n}");
            assert_eq!(
                IsoFlat::new(16).first_player_wins(&q),
                truth,
                "iso-flat n={n}"
            );
            assert_eq!(
                IsoFlat::new_window(16).first_player_wins(&q),
                truth,
                "iso-window n={n}"
            );
            // iso-dense's W9 pc==9 resolver is exercised on n=8/9 (the available graph
            // descends through 9 vertices), so the lineage loop is its parity gate too.
            assert_eq!(
                IsoFlat::new_dense(16).first_player_wins(&q),
                truth,
                "iso-dense n={n}"
            );
            assert_eq!(
                Nimber::new(16).first_player_wins(&q),
                truth,
                "nimber!=0 n={n}"
            );
        }
        // df-pn is correct but hits the transposition (graph-history) pathology
        // on this game, so it is only practical for tiny boards -- validate those.
        for n in 1..=6 {
            let q = Queens::new(n);
            assert_eq!(
                Pn::new(16).first_player_wins(&q),
                Naive::new().first_player_wins(&q),
                "pn n={n}"
            );
        }
    }

    #[test]
    fn iso_window_agrees_on_small_even_boards() {
        for n in [8u32, 10] {
            let q = Queens::new(n);
            let truth = Naive::new().first_player_wins(&q);
            assert_eq!(
                IsoFlat::new_window(16).first_player_wins(&q),
                truth,
                "iso-window n={n}"
            );
            assert_eq!(
                IsoFlat::new_dense(16).first_player_wins(&q),
                truth,
                "iso-dense n={n}"
            );
        }
    }

    /// ABDADA in-flight deferral preserves the verdict. The marker write + tri-state probe +
    /// two-pass deferral are timing-dependent (which children get deferred varies per run), but
    /// the verdict must be invariant — every fallback degrades to a plain re-expansion and only
    /// the completing put records a (deterministic) value. n=12 runs through the real parallel
    /// solver, so concurrent transposition collisions actually fire the deferral path (a single
    /// worker never re-probes its own on-stack markers, since available-popcount strictly shrinks).
    #[test]
    fn abdada_agrees_on_small_even_boards() {
        for n in [8u32, 10, 12] {
            let q = Queens::new(n);
            let truth = Naive::new().first_player_wins(&q);
            assert_eq!(
                IsoFlat::new_window(18).with_abdada().first_player_wins(&q),
                truth,
                "iso-window+abdada n={n}"
            );
            assert_eq!(
                IsoFlat::new_dense(18).with_abdada().first_player_wins(&q),
                truth,
                "iso-dense+abdada n={n}"
            );
        }
    }

    /// Frontier work-stealing preserves the verdict. Publishing a frame's children as rayon scope
    /// tasks (and resolving them back through the shared TT via the in-flight markers) is heavily
    /// timing-dependent — which children are stolen vs expanded locally varies every run — but the
    /// verdict is invariant by construction (a stolen subtree writes the same deterministic value,
    /// and an un-stolen one falls back to local expansion). n=12 drives enough parallel depth to
    /// actually publish + steal.
    #[test]
    fn steal_agrees_on_small_even_boards() {
        for n in [8u32, 10, 12] {
            let q = Queens::new(n);
            let truth = Naive::new().first_player_wins(&q);
            assert_eq!(
                IsoFlat::new_window(18).with_steal().first_player_wins(&q),
                truth,
                "iso-window+steal n={n}"
            );
            assert_eq!(
                IsoFlat::new_dense(18).with_steal().first_player_wins(&q),
                truth,
                "iso-dense+steal n={n}"
            );
        }
    }

    /// The `burr` LSM store stays correct under *frequent* freezes: a tiny freeze
    /// threshold forces the memtable to freeze into BuRR segments and clear many
    /// times over a single search, so the verdict only survives if the cascade
    /// (memtable → segments) and the archive-key round-trip are sound -- including
    /// the false-positive guard (a wrong accept would flip a verdict). The even
    /// boards 8/10/12 push enough nodes through the store to trigger many freezes.
    #[test]
    fn burr_lsm_survives_frequent_freezes() {
        for n in [8u32, 10, 12] {
            let q = Queens::new(n);
            let truth = Naive::new().first_player_wins(&q);
            // Small threshold ⇒ many freeze→segment→clear cycles over the search.
            let burr = Burr::with_freeze_at(20, 50_000);
            assert_eq!(
                burr.first_player_wins(&q),
                truth,
                "burr (forced freezes) n={n}"
            );
        }
    }

    /// Odd boards are first-player wins by the centre + 180°-mirror strategy --
    /// proven O(1), and the produced line is legal, complete, and odd-length.
    #[test]
    fn odd_boards_win_by_the_mirror_strategy() {
        for n in (1..=15).step_by(2) {
            let q = Queens::new(n);
            let s = Parallel::new(14);
            assert!(s.first_player_wins(&q), "n={n}: odd ⇒ first wins");
            let pv = q.principal_variation(&s, true); // mirror_line (root_wins ignored when odd)
            assert_eq!(pv.len() % 2, 1, "n={n}: first player makes the last move");
            let mut blocked = Bits::ZERO;
            for &sq in &pv {
                // is_available also guarantees no square is placed twice.
                assert!(
                    q.is_available(blocked, sq),
                    "n={n}: mirror move must be legal"
                );
                blocked = q.place(blocked, sq);
            }
            assert!(q.no_moves(blocked), "n={n}: board fully blocked at the end");
        }
        // The O(1) verdict agrees with full search on the small odd boards.
        for n in [1u32, 3, 5, 7, 9] {
            assert!(
                Naive::new().first_player_wins(&Queens::new(n)),
                "n={n}: search agrees"
            );
        }
    }

    /// The PV is a legal, complete optimal line, and the winner is whoever makes
    /// the last move (odd-length line ⇒ first player wins).
    #[test]
    fn pv_is_consistent_with_the_winner() {
        for n in 1..=8 {
            let q = Queens::new(n);
            let s = Tt::new(16, true);
            let first_wins = s.first_player_wins(&q);
            let pv = q.principal_variation(&s, first_wins);
            assert_eq!(
                first_wins,
                pv.len() % 2 == 1,
                "n={n}: winner makes the last move"
            );
            let mut blocked = Bits::ZERO;
            for &sq in &pv {
                assert!(q.is_available(blocked, sq), "n={n}: PV move must be legal");
                blocked = q.place(blocked, sq);
            }
            assert!(q.no_moves(blocked), "n={n}: board fully blocked at the end");
        }
    }

    /// A dumped TT image reloads into the same verdict and resumes *warm*: the
    /// fresh solver re-confirms the result almost entirely from cached hits, and a
    /// header for the wrong board is a hard error, not a silent mis-load.
    #[test]
    fn tt_image_round_trips_and_warms() {
        let q = Queens::new(10);
        // Cold solve, populating the table.
        let cold = Tt::new(16, true);
        let v1 = cold.first_player_wins(&q);
        cold.drain(); // flush thread-local node tally before reading nodes()
        let cold_nodes = cold.nodes();
        assert!(cold_nodes > 1000, "n=10 searches a real number of nodes");

        // Dump the warm table to an in-memory image and reload it.
        let mut img = Vec::new();
        cold.tt().unwrap().dump_image(&mut img, q.n as u8).unwrap();
        let reloaded = QueensTt::load_image(&mut img.as_slice(), q.n as u8).unwrap();

        // A fresh solver around the reloaded table re-confirms the verdict from the
        // root cache hit -- the warm-resume property (the TT *is* the progress).
        let warm = Tt::from_tt(reloaded, true);
        let v2 = warm.first_player_wins(&q);
        assert_eq!(v1, v2, "reloaded table yields the same verdict");
        // The reloaded counter carries the snapshot's node total (so a resume's progress
        // reflects the whole search), and the warm re-confirm adds almost nothing on top.
        assert!(
            warm.nodes() >= cold_nodes,
            "resume restores the snapshot node count: {} ≥ {cold_nodes}",
            warm.nodes(),
        );
        let new_nodes = warm.nodes() - cold_nodes;
        assert!(
            new_nodes < cold_nodes / 100,
            "warm resume re-searches almost nothing: {new_nodes} new vs cold {cold_nodes}",
        );

        // The image is rejected for a different board, not silently mis-keyed.
        let reject = |r: io::Result<QueensTt>, what: &str| match r {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{what}"),
            Ok(_) => panic!("{what}: must be rejected"),
        };
        reject(
            QueensTt::load_image(&mut img.as_slice(), q.n as u8 + 2),
            "wrong-n image",
        );
        // ...and rubbish bytes fail the magic check.
        reject(
            QueensTt::load_image(&mut [0u8; TT_HEADER_LEN].as_slice(), q.n as u8),
            "bad magic",
        );
    }

    /// The #9 free-involution loss certificate fires exactly on 180°-symmetric,
    /// off-centre-diagonal masks, and (cross-checked at scale by `count --psym`,
    /// which finds zero false fires) only on genuine losses.
    #[test]
    fn free_involution_certificate_conditions() {
        let q = Queens::new(4);
        let m = |squares: &[(u32, u32)]| {
            let mut b = Bits::ZERO;
            for &(r, c) in squares {
                b.set(q.square(r, c));
            }
            b
        };
        // 180°-symmetric (each square's rot180 partner present) and off both centre
        // diagonals (r≠c and r+c≠3) ⇒ fires.
        assert!(
            q.is_free_involution_loss(m(&[(0, 1), (3, 2)])),
            "symmetric + off-diagonal must fire"
        );
        // Symmetric but on the main diagonal (0,0)↔(3,3) ⇒ a square attacks its own
        // image, mirror strategy breaks ⇒ must NOT fire.
        assert!(
            !q.is_free_involution_loss(m(&[(0, 0), (3, 3)])),
            "on-diagonal must not fire"
        );
        // Not 180°-symmetric ⇒ must not fire.
        assert!(
            !q.is_free_involution_loss(m(&[(0, 1)])),
            "asymmetric must not fire"
        );
        // The empty board is symmetric but every diagonal square is present ⇒ off by
        // the diagonal condition (the certificate must not call the start a loss).
        assert!(
            !q.is_free_involution_loss(q.board),
            "full board must not fire"
        );
    }

    /// The Sprague-Grundy nimbers match OEIS A344227, and `nimber != 0` agrees
    /// with the win/loss ground truth (`Naive`).
    #[test]
    fn nimbers_match_oeis_a344227() {
        // A344227 for n = 0..=13; we only solve n >= 1.
        const A344227: [u8; 14] = [0, 1, 1, 2, 1, 3, 1, 2, 3, 1, 0, 1, 0, 1];
        for n in 1..=9u32 {
            let q = Queens::new(n);
            let g = Nimber::new(16).nimber(&q);
            assert_eq!(
                g, A344227[n as usize],
                "n={n}: nimber must match OEIS A344227"
            );
            assert_eq!(
                g != 0,
                Naive::new().first_player_wins(&q),
                "n={n}: nimber!=0 must agree with win/loss"
            );
        }
    }

    /// The heap-sum nimber engine (`NimberSum`) matches OEIS A344227 and the independent
    /// full-mex reference (`Nimber`): two disagreeing-by-construction engines (α-β heap-sum
    /// rounds + Grundy dense leaves vs whole-DAG mex over a canon TT) must agree exactly.
    #[test]
    fn nimber_sum_matches_full_mex_and_oeis() {
        const A344227: [u8; 14] = [0, 1, 1, 2, 1, 3, 1, 2, 3, 1, 0, 1, 0, 1];
        for n in 1..=11u32 {
            let q = Queens::new(n);
            let g = NimberSum::new(22)
                .nimber(&q, 8)
                .expect("nimber within max_k");
            assert_eq!(g, A344227[n as usize], "n={n}: sum engine vs OEIS A344227");
            if n <= 8 {
                assert_eq!(
                    g,
                    Nimber::new(16).nimber(&q),
                    "n={n}: sum engine vs full-mex reference"
                );
            }
        }
    }

    /// The HyperLogLog estimates a known cardinality within its error budget,
    /// and folding a key in repeatedly does not inflate the estimate (dedup).
    #[test]
    fn hll_estimates_a_known_cardinality() {
        let hll = Hll::new(14); // 16384 registers ⇒ ~0.8% standard error
        let truth = 200_000u64;
        for i in 0..truth {
            let mut b = Bits::ZERO;
            b.0[0] = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            b.0[1] = i; // distinct i ⇒ distinct key
            hll.add(b);
            hll.add(b); // re-fold: must be idempotent
        }
        let est = hll.estimate();
        let rel = (est - truth as f64).abs() / truth as f64;
        assert!(
            rel < 0.03,
            "HLL estimate {est:.0} off by {:.2}% from {truth} (>3%)",
            rel * 100.0
        );
    }

    /// Enabling the counting hook must not change the search verdict, and the
    /// HyperLogLog estimate must track the exact distinct-position count.
    #[test]
    fn counting_preserves_verdict_and_tracks_exact() {
        for n in [4u32, 6, 8] {
            let q = Queens::new(n);
            let truth = Naive::new().first_player_wins(&q);
            let s = Tt::new_counting(16, true, 14, true);
            assert_eq!(
                s.first_player_wins(&q),
                truth,
                "n={n}: counting must not change the verdict"
            );
            s.drain(); // flush thread-local node/HLL tallies before reading the report
            let rep = s.report().expect("counting enabled");
            let exact = rep.exact.expect("exact set kept");
            assert!(exact > 0, "n={n}: the search visited some positions");
            let rel = (rep.estimate - exact as f64).abs() / exact as f64;
            assert!(
                rel < 0.10,
                "n={n}: HLL {:.0} vs exact {exact} off by {:.1}%",
                rep.estimate,
                rel * 100.0
            );
        }
    }

    /// Canonicalisation is symmetry-invariant: a position and its rotation/
    /// reflection share a memo key (and hence a value).
    #[test]
    fn symmetric_positions_canonicalise_together() {
        let q = Queens::new(8);
        let a = q.place(Bits::ZERO, q.square(3, 3)); // a queen on d4
        for t in 0..8 {
            let mut img = Bits::ZERO;
            a.each(|s| img.set(q.sym[t][s as usize]));
            assert_eq!(
                q.canon(a),
                q.canon(img),
                "symmetry {t} must canonicalise the same"
            );
        }
    }

    /// The Chunk-2 compact slot stays 8 bytes, and a stored value round-trips
    /// through the fingerprint: `put` then `get` returns it, and a key never
    /// inserted misses. Guards against accidental slot bloat or a broken
    /// fingerprint (a real false hit at this tiny load is a ~`2^-55` event).
    #[test]
    fn fingerprint_slot_is_compact_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<Slot>(),
            8,
            "compact slot must stay 8 bytes"
        );
        let tt = QueensTt::new(16); // 65_536 slots: a handful of keys ⇒ no eviction
        let q = Queens::new(12);
        // A monotonically shrinking chain of legal placements: each `blocked` has
        // strictly fewer available squares, so the canonical keys are all distinct.
        let mut stored = Vec::new();
        let mut blocked = Bits::ZERO;
        for (i, &sq) in q.order.iter().enumerate() {
            if q.is_available(blocked, sq) {
                blocked = q.place(blocked, sq);
                let (key, val) = (q.pos_key(blocked), (i % 17) as u8); // nimber-sized
                tt.put(key, val);
                stored.push((key, val));
            }
        }
        assert!(
            stored.len() >= 4,
            "the chain should store several positions"
        );
        for &(key, val) in &stored {
            assert_eq!(tt.get(key), Some(val), "stored value must round-trip");
        }
        assert_eq!(
            tt.get(q.pos_key(Bits::ZERO)),
            None,
            "a key never inserted must miss"
        );
    }

    /// The tiny-component shortcut (#18) must induce exactly the same isomorphism
    /// partition as the full WL+IR canon on every small connected component drawn from
    /// a real queen graph: isomorphic components share a key under both keys, and the
    /// two never disagree about whether two components are isomorphic. This is what
    /// keeps the graph-key merge -- and so the distinct working set -- unchanged.
    #[test]
    fn tiny_component_key_matches_full_canon() {
        let q = Queens::new(6);
        let sq: Vec<u32> = (0..q.n * q.n).collect();

        // Enumerate every connected induced subgraph of size 1..=TINY_MAX.
        let mut comps: Vec<(Bits, usize)> = Vec::new();
        for &a in &sq {
            comps.push((single(a), 1));
        }
        for i in 0..sq.len() {
            for j in i + 1..sq.len() {
                let c = single(sq[i]).or(single(sq[j]));
                if q.component(sq[i], c).popcount() == 2 {
                    comps.push((c, 2));
                }
            }
        }
        for i in 0..sq.len() {
            for j in i + 1..sq.len() {
                for l in j + 1..sq.len() {
                    let c = single(sq[i]).or(single(sq[j])).or(single(sq[l]));
                    if q.component(sq[i], c).popcount() == 3 {
                        comps.push((c, 3));
                    }
                }
            }
        }
        for i in 0..sq.len() {
            for j in i + 1..sq.len() {
                for l in j + 1..sq.len() {
                    for m in l + 1..sq.len() {
                        let c = single(sq[i])
                            .or(single(sq[j]))
                            .or(single(sq[l]))
                            .or(single(sq[m]));
                        if q.component(sq[i], c).popcount() == 4 {
                            comps.push((c, 4));
                        }
                    }
                }
            }
        }

        // For the two keys to define the same partition, tiny->full and full->tiny must
        // both be single-valued (a bijection between the value sets actually seen).
        let mut scratch = *IsoScratch::new();
        let mut tiny_to_full: HashMap<u64, u64> = HashMap::new();
        let mut full_to_tiny: HashMap<u64, u64> = HashMap::new();
        let mut counts = [0usize; TINY_MAX + 1];
        for (comp, k) in comps {
            let mut kk = 0usize;
            comp.each(|v| {
                scratch.verts[kk] = v as u8;
                kk += 1;
            });
            assert_eq!(kk, k);
            let tiny = tiny_comp_key(&q.attack, comp, k, &scratch.verts);
            let full = q.comp_canon_full(comp, k, &mut scratch);
            if let Some(&f) = tiny_to_full.get(&tiny) {
                assert_eq!(f, full, "tiny key collides two full classes (over-merge)");
            } else {
                tiny_to_full.insert(tiny, full);
            }
            if let Some(&t) = full_to_tiny.get(&full) {
                assert_eq!(t, tiny, "full key splits into two tiny keys (over-split)");
            } else {
                full_to_tiny.insert(full, tiny);
            }
            counts[k] += 1;
        }
        for (k, &c) in counts.iter().enumerate().skip(1) {
            assert!(c > 0, "corpus saw no size-{k} components");
        }
    }

    #[test]
    fn tiny_iso_table_matches_fast_partition() {
        let q = Queens::new(12);
        let masks: Vec<Bits> = q
            .iso_corpus(20_000)
            .into_iter()
            .filter(|m| m.popcount() <= 7)
            .collect();
        assert!(!masks.is_empty(), "corpus has no tiny available graphs");

        let mut tiny_to_fast: HashMap<u64, u64> = HashMap::new();
        let mut fast_to_tiny: HashMap<u64, u64> = HashMap::new();
        for m in masks {
            let tiny = q.iso_key_tiny_table(m);
            let fast = q.iso_key_fast(m);
            if let Some(&f) = tiny_to_fast.get(&tiny) {
                assert_eq!(f, fast, "tiny table over-merged two fast iso classes");
            } else {
                tiny_to_fast.insert(tiny, fast);
            }
            if let Some(&t) = fast_to_tiny.get(&fast) {
                assert_eq!(t, tiny, "tiny table split one fast iso class");
            } else {
                fast_to_tiny.insert(fast, tiny);
            }
        }
    }
}
