//! Offline move-ordering lab (`queens ranklab`). Consumes a `QUEENS_HITKEY` dump — sampled deep-tail
//! node states (canonical key + available-square set + pc) — and scores candidate child orderings
//! against the production degree ordering WITHOUT a live search.
//!
//! The metric is `ordering_loss` = E − E_perfect, the *avoidable* child-examination waste the
//! `QUEENS_RANK` report surfaces. It collapses to the 0-based rank of the first losing child: a
//! perfect order cuts every winning OR-node at rank 0; a LOSS (no-cut) node is unavoidable. Because a
//! child's win/loss value is order-invariant, the rank ANY candidate ordering achieves is exactly the
//! index of its earliest losing child — so the A/B is exact offline, no re-search per candidate rule.
//!
//! Labeling (which children are losing) happens HERE, in an isolated solver context (its own TT),
//! never in the live run — inline labeling would pollute the production TT and skew the sample (the
//! constraint baked into this design). We reuse the real iso-dense kernel via [`Solver::wins`] so the
//! labels — and thus the baseline rank — are exact, and a shared TT amortises the labeling cost across
//! the overlapping sampled subtrees.

use super::{make_solver, Solver};
use crate::queens::{Bits, Queens};
use rayon::prelude::*;
use std::path::Path;

/// Candidate orderings scored per node. The metric is RECURSE-WEIGHTED: count only the recurse
/// children (child_pc > RECURSE_MIN) examined before the first losing child. getK leaves
/// (child_pc ≤ RECURSE_MIN) resolve instantly and never become search nodes, so reordering them saves
/// no wall — the live A/B confirmed pc18 reorder = nodes flat. Only avoided RECURSE expansions count.
/// The candidates reorder ONLY the recurse suffix (getK children keep degree order), the cheap form
/// that could survive the hot loop.
const NCAND: usize = 6;
const CAND_NAMES: [&str; NCAND] = [
    "degree(cur)", // 0 baseline (live degree order) — the recurse_rank we're trying to cut
    "oracle",      // 1 a losing child first ⇒ recurse_rank 0 (the 100% ceiling)
    "rec-sumgc↓",  // 2 recurse suffix by total reply degree DESC (opponent-stuck-first)
    "rec-sumgc↑",  // 3 recurse suffix by total reply degree ASC
    "rec-mingc↓",  // 4 recurse suffix by min reply degree DESC
    "rec-mingc↑",  // 5 recurse suffix by min reply degree ASC
];

/// getK ceiling: a child with child_pc ≤ this resolves instantly via the dense getK evaluator (no
/// recursion, never a search node); child_pc > this is a recurse child (a subtree expansion). Mirrors
/// the solver default `recurse_min = max(QUEENS_DENSE_K=17, block_k=8, iso_max_avail=7) = 17`.
const RECURSE_MIN: u32 = 17;

/// One scored node. `n_recurse` = recurse children (the reorderable population). `rrank[c]` = number of
/// recurse children examined before the first losing child under candidate `c` (the avoidable recurse
/// expansions). `rrank[0]` is the live degree order.
struct NodeScore {
    pc: usize,
    cut: bool,
    n_recurse: u32,
    rrank: [u32; NCAND],
}

/// One-ply reply-degree stats of a child `h` (the opponent's reply menu): for each opponent move `v`
/// in `h`, the grandchild degree `popcount(h \ attack[v])`. Returns `(min, sum, has_instant_win)`.
/// Cheap — O(child_pc) bitmask ops, no allocation/BFS/canonicalization (the cost tier that can survive
/// the hot loop). The hypothesis: a *losing* child (P-position for the opponent) leaves the opponent no
/// forcing escape ⇒ high reply degrees / no instant win.
fn reply_stats(h: Bits, attack: &[Bits]) -> (u32, u32, bool) {
    if h == Bits::ZERO {
        return (0, 0, true); // empty child = the move already won; nothing to reply with
    }
    let (mut mn, mut sum, mut iw) = (u32::MAX, 0u32, false);
    h.each(|v| {
        let gd = h.and_not(attack[v as usize]).popcount();
        mn = mn.min(gd);
        sum += gd;
        iw |= gd == 0;
    });
    (mn, sum, iw)
}

/// Label every child of `avail` and score each candidate ordering by the rank of its first losing
/// child. A child is *losing* (a winning move for the mover) iff the resulting position is a LOSS for
/// the opponent: `!solver.wins(child)`. `wins(q, blocked)` evaluates the node whose available set is
/// `board \ blocked`, so `blocked = board \ child`.
fn score_node(q: &Queens, solver: &dyn Solver, avail: Bits) -> NodeScore {
    let pc = avail.popcount() as usize;
    // Available moves in q.order order (the live baseline's stable tie-break).
    let moves: Vec<u32> = q
        .order
        .iter()
        .copied()
        .filter(|&sq| avail.get(sq))
        .collect();
    let m = moves.len();
    // Each move's child (placing removes sq + its attacks), and its degree = child popcount. The
    // degree matches `sort_moves_by_degree` exactly (build_att's identity image att[sq][0] == attack).
    let children: Vec<Bits> = moves
        .iter()
        .map(|&sq| avail.and_not(q.attack[sq as usize]))
        .collect();
    let deg: Vec<u32> = children.iter().map(|c| c.popcount()).collect();
    // One-ply reply-degree stats per child (the opponent's reply menu from that child).
    let reply: Vec<(u32, u32, bool)> = children
        .iter()
        .map(|&c| reply_stats(c, &q.attack))
        .collect();
    let min_gc: Vec<u32> = reply.iter().map(|r| r.0).collect();
    let sum_gc: Vec<u32> = reply.iter().map(|r| r.1).collect();
    // Exact label per move via the production kernel (verdict is order-invariant; the shared TT
    // amortises across nodes). An empty child (move fills the board) is a loss for the opponent ⇒
    // `wins` returns false ⇒ losing = true, the instant-win move.
    let losing: Vec<bool> = children
        .iter()
        .map(|&child| !solver.wins(q, q.board.and_not(child)))
        .collect();
    let n_recurse = deg.iter().filter(|&&d| d > RECURSE_MIN).count() as u32;
    let mut ns = NodeScore {
        pc,
        cut: losing.iter().any(|&b| b),
        n_recurse,
        rrank: [0; NCAND],
    };
    if !ns.cut {
        return ns; // nocut — contributes zero ordering loss (a mandatory full scan)
    }
    // recurse_rank under an index order = recurse children (the costly subtree expansions) examined
    // before the first losing child. getK leaves before the cut are free, so they don't count.
    let rrank_of = |order: &[usize]| -> u32 {
        let mut r = 0u32;
        for &i in order {
            if losing[i] {
                break;
            }
            if deg[i] > RECURSE_MIN {
                r += 1;
            }
        }
        r
    };
    // Recurse-suffix-only reorder: getK children (deg ≤ MIN) keep degree order and sort first (they
    // already do — lower degree); recurse children (deg > MIN) sort by the candidate's reply key.
    // Stable ⇒ q.order is the final tie-break. `reckey(i)` is only consulted for recurse children.
    let order_rec = |reckey: &dyn Fn(usize) -> u32| -> Vec<usize> {
        let mut v: Vec<usize> = (0..m).collect();
        v.sort_by_key(|&i| {
            if deg[i] > RECURSE_MIN {
                (1u32, reckey(i), deg[i])
            } else {
                (0, deg[i], 0)
            }
        });
        v
    };
    ns.rrank[0] = rrank_of(&order_rec(&|i| deg[i])); // recurse suffix in degree order = live baseline
    ns.rrank[1] = 0; // oracle: a losing child first ⇒ no recurse child examined before the cut
    ns.rrank[2] = rrank_of(&order_rec(&|i| u32::MAX - sum_gc[i])); // rec-sumgc↓
    ns.rrank[3] = rrank_of(&order_rec(&|i| sum_gc[i])); // rec-sumgc↑
    ns.rrank[4] = rrank_of(&order_rec(&|i| u32::MAX - min_gc[i])); // rec-mingc↓
    ns.rrank[5] = rrank_of(&order_rec(&|i| min_gc[i])); // rec-mingc↑
    ns
}

/// Per-pc (and grand) accumulator over the scored nodes (recurse-weighted).
#[derive(Default, Clone)]
struct Agg {
    nodes: u64,              // sampled nodes at this pc
    nocut: u64,              // excluded (no losing child — unavoidable full scan)
    nrec_sum: u64,           // Σ n_recurse over cut nodes (→ avg recurse children / node)
    rec_cut: u64, // cut nodes whose baseline recurse_rank > 0 (cut needs ≥1 recurse child)
    sum_rrank: [u64; NCAND], // Σ recurse_rank over cut nodes, per candidate (rrank[0] = baseline)
}

/// Run the offline lab over a `QUEENS_HITKEY` dump at `dump_path`. Scores the giant-shoulder band
/// `pc_lo..=pc_hi`, stride-subsampled to `cap_per_pc` nodes per pc, and prints the captured fraction
/// of avoidable `ordering_loss` for each candidate ordering. `QUEENS_TT_BITS` sizes the labeling TT.
pub fn run_ranklab(
    dump_path: &Path,
    cap_per_pc: usize,
    pc_lo: usize,
    pc_hi: usize,
) -> std::io::Result<()> {
    let bytes = std::fs::read(dump_path)?;
    // Header: "QHK1" + n(u32 LE) + count(u64 LE) = 16 bytes; records are 68 bytes
    // (key 32 + avail 32 + pc 2 + hit 1 + pad 1) — see `IsoFlat::write_hitkey_file`.
    const REC: usize = 68;
    if bytes.len() < 16 || &bytes[0..4] != b"QHK1" {
        eprintln!(
            "ranklab: {} is not a QHK1 (QUEENS_HITKEY) dump",
            dump_path.display()
        );
        std::process::exit(1);
    }
    let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let body = &bytes[16..];
    let nrec = body.len() / REC;

    let read_avail_pc_hit = |i: usize| -> (Bits, usize, bool) {
        let o = i * REC + 32; // skip the 32-byte canonical key
        let mut av = [0u64; 4];
        for (w, slot) in av.iter_mut().enumerate() {
            *slot = u64::from_le_bytes(body[o + w * 8..o + w * 8 + 8].try_into().unwrap());
        }
        let pc = u16::from_le_bytes(body[i * REC + 64..i * REC + 66].try_into().unwrap()) as usize;
        let hit = body[i * REC + 66] != 0;
        (Bits(av), pc, hit)
    };

    // Eligible = an expanding node (miss) whose pc is in band. First pass: per-pc totals.
    let mut total: Vec<usize> = vec![0; pc_hi + 1];
    for i in 0..nrec {
        let (_, pc, hit) = read_avail_pc_hit(i);
        if !hit && (pc_lo..=pc_hi).contains(&pc) {
            total[pc] += 1;
        }
    }
    // Second pass: stride-subsample up to cap_per_pc per pc (spread across the dump, not first-N).
    let stride: Vec<usize> = total
        .iter()
        .map(|&t| {
            if cap_per_pc == 0 || t <= cap_per_pc {
                1
            } else {
                t / cap_per_pc
            }
        })
        .collect();
    let mut seen = vec![0usize; pc_hi + 1];
    let mut kept: Vec<Bits> = Vec::new();
    for i in 0..nrec {
        let (avail, pc, hit) = read_avail_pc_hit(i);
        if hit || !(pc_lo..=pc_hi).contains(&pc) {
            continue;
        }
        let s = seen[pc];
        seen[pc] += 1;
        if s.is_multiple_of(stride[pc]) && (cap_per_pc == 0 || (s / stride[pc]) < cap_per_pc) {
            kept.push(avail);
        }
    }
    if kept.is_empty() {
        eprintln!(
            "ranklab: no eligible records in pc {pc_lo}..={pc_hi} (dump has {nrec} records, n={n})"
        );
        return Ok(());
    }

    let bits: u32 = std::env::var("QUEENS_TT_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(28);
    let q = Queens::new(n);
    let solver = make_solver("iso-dense", bits).expect("iso-dense solver");
    let solver = solver.as_ref();

    println!(
        "ranklab — {} · n={n} · scoring {} sampled nodes (pc {pc_lo}..={pc_hi}, cap {cap_per_pc}/pc) · labeling solver iso-dense, TT 2^{bits}",
        dump_path.display(),
        kept.len(),
    );

    // Label + score every kept node in parallel (concurrent `wins` on the shared lockless TT — the
    // production pattern; `Solver: Sync`). The shared TT means overlapping sampled subtrees are paid
    // once across the whole sweep.
    let scores: Vec<NodeScore> = kept
        .par_iter()
        .map(|&avail| score_node(&q, solver, avail))
        .collect();

    // Aggregate per pc and overall (recurse-weighted).
    let mut per: Vec<Agg> = vec![Agg::default(); pc_hi + 1];
    let mut all = Agg::default();
    for s in &scores {
        let a = &mut per[s.pc];
        for t in [&mut *a, &mut all] {
            t.nodes += 1;
            if !s.cut {
                t.nocut += 1;
                continue;
            }
            t.nrec_sum += s.n_recurse as u64;
            if s.rrank[0] > 0 {
                t.rec_cut += 1;
            }
            for c in 0..NCAND {
                t.sum_rrank[c] += s.rrank[c] as u64;
            }
        }
    }

    // Recurse-weighted capture for candidate c: (Σbase_rrank − Σrrank[c]) / Σbase_rrank — the fraction
    // of avoidable RECURSE expansions (the only wall-relevant ones) this ordering removes.
    let capt = |a: &Agg, c: usize| -> f64 {
        if a.sum_rrank[0] == 0 {
            0.0
        } else {
            100.0 * (a.sum_rrank[0] as f64 - a.sum_rrank[c] as f64) / a.sum_rrank[0] as f64
        }
    };

    let mut hdr = format!(
        "    {:>3} {:>9} {:>7} {:>8} {:>9}",
        "pc", "scored", "avgRec", "recCut%", "base_rr"
    );
    for name in CAND_NAMES.iter().skip(2) {
        hdr.push_str(&format!(" {name:>12}"));
    }
    println!("{hdr}   (candidates = % of avoidable RECURSE expansions captured vs live order)");
    let print_row = |label: String, a: &Agg| {
        let cut = (a.nodes - a.nocut).max(1);
        let mut row = format!(
            "    {label:>3} {:>9} {:>7.2} {:>7.1}% {:>9.3}",
            a.nodes,
            a.nrec_sum as f64 / cut as f64, // avg recurse children per cut node
            100.0 * a.rec_cut as f64 / cut as f64, // % of cuts that need a recurse child
            a.sum_rrank[0] as f64 / cut as f64, // mean avoidable recurse exps / cut node
        );
        for c in 2..NCAND {
            row.push_str(&format!(" {:>11.1}%", capt(a, c)));
        }
        println!("{row}");
    };
    for (pc, a) in per.iter().enumerate().take(pc_hi + 1).skip(pc_lo) {
        if a.nodes > 0 {
            print_row(pc.to_string(), a);
        }
    }
    print_row("ALL".to_string(), &all);
    println!(
        "  avgRec = recurse children (child_pc>{RECURSE_MIN}) per cut node · recCut% = cuts needing ≥1 recurse child\n  \
         base_rr = mean recurse expansions examined before the cut (live order) = the per-node node-reduction ceiling"
    );
    let cut_all = (all.nodes - all.nocut).max(1);
    println!(
        "  TOTAL avoidable recurse expansions (sample) = {} over {} cut nodes ({:.3}/node); a recurse-suffix\n  \
         reorder caps at 100% (oracle). If rec-sumgc capture ≈ 0 or base_rr ≈ 0, the lever is DEAD.",
        all.sum_rrank[0],
        cut_all,
        all.sum_rrank[0] as f64 / cut_all as f64,
    );
    Ok(())
}
