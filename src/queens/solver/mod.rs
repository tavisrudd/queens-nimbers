//! The solver lineage: the `Solver` trait, the per-node key knobs shared by
//! the table-backed solvers, and the `make_solver` factory.

use crate::queens::*;

mod burr;
mod fused;
mod incremental;
mod iso_flat;
mod memo;
mod naive;
mod nimber;
mod parallel;
mod pn;
mod ranklab;

pub use burr::{Burr, IsoBurr};
pub use fused::Fused;
pub use incremental::Incremental;
pub use iso_flat::IsoFlat;
pub use memo::{BranchingStats, Tt};
pub use naive::Naive;
pub use nimber::{Nimber, NimberSum};
pub use parallel::Parallel;
pub use pn::Pn;
pub use ranklab::run_ranklab;

// --------------------------------------------------------------------------- //
// Solver lineage
// --------------------------------------------------------------------------- //

/// Work-stealing diagnostics for the `--to-file` JSON results (and the post-solve TTY line):
/// what the dominant-root tail split off late in the solve and how the gate was configured.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StealReport {
    /// Subtrees split off to idle cores.
    pub published: u64,
    /// Published-but-not-landed children the busy worker re-expanded itself (PASS1 fallback).
    pub fallback: u64,
    /// Available-popcount range and mean of the split subtrees (what we split).
    pub pc_lo: u32,
    pub pc_hi: u32,
    pub pc_mean: f64,
    /// Full available-popcount histogram of the split subtrees: `(pc, count)`, ascending.
    pub pc_hist: Vec<(u32, u64)>,
    /// Gate config in force (`QUEENS_STEAL_*`).
    pub width: u32,
    pub min_pc: u32,
    pub max: u32,
    pub delay: u64,
}

/// A win/loss solver for the Non-Attacking Queens game. Implementors compute
/// `wins` (the value for the player to move); the rest is provided.
pub trait Solver: Sync {
    /// The solver's name (for the CLI / reporting).
    fn name(&self) -> &'static str;

    /// Does the player to move win from `blocked` under perfect play?
    fn wins(&self, q: &Queens, blocked: Bits) -> bool;

    /// Does the first player win the empty board? The default is a plain
    /// `wins(empty)`; [`Parallel`] overrides it with the odd-board O(1) theorem
    /// and root parallelism.
    fn first_player_wins(&self, q: &Queens) -> bool {
        self.wins(q, Bits::empty())
    }

    /// Nodes searched (TT misses), for reporting. `0` if not tracked.
    fn nodes(&self) -> u64 {
        0
    }

    /// Flush every worker's thread-local node/HLL tally into the shared totals, so
    /// [`nodes`](Self::nodes) and [`report`](Self::report) are exact. The CLI calls this once
    /// after the search (the hot loop only flushes ≈ once a second to avoid a per-node atomic
    /// on this cross-CCX box). Default no-op for solvers that don't accumulate locally.
    fn drain(&self) {}

    /// Per-node branching / cutoff tally, if built with [`Tt::with_branching`]
    /// (`count --branching`). `None` for an ordinary solve.
    fn branching_stats(&self) -> Option<BranchingStats> {
        None
    }

    /// Transposition-table byte footprint (the memory cap). `0` if none.
    fn cap_bytes(&self) -> u64 {
        0
    }

    /// Distinct-position measurement, if this solver was built with counting
    /// enabled (see [`Tt::new_counting`]). `None` for an ordinary solve.
    fn report(&self) -> Option<CountReport> {
        None
    }

    /// Work-stealing diagnostics (how many subtrees were split off the dominant-root tail, their
    /// available-popcount distribution, the fallback re-expansion count, and the gate config).
    /// `None` unless this solver ran with stealing on. Powers the `--to-file` JSON results.
    fn steal_report(&self) -> Option<StealReport> {
        None
    }

    /// The exact working set (canonical key, win/loss value), for cold post-search
    /// analysis (`count --iso`). `None` unless an exact distinct set was kept.
    fn working_set(&self) -> Option<Vec<(Bits, u8)>> {
        None
    }

    /// Root-move progress as `(resolved, total)` for a live indicator, or `None`
    /// if the solver does not track it. Only meaningful mid-`first_player_wins`.
    fn root_progress(&self) -> Option<(u64, u64)> {
        None
    }

    /// Extra, approach-specific stats for the solve summary -- e.g. table fill
    /// for the memo solvers, the Sprague-Grundy value for `nimber`, the root
    /// proof/disproof numbers for `pn`. Empty by default (e.g. tableless `naive`).
    fn stats(&self) -> String {
        String::new()
    }

    /// The transposition table, if this solver has one -- so a checkpoint can dump
    /// it mid-search (`QueensTt::dump_image`). `None` for tableless solvers (`naive`).
    fn tt(&self) -> Option<&QueensTt> {
        None
    }

    /// Per-available-popcount flat-TT put histogram, indexed by popcount, when the solver
    /// was run with `QUEENS_PC_HIST=1` (segmented-TT band sizing). `None` otherwise.
    fn pc_hist(&self) -> Option<Vec<u64>> {
        None
    }

    /// Stratified TT-latency profile (`QUEENS_PROF=1`): a flat `4 * 257` vector laid out
    /// `[metric * 257 + pc]` with metric ∈ {get-cyc, get-n, put-cyc, nodes}. `None` otherwise.
    fn prof_data(&self) -> Option<Vec<u64>> {
        None
    }

    /// Per-rayon-worker cumulative node tallies for the live per-core throughput display. `None`
    /// for solvers without a shared TT (the watcher then shows only the aggregate rate).
    fn per_worker_nodes(&self) -> Option<Vec<u64>> {
        None
    }
}

/// Which canonical key the search uses per node. `D4` is the production key
/// (`pos_key`, the dihedral-canonical `available` mask). `GraphIr`/`GraphCanon` are
/// the **graph-isomorphism** keys (session-6 lever #7) -- they merge ~3.4× more
/// positions (every isomorphic available-graph), but cost ~µs/node vs `pos_key`'s
/// ~ns, so this is a measurement/spike toggle (`QUEENS_KEY=ir|canon`), not yet the
/// default. Only meaningful for the canonical solvers (`canon == true`).
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum KeyMode {
    D4,
    GraphIr,
    GraphCanon,
    GraphComp,
    GraphFast,
}

/// Resolve the key mode once at construction (never per node -- an env read in the
/// hot loop serialises the rayon workers). `QUEENS_KEY=ir|canon|comp` opts into a
/// graph-isomorphism key; anything else keeps the production D4 key.
fn key_mode() -> KeyMode {
    match std::env::var("QUEENS_KEY").as_deref() {
        Ok("ir") => KeyMode::GraphIr,
        Ok("canon") => KeyMode::GraphCanon,
        Ok("comp") => KeyMode::GraphComp,
        Ok("fast") => KeyMode::GraphFast,
        _ => KeyMode::D4,
    }
}

/// Pack a 64-bit graph-isomorphism key into a tagged table-key namespace. Selective
/// keying can mix graph and D4 keys even on n=16, where no spare board bit exists, so
/// the D4 fallback is also hashed into a disjoint tag by [`d4_bits`].
#[inline]
fn graph_bits(h: u64) -> Bits {
    Bits([h, mix64(h ^ 0x9E37_79B9_7F4A_7C15), 0x150_600D_600D_600D, 0])
}

/// Tagged D4 fallback key for selective graph/D4 modes. Plain D4 mode still stores the
/// exact 256-bit canonical mask; this fast 192-bit reduction is used only when a
/// graph-key mode falls back to D4 and therefore needs a namespace disjoint from
/// [`graph_bits`] on full 16x16 boards. The table/store hashes the resulting `Bits`
/// again; this layer only separates namespaces and keeps enough D4 entropy cheaply.
#[inline]
fn d4_bits(k: Bits) -> Bits {
    let w = k.0;
    Bits([
        w[0],
        w[1],
        w[2] ^ w[3].rotate_left(32) ^ 0xD4D4_D4D4_D4D4_D4D4,
        0xD400_D4D4_D4D4_D4D4,
    ])
}

/// Resolve the selective-keying threshold once: with `QUEENS_KEY_MAX=k`, only positions
/// whose available-graph has ≤ k vertices use the (costly) graph key; larger graphs fall
/// back to the cheap D4 key. Safe because transpositions are strictly intra-ply, and the
/// choice is a pure function of the position (its available popcount). Default: no limit.
fn key_max_avail() -> u32 {
    std::env::var("QUEENS_KEY_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u32::MAX)
}

/// Tuned default for `iso-burr`: the n=16 blend table keeps ~73% of the full iso
/// merge while keying ~46% of positions at this threshold. `QUEENS_KEY_MAX` remains
/// the experiment override for 6/8/9 sweeps.
fn iso_burr_key_max_avail() -> u32 {
    std::env::var("QUEENS_KEY_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7)
}

/// `iso-flat`'s default iso threshold: **7** — the top of the no-WL regime (`iso_key_tiny_table`
/// is exact for popcount ≤7; 8 falls into the costly WL canon). 7 merges more than 6 (smaller
/// resident set → less eviction at n=16) at no WL cost, so it's the wall-optimal corner: n=16
/// **8m20s / 1.15× re-exp** vs KEY_MAX=6's 9m11s / 1.27× (with `QUEENS_TT_SLOTS≈2.4e9`), and
/// wall-neutral at n=14. `QUEENS_KEY_MAX` overrides for the full speed/merge/fit sweep (≥8 is
/// WL — slower/node but a smaller, near-eviction-free set).
fn iso_flat_key_max_avail() -> u32 {
    std::env::var("QUEENS_KEY_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7)
}

/// Plies from the root that [`Parallel`] fans across rayon (resolved once at startup,
/// never per node). `QUEENS_PAR_DEPTH` overrides; default `3`. Below this depth the
/// search recurses sequentially (full α-β cutoff). Higher exposes more parallelism --
/// keeping the dominant root-0 ("elder brother") subtree off a single core at n=16,
/// where that subtree is the entire feasible runtime -- at the cost of some speculation
/// at the OR (prove-a-win) levels; the AND (prove-a-loss) levels, the bulk of a
/// second-player win, parallelise with no speculation.
fn par_depth() -> u32 {
    std::env::var("QUEENS_PAR_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
        .max(1)
}

/// `QUEENS_PAR_MIN_AVAIL` override (resolved once at construction). `None` ⇒ auto by
/// board size (see [`min_avail_for`]). The size split keeps a *big* deep prove-a-loss
/// node fanning so an idle core can steal a straggler -- the #20 tail fix; available
/// count is a cheap proxy for subtree size (it shrinks with depth).
fn par_min_avail_override() -> Option<u32> {
    std::env::var("QUEENS_PAR_MIN_AVAIL")
        .ok()
        .and_then(|s| s.parse().ok())
}

/// The size-split threshold for board `n`: a node below [`par_depth`] keeps splitting
/// while its available count stays above this, else it goes sequential. The auto
/// default is **on only for n ≥ 15** (`96`) and **off below** (`u32::MAX`): the fixed
/// `par_depth` schedule is already well-tuned on the short small-board searches (where
/// extra splitting is pure overhead -- it regresses n=14 ~3%), and only the n=16 tail
/// -- few roots left, all parallelism intra-root, sequential stragglers draining cores
/// -- needs the deeper split. Rayon pays the split cost only on an actual steal, so at
/// n=16 it is ~free while saturated and pays off precisely at the tail. `over` (the env
/// override) wins when set; set it huge (≥ n²) to force the pure fixed-`par_depth` form.
fn min_avail_for(over: Option<u32>, n: u32) -> u32 {
    over.unwrap_or(if n >= 15 { 96 } else { u32::MAX })
}

/// CLI solver names, simplest → most sophisticated (`nimber` computes the full
/// Sprague-Grundy value; `pn` is df-pn proof-number search).
pub const SOLVER_NAMES: [&str; 13] = [
    "naive",
    "memo",
    "symmetry",
    "parallel",
    "incremental",
    "burr",
    "iso-burr",
    "fused",
    "iso-flat",
    "iso-window",
    "iso-dense",
    "nimber",
    "pn",
];

/// Build a solver by name with a `2^bits`-slot table (ignored by `naive`).
pub fn make_solver(name: &str, bits: u32) -> Option<Box<dyn Solver>> {
    match name {
        "naive" => Some(Box::new(Naive::new())),
        "memo" => Some(Box::new(Tt::new(bits, false))),
        "symmetry" => Some(Box::new(Tt::new(bits, true))),
        "parallel" => Some(Box::new(Parallel::new(bits))),
        "incremental" => Some(Box::new(Incremental::new(bits))),
        "burr" => Some(Box::new(Burr::new(bits))),
        "iso-burr" => Some(Box::new(IsoBurr::new(bits))),
        "fused" => Some(Box::new(Fused::new(bits))),
        "iso-flat" => Some(Box::new(IsoFlat::new(bits))),
        "iso-window" => Some(Box::new(IsoFlat::new_window(bits))),
        "iso-dense" => Some(Box::new(IsoFlat::new_dense(bits))),
        "nimber" => Some(Box::new(Nimber::new(bits))),
        "pn" => Some(Box::new(Pn::new(bits))),
        _ => None,
    }
}
