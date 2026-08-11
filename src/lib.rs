//! Solver for the adversarial **Non-Attacking Queens** game — Node Kayles on the
//! n-queens graph, whose Sprague–Grundy values are OEIS A344227.
//!
//! Two players alternately place a queen on an `n×n` board so that no two queens
//! attack each other; the player who cannot move loses (normal play). See
//! [`queens`] for the game, the solver ladder, and the `getK` leaf evaluator.
//!
//! ```no_run
//! use queens_nimbers::queens::{Queens, Solver};
//! // n = 12 is a second-player win.
//! let q = Queens::new(12);
//! # let _ = q;
//! ```
//!
//! The supporting modules are carried because the solver uses them:
//! [`burr`] (the succinct retrieval structure behind the disk-backed archive),
//! [`affinity`] (CPU pinning for the parallel search), and [`table`] (terminal
//! rendering for the CLI reports).

pub mod affinity;
pub mod burr;
pub mod queens;
pub mod table;
