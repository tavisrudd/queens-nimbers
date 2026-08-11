# queens-nimbers

A solver for the adversarial **Non-Attacking Queens** game — Node Kayles on the n-queens graph —
together with the verification resources behind two published results:

- the **n = 18 outcome**: the 18×18 game is a **first-player win**, with winning opening `I9` and a
  15-ply principal variation, established by two independently configured proving runs;
- the **Sprague–Grundy nimbers** `G(14) = 0`, `G(15) = 1`, `G(16) = 0`, `G(17) = 2`, extending
  [OEIS A344227](https://oeis.org/A344227) (catalogued through `n = 13`). `G(17) = 2` refutes the
  eventual 1,0-oscillation conjecture recorded on the OEIS entry.

The game: two players alternately place a queen on an `n×n` board so that no two queens attack each
other (no shared row, column, or diagonal); a player who cannot move loses. It is impartial and
normal-play, so a position is captured entirely by its *blocked mask*, and the game is exactly Node
Kayles on the queens graph (Noon & Van Brummelen 2006; deciding the winner of Node Kayles is
PSPACE-complete, Schaefer 1978).

## Build and run

Rust stable, no system dependencies beyond a C toolchain for `zstd`:

```sh
cargo build --release

./target/release/queens solve 12          # who wins the empty 12x12 board, with an optimal line
./target/release/queens nimber 15         # the Sprague-Grundy value (heap-sum engine)
./target/release/queens --list-engines    # the solver ladder, weakest (ground truth) to strongest
```

`.cargo/config.toml` compiles for the host CPU, so the bit primitives use POPCNT/BMI directly.
Odd boards need no search: the first player wins by taking the centre and answering every reply
with its 180° rotation.

Reproducing the individual results costs, on a 24-thread workstation, roughly: `n = 14` seconds,
`n = 15` half a minute, `n = 16` a few minutes, `n = 17` about 59 hours with a 17 GB table, and
`n = 18` about 8 hours per proving run with a 17 GB table. Exact configurations are in
[verification/RESULTS.md](verification/RESULTS.md).

## What is here

| Path                | Contents                                                                       |
|---------------------|--------------------------------------------------------------------------------|
| `src/queens/`       | the game, the solver ladder, the `getK` dense leaf evaluator, the transposition table, the BuRR-backed store |
| `src/bin/queens.rs` | the command-line driver (`solve`, `nimber`, `count`, `play`, archive tools)      |
| `verification/`     | validation gates, the frozen results and run configurations, and the Lean 4 development |
| `oeis/A344227/`     | the ready-to-paste OEIS extension package and the b-file for `a(0)..a(17)`      |
| `docs/`             | the n = 18 technical report, an HTML report, and an interactive explorable      |

The solver ladder is deliberately kept whole: every step from the memo-less `naive` negamax up to
the production `iso-flat` kernel computes the same win/loss, so the simple ones remain runnable
ground truth rather than deleted history.

## Verification

The trust argument is layered, and its boundary is stated rather than glossed:
lineage agreement against the memo-less recurrence, exact distinct-position invariants,
bit-for-bit differential tests of the optimised `pext` leaf evaluator against a plain scalar
recurrence for every code width the n = 18 verdict bottoms out on, reproduction of the previously
published `n ≤ 16` sequence, and a `sorry`-free Lean 4 proof of the leaf evaluator's semantics.

```sh
cargo test --release              # includes the scalar differential gates for W8-W18
cd verification/lean && lake build # the Lean development (Lean 4 + mathlib, pinned)
```

Read [verification/README.md](verification/README.md) for what each gate covers, what the Lean
development does and does not certify, and the residual trusted base.

## Citing

The n = 18 solve and the nimber extension are described in
[docs/queens-n18-paper.md](docs/queens-n18-paper.md). Releases are archived on Zenodo, which mints
a version DOI for each one and a concept DOI for the software as a whole; cite the version DOI for
a specific result, or the commit if you are citing unreleased work.

## License

MIT — see [LICENSE](LICENSE).
