# Frozen results and their run configurations

Every claim this repository makes, with the configuration and evidence that produced it. Node
counts and wall times are from the actual proving runs on a 24-thread Zen 5 workstation with 26 GB
of RAM (about 16 GB free), compiled with `-C target-cpu=znver5`.

## Sprague–Grundy nimbers (OEIS A344227)

`a(0)..a(13)` are the previously catalogued terms and are reproduced by this engine. The new terms
are `a(14)..a(17)`, computed with the heap-sum engine (`queens nimber <n>`):

| n  | G(n) | evidence                                                                                     |
|----|------|----------------------------------------------------------------------------------------------|
| 14 | 0    | `k = 0` LOSS (1.4 s, 11.0 M nodes); equals the production `iso-dense` second-player verdict     |
| 15 | 1    | `k = 0` WIN and `k = 1` LOSS (23.8 s, 194 M nodes); reproduced at `QUEENS_NIMBER_GK=10`, a different leaf boundary and different code paths, with the same value |
| 16 | 0    | `k = 0` LOSS (2 m 21 s, 1.06 B nodes); independently equals the multi-configuration-validated n = 16 second-player verdict |
| 17 | 2    | `k = 0` WIN, `k = 1` WIN, `k = 2` LOSS, so `G = 2`. About 585 × 10⁹ nodes over about 59 hours with a 17 GB table (`QUEENS_TT_BITS=31`), `bk=20`, `gk=16`. Initial run 2026-07-04, revalidated 2026-07-07 |
| 18 | open | the n = 18 **outcome** below is a first-player win, so `G(18) ≠ 0`, but the exact value is not computed |

`G(17) = 2` is the term that matters mathematically: the OEIS entry records a conjecture that the
sequence oscillates between 1 and 0 after the ninth term. That holds through `n = 16` and breaks at
`n = 17`. The even side was already known to break, since the n = 18 first-player win forces
`G(18) ≠ 0`.

**Why a heap sum rather than a direct mex.** The minimal excludant admits no α-β cutoff, so a
full-DAG mex reference must expand every reachable position — hopeless past about `n = 13`. Instead
`G(board) = k` exactly when the game sum *board + Nim-heap(k)* is a P-position, and the win/loss of
that sum is α-β-searchable. The driver solves `win(board, k)` for `k = 0, 1, 2, …` until the first
LOSS, sharing one transposition table across rounds because `win(avail, h)` is round-independent.

## The n = 18 outcome

**The 18×18 game is a first-player win**, opening `I9` (square 152), with the 15-ply principal
variation

```
I9  K8  G10  J11  H3  M7  N16  E4  P6  D12  O13  F2  R5  L17  A14
```

(squares `152, 136, 168, 189, 43, 120, 283, 58, 105, 201, 230, 23, 89, 299, 234`).

Two proving runs that differ in the dense leaf evaluator they use agree on the verdict, the
winning move, and the entire principal variation byte for byte:

| run     | `dense_k` | `getK` code path     | verdict    | root | nodes           | wall     |
|---------|-----------|----------------------|------------|------|-----------------|----------|
| primary | 17        | W17 (192-bit)        | first wins | I9   | 258,322,944,571 | 8h16m45s |
| confirm | 20        | W18/19/20 (≥190-bit) | first wins | I9   | 114,318,641,519 | 7h08m39s |

Shared configuration: 24 worker threads; a 17 GB flat table
(`QUEENS_TT_SLOTS = 2.125 × 10⁹` 8-byte slots, about 16.7 GB resident); band-skip transposition
work for `pc ∈ [18, 25]` (`QUEENS_SKIP18_PCS=18,…,25`), which is verdict-preserving by construction
because the value is still computed and only the memoisation of a band that is about 100 % cold is
declined. Runs are not resumable: the flat table is not checkpointed.

The node counts differ by more than 2× because the table cannot hold the full working set and
transpositions get recomputed; the higher dense ceiling of the confirm run shrinks the working set
and roughly halves that re-expansion. Agreement on the value across that difference is the point.

The principal variation is not itself the proof. It was independently re-verified by direct board
arithmetic with no search — all fifteen placements pairwise non-attacking and available when
played, and the available set exactly empty after the fifteenth — but the win claim rests on the
searches having refuted every opponent reply after `I9`.

## Replay in this repository

The extracted code was replayed from a clean checkout of this repository on 2026-08-11 (24-thread
Zen 5 workstation, `cargo build --release`, default table settings), reproducing both the
catalogued term `a(13)` and the new terms through `a(16)`:

| n  | G(n) | nodes       | wall    |
|----|------|-------------|---------|
| 13 | 1    | 927,028     | 0.46 s  |
| 14 | 0    | 3,056,616   | 1.05 s  |
| 15 | 1    | 69,793,449  | 17.5 s  |
| 16 | 0    | 584,833,906 | 130.2 s |

Node counts differ from the original runs because the default table size and leaf boundaries here
differ from the ones used then; the values agree, which is what the replay is for. `a(17)` was not
replayed (about 59 hours), and neither was the n = 18 outcome (about 8 hours per run).
`cargo test --release` passes, including the scalar differential gates for the W8–W18 layers.

## Reproduction commands

```sh
cargo build --release
./target/release/queens nimber 14      # seconds
./target/release/queens nimber 15      # about half a minute
./target/release/queens nimber 16      # a few minutes
QUEENS_TT_BITS=31 ./target/release/queens nimber 17   # about 59 hours, needs about 17 GB
```

For the n = 18 outcome, run `queens solve 18 iso-dense` with the band-skip and table settings above:

```sh
QUEENS_SKIP18_PCS=18,19,20,21,22,23,24,25 \
QUEENS_TT_SLOTS=2125000000 \
QUEENS_DENSE_K=17 \
./target/release/queens solve 18 iso-dense
```

`QUEENS_DENSE_K` selects which of the two cross-validating leaf configurations is used: `17` is the
primary run's W17 path, `20` the confirm run's wide W18/19/20 path.
