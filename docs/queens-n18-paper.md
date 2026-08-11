# Solving the Non-Attacking Queens game for n = 18: a transposition-driven impartial-game solver, its performance engineering, and a machine-checked leaf evaluator

**Draft — technical report for specialist review.** Markdown source; render with pandoc for a typeset version.

---

## Abstract

The Non-Attacking Queens placement game is the impartial, normal-play game in which two
players alternately place a queen on an n×n board so that it attacks no previously placed
queen; the player unable to move loses. It is equivalent to Node-Kayles on the *queen graph*
(vertices = squares, edges = mutually-attacking pairs), and deciding the winner is
PSPACE-complete in general. Jenrich (arXiv:1312.5135) determined the winner for all n ≤ 16;
n = 18 was open. We report that **n = 18 is a first-player win**, witnessed by the opening
move **I9** and a 15-ply principal variation, established by an exhaustive boolean game-tree
search. The verdict is obtained by two *independently configured* runs of the same solver —
two distinct dense-leaf evaluators (a 192-bit code path and a ≥190-bit code path) — that
agree on the verdict, the winning move, and the byte-identical principal variation while
visiting very different node counts (≈2.58 × 10¹¹ and ≈1.14 × 10¹¹). We describe (i) the
algorithmic core — a lockless flat transposition table over isomorphism-canonicalised
positions, with a dense leaf evaluator (`getK`) that resolves every position with at most
`dense_k` live squares directly from precomputed Node-Kayles tables; (ii) the performance
engineering that took the n = 16 search from minutes to tens of seconds and made n = 18
tractable on a single 26 GB workstation; and (iii) a layered validation strategy, culminating
in a machine-checked Lean 4 proof of the leaf evaluator's recurrence, its isomorphism and
induced-subgraph invariances, the build recurrence, and the Sprague–Grundy/Grundy
characterisation, kernel-complete and depending only on mathlib's standard axioms. Separately,
a heap-sum Sprague–Grundy engine — which decides `G(board) = k` by α-β-searching the game sum of
the board with a Nim-heap of size `k` — **extends OEIS A344227**, the game's nimber sequence
previously catalogued only through n = 13, with the four new terms **G(14) = 0, G(15) = 1,
G(16) = 0, and G(17) = 2**; the engine reproduces the full-mex nimbers for n ≤ 8 and the A344227
terms for n ≤ 13 exactly, and its n = 14/16 values independently equal the production win/loss
verdicts. The value G(17)=2 also refutes the conjectured odd → 1 continuation. We also give an
elementary **structural theory** of the
even/odd split: a 180°-rotation pairing argument reproves that every odd board is a first-player
win and — new here — proves that an even-board first-player win *requires* a long-diagonal move
at some ply (the responder's mirror refutes any diagonal-free line) — and the n = 18 witness
I9 = (8, 8) lies on the main diagonal, consistent with that theorem, which constrains winning
*lines*, not opening moves (at n = 6, computed non-diagonal openings win by striking a diagonal
later). Two companion studies extend the theory: two further proven general laws (a
**Closed-Pairing** theorem generalising the mirror argument to arbitrary matchings with a closure
property, and a **Well-Covered Parity** law with the rook game as its special case), a computed
geometry of winning openings on every solved small board supporting a *Forcing-Root* conjecture
(the most-forcing square wins on every known first-player-win board), exhaustive ablations showing
that refuting the central strike *requires* border access, torus-queens nimbers computed to
n = 10, and a registered structural conjecture for n = 20 (first player, witness (9, 9)). The
reduction and the new laws are proven; *why* the even pattern first breaks at n = 18, the exact
value of G(18), and the n = 20 outcome are given as explicit heuristics and falsifiable
predictions. We are explicit throughout about what is *certified*, what is *cross-validated*, and
what is *deferred to differential testing* or *conjectural*.

---

## 1. Introduction

### 1.1 The game

In the **Non-Attacking Queens game** (Noon & Van Brummelen 2006), players alternately place a
queen on an n×n board subject to the constraint that no two queens attack each other (same
row, column, or diagonal). This is the *placement* / *misère-free* variant: it is **impartial**
(both players have the same moves from any position) and played under the **normal-play
convention** — the player who cannot move loses, equivalently the last player to move wins.

The game is equivalent to **Node-Kayles** on the *queen graph* `Q_n`: the vertices are the n²
squares, and two squares are adjacent iff a queen on one attacks the other. A position is the
set `S` of squares still available (neither occupied nor attacked). A move selects a live
vertex `v ∈ S`; it removes `v` together with all squares it attacks — that is, the *closed
neighbourhood* `N[v] = {v} ∪ N(v)` — leaving the live set `S \ N[v]`. A terminal position
(`S = ∅`) is a loss for the player to move. Deciding the winner of Node-Kayles is
PSPACE-complete (Schaefer 1978), so there is no expected closed form for general n; the
practical question is how far exhaustive search can be pushed.

For **odd** n the first player wins by an O(1) strategy (play the centre, then mirror the
opponent through the 180° rotation), so the substantive open problems are the **even** boards.
The winners are known to be: every odd n → first; n ∈ {4, 6, 8} → first; n ∈ {10, 12, 14} →
second; and, from Jenrich's 2014 computation, **n = 16 → second**. The Sprague–Grundy nimbers
of small boards are catalogued in OEIS **A344227**. The next open even board was **n = 18**.

### 1.2 Contribution

1. **A new game-theoretic result: n = 18 is a first-player win.** The witness is the opening
   move **I9** (square 152 in the 0-based row-major numbering); the game is decided by an
   exhaustive negamax search of the I9 subtree. This extends Jenrich's n ≤ 16 **win/loss**
   sequence. It does **not** add a term to OEIS A344227, which records the exact Sprague–Grundy
   *nimber* and is catalogued only through n = 13: an exhaustive solve yields the *outcome*, and a
   first-player win fixes only that the nimber is nonzero, not its value. Indeed A344227's listed
   values follow a conjectured n ≥ 10 oscillation 0 (even) / 1 (odd) — so an **even**-board
   first-player win at n = 18 *contradicts* the even → 0 prediction rather than extending it. (The
   nimber *values* are extended separately, by the engine of contribution 5.)

2. **A solver design** combining a lockless flat transposition table, isomorphism-aware
   position canonicalisation, and a *dense leaf evaluator* (`getK`) that resolves all positions
   with `pc(S) ≤ dense_k` live squares directly from precomputed complete Node-Kayles tables —
   without subtree expansion or table probes — using a BMI2-`pext` child sweep over a
   bit-packed adjacency code.

3. **A performance-engineering account** — measured, with both wins and instructive negatives —
   that reduced the n = 16 search from the first complete run's ≈10.0 × 10⁹ node evaluations /
   ≈56 min to ≈1.79 × 10⁸ nodes / ≈13.4 s on the same machine, and made the n = 18 root feasible
   via a deliberate *capacity* configuration (band-skipped transposition work + a 17 GB flat
   table).

4. **A layered validation and verification strategy**, including a machine-checked **Lean 4**
   proof of the leaf evaluator's *semantics* — the win/loss recurrence, its termination, its
   isomorphism and induced-subgraph invariances, the table-build recurrence, and the
   Grundy/Sprague–Grundy characterisation — kernel-complete with no `sorry` and depending only
   on the three standard mathlib axioms.

5. **An extension of OEIS A344227 (the nimber sequence itself).** A separate *heap-sum
   Sprague–Grundy engine* computes the exact nimber `G(n)` by α-β-searching the game sum
   `board + Nim-heap(k)` for ascending `k` (a loss at `k` pins `G = k`), contributing the new
   terms **G(14) = 0, G(15) = 1, G(16) = 0, G(17) = 2** beyond the previously catalogued
   n ≤ 13 (Section 6). The engine reproduces the full-mex nimber for n ≤ 8 and A344227 for
   n ≤ 13, and its n = 14/16 values independently coincide with the production win/loss verdicts.
   The new values break both halves of the conjectured oscillation: odd → 1 fails at n = 17, and
   even → 0 fails at n = 18.

6. **An elementary structural theory of the even/odd split** (Section 6). A 180°-rotation (`ρ`)
   pairing argument reproves that every odd board is a first-player win, and — new — proves that
   an even-board first-player win *requires* a long-diagonal move: the responder's mirror strategy
   refutes any first-player line that avoids the long diagonals. The n = 18 witness I9 = (8, 8)
   sits on the main diagonal, consistent with the theorem — which constrains winning *lines*, not
   first moves (computed n = 6 counterexample: non-diagonal openings win by striking a diagonal
   later; Section 6.6). Two companion studies (Sections 6.6–6.8) extend the theory: the computed
   geometry of winning openings and the *Forcing-Root* conjecture, root option-value spectra that
   mechanically explain the small catalogued nimbers, exhaustive border-tempo ablations with a
   torus-queens control computed to n = 10, and two new proven general laws (Closed-Pairing;
   Well-Covered Parity). The reduction and the laws are proven; *why* the pattern first breaks at
   n = 18, the exact nimber value G(18), and the n = 20 outcome (Section 9) are stated as
   heuristics, conjectures, and falsifiable predictions — never with the theorem's status.

### 1.3 What this paper claims, and what it does not

A complete formal certificate of a 10¹¹-node search is infeasible. We therefore set, and meet,
a different bar for the n = 18 verdict: **cross-validated agreement** of two independently
configured exhaustive searches, on top of a leaf evaluator that is differential-tested against
an independent scalar reference, checked against an independent raw-mask oracle on thousands of
high-index subpositions, audited for integer-width defects, and validated end-to-end against
Jenrich's published n ≤ 16 sequence. The Lean proof hardens the *recurrence semantics* of the
leaf evaluator (the historically bug-prone component), with the bit-level serialization left to
differential tests. Section 8 states the residual trusted base precisely. We deliberately avoid
claiming any "floor" on performance: the numbers below are milestones, not limits.

---

## 2. Problem formalisation and prior work

### 2.1 The win recurrence

Write `G` for the queen graph and `S ⊆ V(G)` for the set of live vertices. Define the boolean
predicate `win(G, S)` — "the player to move from `S` wins under normal play" — by

```
    win(G, S)  ⟺  ∃ v ∈ S .  ¬ win(G, S \ N[v])           (P/N recurrence)
    win(G, ∅)  =  false                                    (terminal = loss)
```

This is the standard finite, impartial, normal-play P/N recurrence: a position is an
N-position (win for the mover) iff some move reaches a P-position (loss for the mover), and the
empty position is a P-position. It is well-founded because `v ∈ S ∩ N[v]`, so `S \ N[v] ⊊ S`
and `|S \ N[v]| < |S|`. The whole game is `firstPlayerWins(G) ≡ win(G, V(G))`.

Equivalently, the **Grundy value** `grundy(G, S) = mex { grundy(G, S \ N[v]) : v ∈ S }`
characterises the outcome: `win(G, S) ⟺ grundy(G, S) ≠ 0`, and for a position that splits into
two parts with no edges between them, `grundy(G, S₁ ∪ S₂) = grundy(G, S₁) ⊕ grundy(G, S₂)`
(Sprague–Grundy). The solver uses the boolean form (with α-β pruning); the Grundy form is
relevant to an optional component-decomposition lever (Section 4.6), underpins the heap-sum
nimber engine (Section 6), and is machine-checked in the Lean verification (Section 7.3).

### 2.2 Prior work

- **Jenrich (arXiv:1312.5135), 2014.** Determined the winner for all n ≤ 16, finding n = 16 to
  be a second-player win. The reported computation used a backtracking search with partial
  symmetry handling and no transposition table, on the order of **7.146 × 10¹⁰** recursive calls
  over roughly 23 hours. This is the baseline our kernel reproduces (verdicts and the n ≤ 16
  sequence) and improves on in node efficiency.
- **OEIS A344227** records the Sprague–Grundy nimber of the game; our verdicts agree through the
  catalogued range (n ≤ 13 for the nimber, the win/loss outcome through n = 16). Section 6 extends
  the nimber sequence itself through n = 17.
- **Schaefer (1978)** established PSPACE-completeness of Node-Kayles, framing why exhaustive
  search, not a formula, is the tool. The complexity frame has since been refined in a way that
  matches our result shape exactly: nimber-preserving reductions strictly refine
  winnability-preserving ones, and Generalized Geography is complete for polynomially-short
  impartial rulesets under them, while not every PSPACE-complete ruleset is Sprague–Grundy-complete
  [13]. Computing the nimber is thus harder *in kind* than deciding the outcome — and our game is
  itself polynomially short (game length ≤ α(Q_n) ≤ n), so it sits inside that completeness class.
  The n = 18 *outcome* falling (Section 5) while the n = 18 *nimber* is still being computed
  (Section 6) instantiates the gap.
- **Nimber sequences of Node-Kayles families.** Grundy sequences of Node-Kayles are known to be
  eventually periodic on several sparse structured families — paths (Dawson's chess, octal 0.137,
  period 34), cycles, hypercubes, generalized Petersen graphs, and trees [10]. The queen family is
  a dense, module-free test case where the analogous periodicity question (the graph counterpart
  of Guy's nim-sequence question) is open; our extension supplies the terms the question needs.
- **Parameterized hardness.** Node-Kayles is fixed-parameter tractable by vertex cover,
  neighbourhood diversity, modular width, and twin cover, and W[1]-hard in the number of turns
  [14]. The queen graph Q_n simultaneously defeats every one of these parameters — Θ(n)-cliques
  (rows) blow up treewidth, the vertex cover has size n² − n, and our own measured module
  statistics show the search tail is essentially module-free (Section 4.6) — so a solver's
  reliance on transposition and move ordering rather than structural decomposition is forced by
  the parameter map, and W[1]-hardness in turns says depth-parameterization cannot rescue it.
- **The same object in statistical mechanics.** Viewed as a long-range hard-core lattice gas, the
  n-queens system shows **no bulk thermodynamic phase transition** (Monte Carlo to N = 1024 plus a
  transfer-matrix tensor network, recovering Simkin's counting constant) [15]. This is independent,
  physics-side evidence complementary to our central structural finding: the hardness is not bulk
  criticality — the torus game collapses to a parity law while all n-dependent difficulty
  concentrates at the border (Section 6.7). Rooks correspond to the free-fermion case (dimers on
  K_{n,n}); queens are the first interacting layer above it.
- **Static symmetric solutions.** The static analogue of our mirror obstruction is classical: no
  n-queens *solution* is fixed by any reflection (the symmetry classes are trivial, C₂, C₄ only).
  Recent work on reflecting configurations — equivalent by Klarner to Slater's problem of pairing
  1..n with n+1..2n under distinct sums and differences — shows the diagonal sum/difference
  structure remains an active existence frontier in extremal combinatorics [16]. Our
  Mirror-Obstruction Lemma (Section 6.4) is the strategic counterpart: the obstruction moves from
  configurations to pairing strategies and localizes on the two long diagonals.
- **Sprague (1935), Grundy (1939); Conway,** *On Numbers and Games.* The impartial-game theory
  underlying Sections 2.1 and 6.3.

n = 17, being odd, is a first-player win by the centre-symmetry argument and requires no search;
the present work concerns the even boards, where n = 18 was the open frontier.

---

## 3. Algorithmic approach

The solver is a parallel boolean game-tree search over canonicalised positions, with a dense
leaf evaluator that short-circuits the deepest ~21 %+ of the tree. Four components carry the
design.

### 3.1 Negamax with α-β over the boolean value

The search realises the recurrence of Section 2.1 directly as a depth-first negamax over
win/loss values with α-β pruning. The tree alternates between two node kinds, which the solver
exploits both for cutoffs and for parallelism (Section 4.5):

- **"prove-a-win" nodes** (the mover seeks any winning move): a single losing child suffices —
  cutoff on the first child that returns a loss for the opponent;
- **"prove-a-loss" nodes** (the mover must be shown to lose): *every* child must be searched
  (no cutoff), since one surviving win for the opponent would refute the loss.

At the root, an existence proof needs only one winning first move; the n = 18 run searched the
single move I9 and stopped (Section 5).

### 3.2 Position representation and canonicalisation

A position is the live mask `S = board & ¬blocked`, where `blocked` is the union of occupied
and attacked squares. For n = 18 the board has 324 squares, so masks are 384-bit (`[u64; 6]`)
and square indices are 16-bit (the consequences of getting this width wrong are the subject of
Section 7.1).

Two canonicalisations merge equivalent positions before they reach the transposition table:

- **Dihedral D₄ board symmetry.** The eight board orientations (rotations and reflections) are
  folded to a canonical representative via a `d4_bits` bijection and a 128-bit hash. This is the
  cheap, always-applied merge.
- **Graph-isomorphism key (selective).** Beyond D₄, isomorphic *available graphs* are merged by
  a Weisfeiler–Leman + individualisation-refinement canonical key, applied selectively to small
  components (a tunable cap). The isomorphism merge is worth roughly a 3.4× reduction in distinct
  positions at small n; it is bounded so its per-node cost does not dominate.

Canonicalisation is the largest single source of node-count reduction and the reason a flat
table is competitive: equivalent subtrees collapse to one entry.

### 3.3 The flat lockless transposition table

The transposition table is a single contiguous `Box<[AtomicU64]>` accessed with relaxed
loads/stores — no sharding, no per-bucket mutex, no read-modify-write. Each slot is **one 64-bit
word** holding `{ used: 1 bit, value: 8 bits, fingerprint: 55 bits }`: the slot stores a 55-bit
fingerprint of the canonical key rather than the full key, so a colliding probe returns a wrong
value with probability ≈ 2⁻⁵⁵ (cross-checked against Jenrich's known verdicts). Indexing uses
Lemire's multiply-shift (`fastrange`) so the table can be any number of slots — sized once at
startup from an environment variable — and the next child's slot is software-prefetched. The
table is backed by 2 MB huge pages (Section 4.7).

A single-word, fingerprint-only slot is what makes the lockless design sound: a `u64` write
cannot tear, and the fingerprint self-validates a foreign key, so concurrent workers need no
synchronisation beyond atomicity.

### 3.4 The dense leaf evaluator `getK` and the W_K hierarchy

The deepest part of the tree is where most nodes live and where memoisation pays least (subtrees
are shallow and rarely revisited). The solver resolves every position with `pc(S) ≤ dense_k`
**directly**, without a table probe or any subtree expansion, via the `getK` evaluator:

1. Precompute, once at startup, the **complete** Node-Kayles win tables `W0 … W8` for *all*
   labelled graphs on up to 8 vertices. There are `8·7/2 = 28` possible edges, so `W8` is a
   2²⁸-bit value bitset — **32 MiB, n-independent, eviction-free** (every lookup is a hit), built
   bottom-up by vertex count in ≈2 s. `W8` lives in 16 huge pages and is TLB-friendly.

2. For a position with `k = pc(S) ≤ dense_k` live vertices, build the `k(k−1)/2`-bit
   upper-triangular adjacency *code* of the induced available graph (one BMI2-`pext` per row;
   Section 4.3). Then for each of the ≤ k moves, project the surviving subgraph's code with a
   single `pext` and look the child up in `W[pc(child)]`; the position is a win iff some child is
   a loss. Because a move deletes `1 + deg(v)` vertices, children fall several layers down, so
   the nested sweep bottoms out in the complete `W0 … W8` base within a couple of levels.

`getK` therefore computes the *exact* value of layers `W9, W10, …` that would be astronomically
expensive to *store* (`W9` as a table is 8 GiB; `W10` ≈ 4 TB) at **zero storage cost**. The
dense ceiling `dense_k` is a tuning parameter (Section 4.2): raising it trades more per-node
evaluator work for fewer search nodes. The dispatch on `dense_k` is a compile-time generic
resolved once per run, never a per-node branch.

---

## 4. Performance engineering

The solver is **transposition- and memory-latency-bound**, not compute- or
parallelism-bound: the binding cost is the depth-first entry probe into a multi-gigabyte table,
which is an inherently serial, path-dependent random access. This shaped every lever below. All
n = 16 figures carry ≈ ±18 % run-to-run node-count noise from parallel cutoff timing; we cite
**deterministic single-threaded n = 14 node counts** as the noise-free measure of direction, and
**cycles/node** (perf cycles ÷ solver nodes) for byte-identical micro-changes. Methodology is in
Section 4.8.

### 4.1 The solver lineage

The n = 16 search wall-clock evolved through a sequence of solvers, each a named, reproducible
configuration (best clean-box search wall; node counts carry parallel noise except where the
n = 14 deterministic figure is cited):

| solver                                      | n=16 wall | nodes  | mechanism                                                   |
|---------------------------------------------|-----------|--------|-------------------------------------------------------------|
| `iso-flat`                                  | 3m29s     | 6.1 B  | single selective-iso key over the flat lockless TT          |
| `iso-window`                                | 2m15s     | ≈5.1 B | dense `W8` tail table over a huge-page-collapsed flat TT    |
| `iso-dense` (W12, fused ETC)                | 1m32s     | 1.70 B | `getK` to ceiling 12 + fused enhanced-transposition cutoff  |
| `iso-dense` + dynamic move ordering         | 1m02s     | 1.14 B | re-sort children by current degree (Section 4.4)            |
| `iso-dense` (W16) + ordering + ETC          | ≈34s      | 0.40 B | pext code-build + ceiling raised 12 → 16 (Section 4.2/4.3)  |
| `iso-dense` (W17) + degree-ordered `getK`   | ≈24.5s    | 0.31 B | ceiling 17 (192-bit code), children swept degree-descending |
| `iso-dense` (W17) + ordering + `skip18`     | 23.44s    | 0.31 B | also skip TT work for the `pc = 18` band (Section 4.5)      |
| `iso-dense` W17 + killers + micro-ops (now) | 13.43s    | 0.18 B | cross-root killer replies + kernel micro-ops (Section 4.5)  |

For reference, the first complete n = 16 solve (a D₄-parallel search, no dense evaluator)
visited **10,017,867,872** nodes in ≈56 min on a thermally throttled box (≈42 min clean) — so
the current default (13.43 s, ≈1.79 × 10⁸ nodes; fastest single 12.43 s) represents roughly a
55× reduction in node evaluations and a ~250× reduction in wall time on the same hardware, with
the verdict (second-player win) unchanged throughout.

### 4.2 Raising the dense ceiling

The per-layer node-count reduction from raising `dense_k` is large and barely diminishing.
Deterministic single-threaded n = 14 node counts:

| ceiling | n=14 deterministic nodes | Δ vs previous |
|---------|--------------------------|---------------|
| W8      | 27,539,495               | —             |
| W9      | 22,527,149               | −18.2 %       |
| W10     | 18,825,047               | −16.4 %       |
| W11     | 15,724,135               | −16.5 %       |
| W12     | 12,896,443  (−53 % vs W8)| −18.0 %       |
| W13     | 10,339,019               | −19.8 %       |
| K = 16  | ≈4.0 M  (−50 % vs K=12)  | (K15→16 −22 %)|

(The `W8…W13` chain and the `K = 16` figure come from different measurement baselines: the
chain was measured with the earlier static move ordering, the `K = 16` row under the dynamic-
ordering default of Section 4.4, where `K = 12` measures ≈7.9 M — hence "−50 % vs K = 12"
rather than −69 % against the row above.)

Because the node-count cut is *inherent* (independent of table size), it holds at production
scale: at n = 16 with a 17 GB table, raising the ceiling to 16 collapsed the working set so the
table sat ~16.5 % full. The economically optimal ceiling moved over time as the per-node
evaluator got cheaper: when the evaluator was scalar, **K = 12** was the wall-clock optimum
(node cuts continued but per-node cost grew); the `pext` code-build (Section 4.3) made the deep
builders cheap and moved the optimum to **K = 16** (the u128 code ceiling, −35 % wall vs K = 12);
the 192-bit code path then made **K = 17** the wall sweet spot. Pushing further (W18–W20) keeps
cutting nodes (≈ −52 % at K = 20) but is *work-conserving* — deeper `getK` does the same
combinatorial work — so wall is flat above K ≈ 17. The node cut above 17 is nonetheless valuable
for **memory** (a smaller working set), which is exactly what n = 18 needed (Section 5).

### 4.3 `pext`-per-row code construction

Building the `k(k−1)/2`-bit adjacency code originally cost `k(k−1)/2` scalar bit-tests. Replacing
this with one 4-word BMI2 `pext` (`_pext_u64`) per vertex row (`adj_row_pext`) cut ≈3.8 %
cycles/node at n = 16, byte-identically — and, crucially, made deep ceilings affordable, which
unlocked the −35 % node-cut win of Section 4.2. (An earlier "pext is negative" measurement had
only tested a scalar reshape at tiny k; at K ≥ 12 scale the trade flips.)

### 4.4 Dynamic move ordering and ETC

The single largest search-shaping win was **dynamic move ordering**: at each node, re-sort the
children by their *current* available-block degree (`pc(child)` ascending — most-forcing first),
which surfaces instant wins (`child = ∅` sorts first) and reaches α-β cutoffs earliest. Against
the prior default this was **−34.3 % nodes / −30.2 % wall** at n = 16 (98.1 → 68.5 s), with only
+8.5 % cycles/node for the cheap sort; deterministic n = 14 confirmed −31.3 %. An **enhanced
transposition cutoff** (ETC) — probe the children's table entries before recursing, and cut on a
found loss — stacks on top for a further ≈ −18 % nodes. Move ordering is worth roughly a 2×
node reduction, a fact that recurs as the reason several throughput ideas failed (Section 4.6).

### 4.5 Branchless ordering, `skip18`, and other survivors

- **Branchless counting sort.** The move-ordering sort was the single largest branch-mispredict
  site (≈28 % of all mispredicts in a frontend-bound kernel). Replacing the comparison-based
  insertion sort with a count/prefix/stable-scatter counting sort (no data-dependent compare) was
  **−9.9 % cycles/node / −12.5 % wall**, byte-identical.
- **`skip18`.** For the `pc = 18` band specifically, skip *all* transposition work — the
  canonicalisation, the probe, and the put. This is safe and cascade-free because every child of
  a `pc = 18` node is a `getK` leaf, so a re-expanded `pc = 18` node re-runs one bounded
  evaluator sweep rather than an unmemoised subtree; the band is ~100 % cold anyway. Measured
  −2.5 % wall / −3.6 % cycles, n-agnostic, verdict-preserving. The band is *unique*: extending it
  to neighbouring popcounts measured net-negative. (At n = 18 the analogous skip covers the
  bands 18–25; Section 5.)
- **Flat `W0…W8` arena.** Concatenating the per-layer tables into one contiguous slice removed a
  serial bounds-check load in every `getK` leaf: −2.0 % cycles/node (it won precisely because it
  removed a *serial* load the out-of-order engine could not otherwise hide).
- **Warm-restart off.** A 2-second warm pass plus staggered restart had paid when the kernel was
  slower; once counting sort sped it up, the warm ramp stopped paying, so disabling it by default
  was −3.2 % wall. (Levers are re-tested after each win, because wins change what the next lever
  is worth.)
- **Cross-root killer replies** (the current record, −43 % wall over `skip18`). Each odd-ply
  prove-a-loss fan-out in the parallel upper tree *publishes* the square that refuted it; later
  root loops jump straight to an already-proven refuting reply instead of re-searching for one,
  and the killer table is re-read mid-loop. This turned out to be the cheap predictor that a
  prior saturation audit had judged missing for the wall-determining root's refuting second-ply
  reply — the proven replies from sibling roots *are* that predictor. Depth-1 A/B: **−37.6 %
  nodes / −43.3 % wall**; the deeper bands add −7.5 % / −4.5 %; an ETC pc-gate re-tested positive
  at the resulting ~8 % table fill (so a 12 GB table now beats the 17 GB one at n = 16). Verdict
  SECOND every round.
- **Kernel micro-optimisations** (−8 % cycles/node cumulative, all byte-identical): a
  `vpcompressb` (AVX-512-VBMI2) replacement for the ~9 %-of-cycles serial square-scatter
  (`verts_of`, −4.3 %); a one-ahead `getK` mask prefetch with a `TABLE_OFF` mask index (−2.1 %);
  and carrying the root adjacency into `getK` to skip a re-extraction (−2.5 %). Together with the
  killers these took the clean-box n = 16 record from ≈23.4 s to **13.43 s** (fastest single
  12.43 s) — a further reminder, on a search a prior audit had called "near-floor," that a
  measured limit is a hypothesis, not a bound.

Parallelism is parity-aware Young-brothers-wait: children are fanned out only at "prove-a-loss"
plies, where every child must be searched anyway, so there is **zero speculation** — this scaled
the search from ~1.4 to 24 cores with no added work.

### 4.6 Instructive negatives

For a methods audience the rejected levers are as informative as the wins; each was kept behind
a disabled flag with its measurement recorded.

- **Sorted-frontier "wave" pipeline (idle-core throughput).** A producer/consumer scheme that
  reorders the frontier into table-friendly order. Forfeiting move order cost **+94 % nodes** at
  n = 16 (the n = 14 proxy had lied at only +13.3 %) — confirming move ordering's ≈2× value. The
  pipeline is dead as built.
- **DFS parallelisation of the giant root tail (ABDADA in-flight markers; frontier
  work-stealing).** All variants *added* re-expansion (best-tuned work-stealing: +8.7 % nodes /
  +13.3 % wall), because the tail is **transposition-saturated**: the work that would fill idle
  cores is shared transpositions, not disjoint subtrees. The route to the tail is *not*
  parallelisation.
- **Component / nimber decomposition.** Splitting a disconnected position into components and
  XOR-ing their Grundy values cut nodes (up to −74 %) but cost **6.6× wall** — the root cause is
  Sprague–Grundy, not the implementation: component nimbers are cutoff-free (every move must be
  refuted) and the value-bearing components are sizes 9–12 (millions of distinct graphs, not
  tabulable). Across the dense layers, positions are overwhelmingly a single component, so the
  premise rarely fires. A companion census of graph *modules* over the search tail found the same
  emptiness one level down: twin pairs fall from ~4 % of vertices at pc = 13 to ~0 by pc = 18, and
  size-≥3 modules are entirely absent — the tail is module-free, so modular-decomposition
  reductions (and the FPT parameters built on them; Section 2.2) have nothing to fire on.
- **Memo-less `get17`.** A table-free K = 17 evaluator cut nodes −19.4 % but cost +30.7 %
  cycles/node / +5.7 % wall: the `pc = 17` subtree is shallow, so a memoised recurse beats a
  memo-less recompute (the opposite of the deep layers).
- **Set-associative TT, L0 probe-cache dedup, software-prefetch helpers, PGO, isolated-vertex
  pair-strip.** Each measured wash-to-negative on this single-box, memory-latency-bound workload
  (e.g. the entry probe is serial, so there is no memory-level parallelism for prefetch helpers
  to exploit). Several are parked for the *oversubscribed* small-table / large-n regime.

### 4.7 Memory and the transposition table

Capacity is not the binding constraint; per-probe DRAM latency is. A table-size sweep at n = 16
(8 / 12 / 17 GB, all fully huge-paged) showed warm throughput of 42.7 / 40.4 / 37.7 M nodes/s —
a larger table buys TLB residency but loses it back to eviction, so the curve is shallow.
Raising the dense ceiling shrinks the *working set* (the table sat ~16.5 % full at K = 16) but
not the resident footprint, because the table touches its full span via random page-spread
within seconds. Two memory mechanics matter:

- **Huge pages.** A randomly faulted multi-gigabyte table reaches only ~73 % 2 MB coverage under
  transparent huge pages; an explicit prefault plus `MADV_COLLAPSE` reaches 100 %, worth ~5 %
  wall and cutting startup from ~7 s to ~2 s.
- **Compact slot.** The 8-byte fingerprint slot (vs a full-key slot) cut the n = 14 resident TT
  from 5.4 GB to ~1.07 GB at the same slot count, with re-expansion essentially unchanged.

### 4.8 Benchmarking methodology

- **Interleaved A/B only.** The box thermally throttles within ~1 s of a ~12 s solve, so
  all-A-then-all-B comparisons fabricate deltas; we alternate the two binaries round-by-round.
  The n = 14 proxy can lie about direction (the wave pipeline above), so the **interleaved n = 16
  A/B** is the trustworthy measure, with deterministic n = 14 node counts for noise-free
  node-direction.
- **Metric discipline.** Cycles/node for byte-identical changes (node-count-independent); total
  cycles and wall for node-count-changing levers (where cycles/node rises by design).
- **Box hygiene.** Before any benchmark: compressed-RAM swap off (this box's "swap" is `zram`, a
  per-access decompress CPU cost, not disk), the filesystem cache cap lowered, page cache dropped
  and memory compacted, and the RAM-backed `/tmp` cleared — a degraded box once produced a
  spurious "floor" that clean hardware halved.

We treat any apparent performance limit as a measurement artefact or an untried lever until
proven otherwise; the figures here are milestones.

---

## 5. Solving n = 18

### 5.1 The capacity problem

The n = 16 optimisations are about *speed*; n = 18 is about *memory*. The proving search visits
on the order of 10¹¹ nodes, and on a 26 GB workstation (≈16 GB free) the binding question is
whether the transposition table can hold enough of the working set to avoid catastrophic
re-expansion. An initial unconfigured attempt thrashed: the table filled to 100 % and ~99.7 %
cold, and the root never converged.

The configuration that converged combines two ideas from Section 4:

1. **Band-skip transposition work for `pc ∈ [18, 25]`** (`QUEENS_SKIP18_PCS=18,…,25`). As with
   `skip18` at n = 16, these high-popcount bands are ~100 % cold and their children bottom out
   in `getK` leaves, so skipping their canonicalise/probe/put is verdict-preserving by
   construction (the value is still computed) and merely declines to memoise work that would not
   be reused. This frees the table to hold the lower, genuinely-reused bands.

2. **A 17 GB flat table** (`QUEENS_TT_SLOTS = 2.125 × 10⁹` 8-byte slots; ≈16.7 GB resident),
   sized to the box.

With this configuration the giant root I9 converged at ~10 M nodes/s. The runs are not
resumable (the flat table is not checkpointed); each proving run completed in a single ~7–8 hour
session on 24 worker threads (compiled `-C target-cpu=znver5`).

### 5.2 The result and its cross-validation

The verdict is established by **two independently configured proving runs** that differ in the
dense leaf evaluator they use, and that agree on every observable:

| run     | `dense_k` | `getK` code path | verdict     | root | nodes            | wall      |
|---------|-----------|------------------|-------------|------|------------------|-----------|
| primary | 17        | W17 (192-bit)    | first wins  | I9   | 258,322,944,571  | 8h16m45s  |
| confirm | 20        | W18/19/20 (≥190-bit) | first wins | I9 | 114,318,641,519 | 7h08m39s  |

Both runs report **n = 18 is a first-player win**, both identify **I9** (square 152) as a
winning opening, and both produce the **byte-identical 15-ply principal variation**

```
    I9  K8  G10  J11  H3  M7  N16  E4  P6  D12  O13  F2  R5  L17  A14
```

(squares `152, 136, 168, 189, 43, 120, 283, 58, 105, 201, 230, 23, 89, 299, 234`). The PV was
checked to be a sequence of 15 legal non-attacking moves ending in a terminal position with the
side-to-move unable to move — consistent with a first-player win (the first player makes the
last, 15th, move). The PV is *not* itself the proof: it certifies the legality, terminality, and
parity of one line, whereas the win claim rests on the exhaustive searches having refuted **every**
opponent reply after I9 — which is exactly what the two independently configured runs each did.

The two runs use different code (a 3-word code path vs a ≥190-bit path), exercise different
internal table dynamics, and converge at node counts differing by more than 2× — yet agree on
the verdict, the move, and the entire PV. Because the leaf evaluator is the component where this
class of solver has historically had bugs (Section 7.1), evaluating the same game two ways and
obtaining the same answer is the central evidence.

### 5.3 On the node count

The realised node counts (≈2.58 × 10¹¹ and ≈1.14 × 10¹¹) substantially exceed a pre-run estimate
(≈4.6 × 10¹⁰ central, from a 3-point extrapolation of n = 12/14/16). The gap is attributable to
**re-expansion**: the table cannot hold the full working set, so transpositions that would be
single entries in an unbounded table are recomputed. This is why the two configurations differ
so much in node count (the higher dense ceiling of the confirm run shrinks the working set and
roughly halves re-expansion) while agreeing on the value — and it is exactly the regime the
band-skip configuration was designed for.

### 5.4 An independent check on the principal variation, and its geometry

As a third correctness check — independent of *both* solver kernels — the 15-move PV was
re-verified by direct board arithmetic, with no search. All fifteen placements are pairwise
non-attacking and available when played, and after the winner's fifteenth move the available set
is *exactly empty*: the loser is left with no move, consistent with a first-player win (the first
player makes the odd-numbered last move). The per-move deletion schedule (available squares
remaining after each move) is

```
    I9:−68→256  K8:−56→200  G10:−44→156  J11:−46→110  H3:−22→88   M7:−25→63
    N16:−19→44  E4:−12→32   P6:−9→23     D12:−9→14    O13:−5→9    F2:−4→5
    R5:−2→3     L17:−2→1    A14:−1→0
```

The opening I9 deletes **68** squares — the maximum on an 18×18 board (its row and column, the
full-length main diagonal, and a length-17 anti-diagonal): the two central main-diagonal squares
are the most-forcing on any even board, so the winning strike is also the highest-degree opening
(which the degree-ordering of Section 4.4 already tries first).

The move also has a clean geometric reading that ties it to the structural theory of Section 6.
I9 = (8, 8) is the exact centre of the *embedded 17×17 sub-board* [0..16]², and the squares it
deletes are precisely the self-mirroring lines of the point reflection τ(x) = (16, 16) − x. The
strike therefore reproduces the *odd-board* centre-and-mirror structure (Section 6.4, Lemma 2)
inside that embedded odd sub-board, leaving only a 32-square live "L-border" (the last column and
last row, minus the two squares I9 attacks) as the region τ cannot pair. Consistent with this,
the winner's first reply **G10 is exactly τ(K8)** — the point-reflection of the loser's first
reply through I9. Later winner replies are *not* τ-mirrors, so pure mirroring is not the played
strategy (and opponent moves within a won line are ordering artefacts in any case); but the first
exchange, the record deletion count, and the fact that the only two PV squares on long diagonals
are I9 (main, the winning strike) and K8 (anti, the loser's reply) all match the picture the
even-board theorem draws (Section 6.4).

---

## 6. The Sprague–Grundy nimbers: extending A344227, and the structure of the even/odd split

The n = 18 verdict of Section 5 is an *outcome* (first player wins ⟺ nimber ≠ 0). The nimber
itself is a finer invariant, and two further threads sharpen the picture: a dedicated engine that
computes the exact nimber `G(n)` and thereby extends OEIS A344227 beyond its previously catalogued
range (Sections 6.1–6.3), and an elementary theory that explains the even/odd outcome split and
locates where the even → 0 pattern must break (Sections 6.4–6.8). The engine's results are
*cross-validated* to the same standard as the main solver. Throughout the theory sections we keep
three claim levels strictly apart: (i) the **proven theorem** — every first-player winning line on
an even board must eventually contain a long-diagonal move (Theorem 3); (ii) the **Forcing-Root
conjecture** — when a board is an N-position, the central diagonal strike `c*` in particular wins
(Section 6.6); and (iii) the **large-n heuristic** — winning roots tend central and diagonal.
Levels (ii) and (iii) never borrow (i)'s status. The geometric and spectral claims of Sections
6.6–6.7 are *computed* (exhaustive at the stated sizes, on independent brute-forcers
cross-validated against A344227); the general laws of Section 6.8 are *proven*; the "why n = 18"
mechanism and the exact value of G(18) are marked *heuristic* and *predicted*.

### 6.1 A heap-sum engine for the nimber

The Grundy value cannot be obtained by the boolean solver directly: `mex` admits no α-β cutoff, so
a full-DAG minimal-excludant reference must expand every reachable position — hopeless past
n ≈ 13. The engine instead uses the **heap-sum reduction**. By Sprague–Grundy, `G(board) = k`
iff the disjunctive game sum `board + Nim-heap(k)` is a P-position (a loss for the mover), because
`G(board + Nim-heap(k)) = G(board) ⊕ k`, which is 0 exactly when `G(board) = k`. Win/loss of that
*sum* **is** α-β-searchable. The driver solves `win(board, k)` for `k = 0, 1, 2, …` until the first
LOSS: since the sum is a P-position for exactly one `k` (namely `k = G(board)`), any round that
returns LOSS pins `G` exactly, and a round that returns WIN excludes that single value. One
transposition table is shared across rounds (the state `(avail, h)` keys `win` independently of the
round), and a `k = 0` LOSS is just the ordinary second-player win, so P-position boards cost one
plain solve.

The state `(avail, h)` carries the queen placements (which never change `h`) and the heap size
(a move may reduce `h` to any `h' < h`). Three evaluator layers resolve it: a **Grundy dense
leaf** (`pc ≤ gk`) that reads `G(avail)` from a new complete nimber table `GrundyW8` — the exact
Grundy value of every labelled graph on ≤ 8 vertices — extended by nested `mex` sweeps `g9…g16`
over the *same* projection geometry the boolean `getK` uses; a **boolean h = 0 leaf** (`pc ≤ bk`)
that reuses the production dense evaluator whenever only "is `G ≠ 0`" is needed; and a **deep α-β**
layer over the flat lockless table, heap moves probed first (an `h → 0` move is one dense lookup
and fires whenever `G(avail) = 0`), queen moves in the production dynamic order.

### 6.2 The extension: G(14) = 0, G(15) = 1, G(16) = 0, and G(17) = 2

A344227 was catalogued through n = 13 as `0, 1, 1, 2, 1, 3, 1, 2, 3, 1, 0, 1, 0, 1` (offset 0).
The engine adds the next four terms:

| n  | G(n) | how established                                                                           |
|----|------|-------------------------------------------------------------------------------------------|
| 14 | 0    | `k = 0` LOSS (1.4 s / 11.0 M nodes); independently equals the production SECOND verdict   |
| 15 | 1    | `k = 0` WIN + `k = 1` LOSS (23.8 s / 194 M nodes); reproduced at a different leaf ceiling |
| 16 | 0    | `k = 0` LOSS (2 m 21 s / 1.06 B nodes); equals the production n = 16 SECOND verdict       |
| 17 | 2    | `k = 0` WIN + `k = 1` WIN + `k = 2` LOSS; 17 GB table, about 585 B nodes / 59 h, verified 2026-07-07 |

so the sequence through n = 17 reads `0, 1, 1, 2, 1, 3, 1, 2, 3, 1, 0, 1, 0, 1, 0, 1, 0, 2`.
This confirms the OEIS-listed conjecture that for n ≥ 10 the nimber oscillates 0 (even) / 1
(odd) only through n = 16. It is **broken** at n = 17, where `G(17) = 2` contradicts the odd → 1
half, and independently at n = 18, whose first-player win forces `G(18) ≠ 0` (Section 5),
contradicting the even → 0 half.

The exact value of **G(18)** remains open. Since the production outcome already proves
`G(18) ≠ 0`, the exact-nimber plan can skip the `k = 0` reproof and fire `k = 1` first; a LOSS at
any `k ≥ 1` pins the value, while a WIN excludes that one value. The `h = 0` table-skip band and
the wide boolean leaf are what make such a round plausible on this box.

### 6.3 Validating the nimber engine

The engine carries its own layered validation, mirroring the main solver's:

- **The nimber tables against a scalar `mex` reference.** `GrundyW8` is checked against a pure
  scalar minimal-excludant recursion (exhaustively for ≤ 6 vertices, sampled at 7 and 8), and the
  identity `G ≠ 0 ⟺ boolean-win` is cross-checked against the independently built boolean `W`
  tables. The extension layers `g9…g16` are pinned to the validated `G ≤ 8` base by the same
  differential-test pattern the boolean `direct_w*` chain uses.
- **The engine against an independent full-`mex` reference and OEIS.** For n ≤ 8 the heap-sum
  engine is checked against the full-DAG `mex` reference, and the command-line runs for n = 1…13
  reproduce A344227 *exactly*.
- **The new terms cross-checked two ways.** G(14) and G(16) independently equal the production
  win/loss verdicts (a P-position is a `k = 0` LOSS by definition), G(15) was reproduced under a
  different Grundy-leaf ceiling (different leaf code paths, same value), and G(17) was verified
  after the initial long run on the n = 18 branch.
- **Production untouched.** The engine is additive: the full test suite and the n = 12 / n = 14
  distinct-count gates still pass.

A subsequent engine tune (2026-07-02) raised the boolean `h = 0` leaf ceiling to `pc ≤ 20`
(−63 % nodes at n = 15, because the `k = 0` round *is* a plain solve and inherits the production
`dense_k` lever) and the Grundy-leaf ceiling to 16 (a *wash* — the heap-sum wall is almost
entirely the `k = 0` round's `h = 0` search, since the `k ≥ 1` rounds ride the shared table, so
the Grundy leaves, which serve only `h > 0` states, have little wall leverage until the table is
oversubscribed at n ≥ 17).

The engine's soundness rests on the Sprague–Grundy sum theorem `G(board + Nim-heap(k)) =
G(board) ⊕ k` and the characterisation `win ⟺ G ≠ 0`. Both are exactly the results the Lean 4
development machine-checks in the graph model (`grundy_sum`, `win_iff_grundy_ne_zero`; Section
7.3) — so the *principle* the engine relies on is formally certified, while the engine's specific
board + heap instantiation is validated by the differential and OEIS-agreement chain above rather
than in Lean.

### 6.4 The structure of the even/odd split

The whole outcome split rests on one geometric fact. Let `ρ(r, c) = (n−1−r, n−1−c)` be the 180°
rotation of the board, and call a square **self-mirroring** if a queen on it attacks its own
`ρ`-image.

**Lemma 1 (self-mirroring squares) — proven.** A square `s = (r, c)` is queen-adjacent to `ρ(s)`
iff `s` lies on the main diagonal (`r = c`), the anti-diagonal (`r + c = n−1`), the centre row
(`2r = n−1`), or the centre column (`2c = n−1`). *Proof.* `s` and `ρ(s)` share a row iff
`r = n−1−r`; a column iff `c = n−1−c`; the main diagonal iff `r − c = (n−1−r) − (n−1−c) = c − r`,
i.e. `r = c`; the anti-diagonal iff `r + c = (n−1−r) + (n−1−c)`, i.e. `r + c = n−1`; and
queen-adjacency is exactly "shares a row, column, or diagonal." ∎ For **even** n, `2r = n−1` has
no integer solution, so there is no centre row or column and the self-mirroring set is exactly the
**two long diagonals** — which are disjoint for even n (they would meet only at a square with
`r = c` and `r + c = n−1`, i.e. `2r = n−1`), hence exactly **2n squares**; for **odd** n it is the
**four central lines** through the fixed centre cell. (`ρ` is moreover the *only* useful pairing symmetry: it is an involution with a small
self-mirroring set, whereas each D₄ reflection makes an entire row, column, or diagonal
self-mirroring, and the 90°/270° rotations are not involutions.)

**Lemma 2 (odd boards) — proven.** For odd n the first player wins, so `G(n) ≥ 1`. *Proof.* Play
the centre `c`. Its queen attacks the whole centre row, centre column, and both diagonals — by
Lemma 1 (odd case) exactly every self-mirroring square. So the residual `R` is `ρ`-symmetric and
contains no available self-mirroring square. The first player then *mirrors*: to any opponent move
`s`, reply `ρ(s) ≠ s`, which is available because the placed set is `ρ`-symmetric (any earlier
queen attacking `ρ(s)` would have a mirror image attacking `s` itself) and because `s`, being
non-self-mirroring, does not attack `ρ(s)` (Lemma 1). The first player
thus always has a reply and makes the last move, so `R` is a P-position and the root has a `G = 0`
option, giving `mex ≥ 1`. ∎

**Theorem 3 (even boards reduce to the diagonals) — proven.** For even n, from a `ρ`-symmetric
position (the initial board is one), if the player to move plays a **non-diagonal** (non-
self-mirroring) square `s`, the opponent can reply `ρ(s)` and restore a `ρ`-symmetric position:
by Lemma 1, `s` non-diagonal means `ρ(s) ≠ s` and `s` does not attack `ρ(s)`, so `ρ(s)` survives
`s`'s deletions, and removing `s` then `ρ(s)` (each with its attacks) is symmetric. Hence the
mirror strategy is a valid winning strategy for the *responder* against any line that never plays
a long-diagonal square. *Consequences:* **an even-board first-player win requires a long-diagonal
move at some ply** (avoid them forever and the responder mirrors and wins); equivalently,
`G(B_n) = 0` iff every long-diagonal deviation is refutable. In the strategy sense the even-board
outcome turns *entirely* on the 2n long-diagonal squares: the responder's certificate is a mirror
rule plus refutations of the diagonal deviations. ∎

**Scope — lines, not first moves.** The theorem constrains winning *lines*, not the winning
*opening*: it says every winning line must contain a long-diagonal move at some ply, not that the
first move must be diagonal. The distinction is computed fact, not pedantry: at n = 6 the
non-diagonal openings (1, 2) and (0, 2) are first-player wins — after (0, 2) and the mirror reply
(5, 3), the opener's only winning continuations are the anti-diagonal squares (1, 4)/(4, 1), i.e.
the mandatory diagonal strike arrives at ply 3 (Section 6.6). So the theorem collapses the even-n
*strategy* certificate, but it does not by itself license restricting a root search to the
diagonal opening classes; that stronger, root-level statement is precisely the (unproven)
Forcing-Root conjecture of Section 6.6, and "winning roots tend central and diagonal" is a
heuristic, one level weaker still. (One bookkeeping point: the four central cells of an even board
form a *single* class under the dihedral symmetry — the diagonal classes mod D₄ are exactly
`(d, d)`, `d = 0 … n/2 − 1` — so "the central class" is one candidate, not four.) The data remain
consistent throughout: the n = 18 winning move I9 = (8, 8) is on the main diagonal, and the
refutation of the even → 0 conjecture arrives exactly through the squares the theorem identifies
as the only escape from the responder's mirror. The PV geometry of Section 5.4 shows the same
structure concretely: I9 reproduces an odd-board centre-and-mirror on the embedded 17×17
sub-board.

### 6.5 Why n = 18, boundedness, and predictions

Theorem 3 makes the long-diagonal moves the crux of the even-n question — every winning line must
contain one — but does not say *when* a winning line first exists. The following is **heuristic,
not a theorem.** A central-diagonal
queen at `(d, d)`, `d ≈ n/2`, deletes its full row, full column, and both long diagonals — a
"cross + X" whose arms scale with n — and its residual is *not* `ρ`-symmetric (row `d` maps to row
`n−1−d`, only one of which is deleted), so the responder has no mirror and must out-play the first
player in a genuinely asymmetric position. As n grows the central strike controls a larger absolute
swath while the residual stays queen-dense, and at some size the first player's tempo outruns the
responder's ability to re-establish a losing symmetry; n = 18 is empirically that size. No clean
invariant (a monotone potential, a maximal-independent-set parity, a strategy-stealing argument)
is known that predicts the n = 18 threshold — and the even subsequence was never monotone to zero
anyway (`G(8) = 3`), so "eventually 0" was a fragile empirical pattern, now broken, not a law.

The "no clean invariant" observation can be sharpened into a small organising theorem: the even-n
verdict sequence N, N, N, P, P, P, P, N (n = 4 … 18) is non-monotone, so **no quantity monotone in
n can decide the even-board outcome**. Every natural scalar in the geometric family *is* monotone
in n — the central strike's deletion count `4n − 4`, the residual area `(n − 2)²`, the live border
`2(n − 2)`, the border/area ratio `2/(n − 4)` — so each is individually disqualified as a deciding
invariant; any true threshold criterion must be non-monotone (modular, spectral, or genuinely
game-tree-deep). The refutation-margin data of Section 6.6 make the same point empirically: the
central strike's fortunes do not improve smoothly with n even inside the second-player band.

**Boundedness is open, with a standing caveat.** Every known term is ≤ 3, but Node-Kayles Grundy
values are **unbounded on general graphs** (explicit constructions in the Arc-Kayles / vertex-
deletion literature), so *no* bound on the queen family is inherited — any bound must be a special
structural fact, earned rather than assumed. (Structured graph families — trees, bounded
neighbourhood-diversity — have provably eventually-periodic, hence bounded, Grundy sequences, but
queen graphs are dense and irregular and are covered by none of those theorems.)

The theory yielded one falsified prediction and one live one (the quantified priors live in the
project's research notes [11]; here we state only the structure). The odd-board theorem proves
`G ≥ 1`, and every computed odd n ≥ 9 had been 1 (n = 9, 11, 13, 15), so the pre-run prediction
was `G(17) = 1`; the verified value `G(17) = 2` is the first odd-side failure and a useful
calibration point. The live prediction is that `G(18)` is small, most plausibly **1**: a single
central-diagonal threat over an otherwise mirror-balanced remainder "looks like" a `*`-valued
(value-1) game, and the root-spectrum data of Section 6.6 make 1 the modal option value near even
roots. Because a LOSS round pins `G` exactly, the exact-nimber driver fires `k = 1` first. This is
a prediction, not a result.

### 6.6 The geometry of winning openings: spectra, margins, and the Forcing-Root conjecture

The computed claims in this and the next two subsections come from independent, from-scratch
Python brute-forcers (bitmask-memoised win/loss search plus a full-`mex` Grundy variant) —
independent of the production solver, the heap-sum engine, and the `naive` reference — which
reproduce every value they can reach (`G(5 … 10) = 3, 1, 2, 3, 1, 0`, A344227 exact, and the
outcome verdicts for n = 4 … 12) before being trusted on anything new. Full data, scripts, and
proofs are in the companion notes [11, 12].

**Which openings win.** Exhaustive enumeration of the winning opening classes (mod D₄), with
`c*` denoting the central diagonal class `(n/2 − 1, n/2 − 1)` — the unique-mod-D₄
maximum-deletion square of an even board (the centre plays that role for odd n):

| n  | verdict | winning opening classes (mod D₄)             | most-forcing square wins? |
|----|---------|----------------------------------------------|---------------------------|
| 4  | first   | c* = (1,1) and (0,0)                         | yes                       |
| 5  | first   | centre and (1,2)                             | yes                       |
| 6  | first   | (2,2) = c*, (1,1), (0,0), (1,2), (0,2)       | yes                       |
| 7  | first   | centre and (1,3)                             | yes                       |
| 8  | first   | (3,3) = c* only — the unique winning opening | yes, uniquely             |
| 9  | first   | centre, (2,4), (1,3), (0,1)                  | yes                       |
| 10 | second  | none                                         | vacuous                   |
| 11 | first   | every opening class wins                     | yes                       |
| 12 | second  | none                                         | vacuous                   |
| 18 | first   | I9 = c* (existential run; other roots open)  | yes                       |

The n = 6 row contains the counterexample of Section 6.4's scope note: the non-diagonal classes
(1, 2) and (0, 2) win, so Theorem 3 does not restrict first moves. The n = 8 row is the opposite
extreme: `c*` is the *only* winning opening.

**The Forcing-Root conjecture.** One pattern has never failed: **on every known first-player-win
board, the maximum-deletion (most-forcing) square is a winning opening** — the centre for odd n
(proven, Lemma 2), `c*` for even n (computed at 4, 6, 8; vacuously consistent at the second-player
boards 10, 12, 14, 16; and the n = 18 witness I9 *is* `c*`, the board maximum at 68 deletions).
Equivalently, conjectured: *a board is an N-position iff its most-forcing square wins.* This is a
conjecture, supported by all data — nothing yet excludes an even N-board whose `c*` is refuted
while some other root wins. Promoted to a theorem it would collapse the even-n *outcome* decision
to a single root solve; even unproven it is the strongest known root-ordering prior (and is what
the solver's degree ordering already tries first). It is a claim about *membership*, not
uniqueness: only n = 8 has `c*` as the sole winner.

**Root spectra mechanically explain the small nimbers.** The full option-value histograms at the
root (mod D₄) are startlingly thin:

| n  | root option-value histogram | mex = G(n) | mechanism                                 |
|----|-----------------------------|------------|-------------------------------------------|
| 9  | {0: 4, 3: 11}               | 1          | no option of value 1 or 2, so mex = 1     |
| 10 | {1: 15}                     | 0          | every opening has value 1, so mex = 0     |
| 11 | {0: 21}                     | 1          | every opening is a P-position, so mex = 1 |

G(10) = 0 because *all fifteen* opening classes have value exactly 1; G(11) = 1 because every
option is a P-position; G(9) = 1 despite value-3 options because values 1 and 2 are absent from
the spectrum. **Conjecture S** (from the n = 10 datum): for even P-boards n ≥ 10 every opening has
G = 1 — checkable one ply down with the heap-sum engine. Together with n = 11's all-zero spectrum
it suggested a two-level 0 ↔ 1 resonance near the root. The verified value `G(17) = 2` shows that
resonance is too tight on the odd side; `G(18)` remains the real exact-value probe.

**Refutation margins — a registered open puzzle.** The fraction of replies in `R_n` (the position
after `c*`) that refute the strike is violently non-monotone: 0/36 at n = 8 (`c*` wins), 8/64 at
n = 10, **100/100** at n = 12 (every reply refutes), 0/256 at n = 18 (`R` is again a P-position) —
even though `G(R_10) = G(R_12) = 1`. No smooth "the strike gradually strengthens with n" story
fits this; the margins at n = 14 and 16 are the cheap missing data points. (At n = 10 the four
interior refuters are exactly the knight-neighbours of `c*` on the strike's own diagonal side —
the nearest squares a queen does not attack — recorded as move-ordering fodder, not as a law.)

### 6.7 Border tempo, the τ-scar trajectory, and the torus as a borderless control

**The emerging decomposition.** After the central strike, `R_n` splits into three interacting
parts: the *embedded odd-board centre residual* (the sub-board `S = [0 .. n−2]²` after its centre —
literally `B_{n−1}` after Lemma 2's opening, hence a P-position on its own), the *live L-border*
(the last row and column, `2(n − 2)` live squares), and the *cross-attack entanglement* between
them. Write `τ(x) = (n − 2, n − 2) − x` for the point reflection of the sub-board about the
strike — the general-n form of Section 5.4's τ. All the n-dependent difficulty lives in the
border/scar entanglement: pure pairing cannot repair a border intrusion — a border square's
τ-partner is off the sub-board, and its transpose partner is self-attacked — so any future even-n
theorem needs inexact repair plus scar control, not a static mirror.

**Border ablations (exhaustive at n = 8, 10, 12).** Three modified games on `R_n` quantify that
frame:

| n  | `R_n` outcome    | border deleted | intruder border-banned   | responder border-banned  |
|----|------------------|----------------|--------------------------|--------------------------|
| 8  | P — `c*` wins    | P              | P — still no refutation  | N — the `c*` win fails   |
| 10 | N — `c*` refuted | P              | P — refutation collapses | N                        |
| 12 | N — `c*` refuted | P              | P — refutation collapses | N                        |

Deleting the border outright machine-verifies the embedded-odd-centre identity (column 3: the
sub-board part of `R_n` is exactly `B_{n−1}` after its centre, so Lemma 2 applies verbatim). The
decisive result is column 4: at n = 10 and n = 12, an intruder forbidden from ever playing a
border square **cannot refute `c*` at all** — every refutation of the central strike *requires*
intruder border tempo, a necessity established by exhaustion (upgrading what had previously been
only a tendency). The interior scar battle alone is not enough. Dually (column 5), at n = 8 it is
the *responder* who needs border access: banned from it, the `c*` win fails — the border battle is
genuinely two-sided. In all three extracted optimal lines the side that wins `R_n` makes the
*last border move*; border-tempo parity as the decision mechanism is conjecture-grade (three
lines, and loser moves in a won line are ordering artefacts), but it matches the n = 18 PV
exactly.

**The τ-scar trajectory of the n = 18 PV.** Per-ply accounting of Section 5.4's PV (Δ = the scar
set: live sub-board squares whose τ-partner is dead) shows the same shape in play [11]: exactly
one literally mirrored exchange (G10 = τ(K8), restoring perfect sub-board symmetry at ply 3),
after which the winner *abandons* τ even where the mirror reply was live — mirroring is the
opening posture, not the strategy; from ply 6 to the end every loser move lands inside Δ (the
loser is confined to scar squares, each move consuming scar); |Δ| after the winner's replies
descends monotonically after an early spike and reaches 0 on the winner's last move; and the
whole game contains exactly *one* border move — the winner takes the last live border square at
ply 13, denying the loser the border tempo the ablations show a refuter needs. One line, so this
is a computed trajectory and a conjectured mechanism, not a strategy extraction.

**The torus as a borderless control.** On the torus (rows, columns, and both diagonal directions
wrap), the queen graph is vertex-transitive; A344227's comments already record that the torus
game's value therefore lies in {0, 1} for every n — that fact is cited, not claimed. New here are
the computed terms:

| n         | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-----------|---|---|---|---|---|---|---|---|---|----|
| G(torus)  | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0  |

with **Conjecture T1: `G(torus_n) = n mod 2` for n ≥ 4.** The borderless board, in other words,
does exactly what the plane was conjectured to do — and the plane's even → 0 pattern broke at
n = 18 via structure the torus does not have. If T1 holds while the plane oscillation stays
broken, the border is *provably* the whole story, isolated as the difference between two concrete
sequences. The formally correct frame for any plane↔torus transfer is the proven embedding: the
n×n plane queen graph is an induced subgraph of the m×m torus queen graph for every `m ≥ 2n − 1`
[12] — though no *general* graph law can carry values across it (Section 6.8).

### 6.8 Two proven general laws, and the no-go results that frame them

Both laws are stated with proof sketches only; full proofs are in the companion note [12].

**Theorem S1 (Closed-Pairing) — proven.** Let `A` be a Node-Kayles position and `π : A → A` a
fixed-point-free involution (a perfect matching on the available set) such that (a) no pair is
internally adjacent (`π(v) ∉ N[v]`), and (b) every pair's joint closed neighbourhood is
π-invariant within `A` (`(N[v] ∪ N[π(v)]) ∩ A` is a union of π-pairs). Then `A` is a P-position:
`G(A) = 0`. *Sketch.* Induction on |A|: the responder answers `v` with `π(v)` (legal by (a)); the
joint deletion is π-invariant by (b), so π restricts to a closed pairing of the residual and the
responder always has a reply, moving last. ∎ S1 strictly generalises the mirror/Copying-Lemma
arguments of Section 6.4 — an involutive automorphism with no self-mirroring available square
satisfies (a) and (b) automatically — but the pairing **need not come from a symmetry at all**: it
is a purely combinatorial matching condition, checkable in `O(|A|³)` with no game search, so a
closed pairing is a complete, search-free P-certificate. It is sufficient, not a characterisation
(the 5-cycle has `G = 0` but, having odd order, admits no pairing). Where such certificates exist
is computed [12]: they exist for the n = 4 central-strike residuals; for the P-boards n = 10 … 16
their nonexistence on every opening residual is a *theorem* (those residuals are N-positions,
which S1 forbids pairing); and none of the n = 6/8 winning residuals admits one. Past the smallest
boards, winning defence therefore has irreducible adaptive strategy content — exactly where all
current proof attempts stop, and consistent with Section 6.7's conclusion that the border demands
inexact repair.

**Well-Covered Parity Law — proven.** If a graph is *well-covered* (every maximal independent set
has the same size `m`), then Node-Kayles on it has `G = m mod 2`, and every position's value is
its own remaining fixed play length mod 2. *Sketch.* Residuals of well-covered graphs are
well-covered with parameter `m − 1` (the maximal independent sets of `Γ ∖ N[v]` are exactly the
maximal sets through `v`, minus `v`); by induction every option has value `(m − 1) mod 2`, and the
mex is `m mod 2`. ∎ The classical rook-game solution `G = min(m, n) mod 2` is the special case
"rook graphs are well-covered". Queen graphs fail well-coveredness decisively — already at n = 4
maximal independent sets of sizes 3 and 4 coexist, and the spread widens with n — which is *why*
the parity route closes the rook game and cannot close queens: fixed game length is exactly the
well-covered property.

**Small exact laws and framing no-gos** (each recorded to stop re-derivation [12]):

- **True-Twin Deletion Lemma (proven).** If `N[u] = N[v]`, deleting `v` leaves every Grundy value
  unchanged (twins persist or die together, and duplicate options never change a mex). Verified on
  random graphs; consistent with the solver-side census that found the deep queen tail graphs
  essentially twin-free — the lemma has nothing to fire on there.
- **No Lipschitz edge-transfer law (computed no-go).** On random 9-vertex graphs a *single* added
  edge moved `G` by as much as 5, so no bound of |G(plane) − G(torus)| by wraparound-edge counting
  can exist as a general graph law; any plane↔torus transfer principle must use queen-specific
  structure.
- **Round cap (proven).** `G(position) ≤` the maximum remaining play length, so
  `G(B_n) ≤ α(Q_n) ≤ n`; in particular `G(18) ∈ [1, 18]`, and the ascending-`k` heap-sum driver of
  Section 6.1 never needs rounds `k > n`.
- **Small-defect value bounds are false (computed refutation).** The natural conjecture that a
  ρ-symmetric even-board position with exactly one available diagonal pair has `G ≤ 1` (D1;
  tested-consistent at n ≤ 8) fails at n = 10: exhaustive enumeration of the reachable game DAG
  exhibits `d = 1` positions with `G = 2` and `G = 3`, including cases whose diagonal pair is a
  live true twin with empty scar. The linear budget bound `G ≤ d` already fails at `d = 2` (n = 8,
  `G = 3`). Live diagonal-pair count therefore measures the *obstruction to the mirror pairing*,
  not the Grundy value, and no budget-style invariant can bound `G`. What survives is
  outcome-level: an empty-scar strike child is a P-position, so such positions are first-player
  wins; and boundedness statements must carry an explicit repair-strategy object (an
  oracle-conditional pairing theorem is proven in the companion notes) rather than a static
  position invariant.

---

## 7. Validation and verification

Correctness rests on a layered stack: a lineage agreement gate, exact distinct-count
invariants, differential tests against an independent scalar reference, an independent raw-mask
oracle on adversarial subpositions, an integer-width audit, reproduction of Jenrich's published
sequence, and a machine-checked Lean proof of the leaf evaluator's semantics. The motivating
defect class is described first.

### 7.1 The motivating defect: `u8` square-index truncation

The one real bug found in this code class was a **leaf-decode defect**. The migration from n ≤ 16
to n = 18 widened square indices from 8 to 16 bits across most of the code, but missed the
small-graph / component-canonicalisation path in one module, which still stored board-square
indices in `u8`. At n ≤ 16 the maximum square index is 255 and fits exactly; at n ≥ 17 squares
reach 323 and silently truncate (256 → 0, …), corrupting the adjacency rows, hence the code,
hence the looked-up value — a loss↔win flip. It was caught at `pc = 3` in milliseconds by the
independent-oracle differential (below) and fixed by widening the indices. This defect is the
direct motivation for the Lean verification of the leaf evaluator's decode and recurrence.

### 7.2 Test and cross-check hierarchy

- **Lineage agreement.** Every solver variant matches the memo-less `naive` recurrence's verdict
  for all n ≤ 9. `naive` is the ground truth the entire lineage is pinned to.
- **Exact distinct-count invariants.** `iso-flat` reports its exact distinct-position count; the
  n = 12 value is **1,060,823** (a second-player win) and the n = 14 value is ≈29.2 M with
  re-expansion ≈1.0×. A change in the distinct count signals a lost transposition merge; a jump
  in re-expansion signals an undersized table. Key/table changes must preserve both. (The ≈49.3 M
  figure sometimes quoted is the *D₄*-distinct count; `iso-flat`'s isomorphism key merges below
  it, so its own distinct count is ≈29.2 M — the figure the gate checks.)
- **Differential tests against a scalar reference** (in `dense.rs`). The optimised `pext`
  evaluator is compared bit-for-bit against a plain scalar recurrence across tens of thousands of
  density-spread codes per layer: `graph_wins8_matches_scalar` pins the `pext` `W8` build to the
  scalar build; `direct_w9..w16_matches_scalar_recurrence` pins the `getK` layers 9–16 to the
  `u16`-coded scalar reference `wins_rec`; and `direct_w17..w20_matches_scalar_recurrence` pins
  the wide layers — **including the W17/W18 leaves the n = 18 verdict bottoms out on** — to a
  3-word scalar reference `winsw_scalar`. Both n = 18 evaluator configurations are thus
  scalar-validated.
- **Independent raw-mask oracle.** A separate test drives ~3,400 18×18 subpositions whose live
  sets span the high-index words and checks both `iso-dense` and `iso-flat` against a *different
  implementation* — a raw-mask negamax with no `getK`, no canonicalisation, no table. This is the
  test that caught the truncation bug; it exercises exactly the high-square decode path that
  differs between n ≤ 16 and n = 18.
- **Integer-width audit.** A from-scratch read of the value path (mask words, geometry, the
  `d4_bits` bijection, the 128-bit hash, the code build, the wide W17–W20 path) for the
  truncation bug class found no defect that could flip an n = 18 value; the two residual findings
  were off the proving runs' configuration.
- **Reproduction of prior work.** The kernel reproduces Jenrich's full n ≤ 16 sequence
  (n = 16 second-player win) and agrees with OEIS A344227 through the catalogued range.

### 7.3 Machine-checked verification of the leaf evaluator (Lean 4)

The leaf evaluator is where the historical bug lived and where ~21 %+ of search nodes are
decided, so it is the target of a machine-checked proof. We follow a **"2-lite"** scope: prove
the *recurrence/algorithm semantics* in Lean, and leave the bit-level `pext`/serialization to the
differential tests of Section 7.2 (where they are cheapest to check). The development is a
self-contained Lean 4 + mathlib project (`lean/NodeKayles/`), `lake build` green with **no
`sorry`**; every theorem depends only on the three standard mathlib axioms
`[propext, Classical.choice, Quot.sound]` — no `sorryAx`, no `native_decide`, no custom axioms.

**What is machine-checked.** An abstract finite simple graph `Graph k` on `Fin k` with a
`closedNbhd` operation, and:

- `win` — the P/N recurrence of Section 2.1 — is **well-defined and terminating** (well-founded
  on `|S|`; the played vertex lies in its own closed neighbourhood, so the child set strictly
  shrinks). This mirrors the scalar reference `wins_rec`.
- `win_iso` — the value is **invariant under graph isomorphism** (same-size relabelling). This
  justifies the freedom to use any labelling of a position's vertices.
- `win_emb` — the value is **invariant under induced-subgraph relabelling**. This is the
  soundness of `projected_code`: the `getK` step that relabels a child's surviving vertices to
  `0 … k′` and reads the smaller `W{k′}` table cannot change the value.
- `buildPred_correct` — the **one-ply build recurrence equals the true value** (mirroring the
  table-build function `graph_wins`), with `not_win_empty` (the empty graph is a loss) as the
  `W0` base case.
- `mex`, `grundy`, and `win_iff_grundy_ne_zero` — the **Grundy characterisation**
  `win ⟺ grundy ≠ 0`, with `grundy` the minimal-excludant of the children's Grundy values.
- `grundy_iso` and `grundy_sum` — Grundy-value **isomorphism invariance** and the
  **Sprague–Grundy component sum** `grundy(S₁ ∪ S₂) = grundy S₁ ⊕ grundy S₂` when no edges run
  between the parts (built on mathlib's `Nat`-xor theory, e.g. `Nat.lt_xor_cases`).

The proofs were subjected to three rounds of adversarial review (integrity, statement
faithfulness, Lean↔Rust correspondence, mathematical adequacy, reproducibility). Adequacy was
corroborated against the literature: the path `P₃` computes to Grundy value 2 (Dawson's-chess /
OEIS A002187), and the isolated-vertex parity computes to `n mod 2`.

**What is deferred (the 2-lite boundary).** Not modelled in Lean, and carried by the differential
tests of Section 7.2: the u128 / 3-word **code bit-layout and its `pext` decode**
(`adj_from_code`, `projected_code`); the `pext` `W8` build fast path; the generic high-popcount
α-β combination logic above the dense ceiling (test-covered for n ≤ 16); and the board→code build
(the queen-graph construction). The **Lean↔Rust correspondence itself** — that the Lean
definitions faithfully mirror the Rust functions — is hand-audited, not machine-checked
end-to-end (auto-translation is not viable through `pext` intrinsics, const-generic
monomorphisation, and unchecked indexing). The move polarity is recorded explicitly: the Lean
`closedNbhd G v` (the deleted set `{v} ∪ N(v)`) corresponds to the Rust `(1<<i) | adj[i]`, and
the surviving child `S \ closedNbhd` to its complement `full & ¬((1<<i) | adj[i])`.

**Scope caveat.** `grundy_sum` and `grundy_iso` are **not** on the default `getK`/`iso-dense` path
and **not** part of the n = 18 verdict, which use only the boolean `win` recurrence. They do,
however, formalise the exact principle behind the heap-sum nimber engine of Section 6 — the
Sprague–Grundy component sum `G(S₁ ∪ S₂) = G(S₁) ⊕ G(S₂)` and `win ⟺ G ≠ 0` — so the *mathematics*
the engine relies on is machine-checked, even though the Lean statement is the graph-disjoint-union
instantiation rather than the engine's board + Nim-heap sum (whose specific instantiation is
validated empirically in Section 6.3). They also harden an optional, default-off component-nimber
decomposition lever (Section 4.6). The Grundy characterisation is included because it is the
cleanest formal statement of the leaf evaluator's meaning.

The combinatorial-game theory was, until recently, in mathlib's `SetTheory/Game/`; it has since
been extracted to a standalone library tracking an older Lean toolchain than this project's, so
the Grundy layer is built **self-contained** rather than anchored to mathlib's `Impartial` /
`grundyValue`. The consequence is that the statement "`win` *is* the game value" rests on the
standard-recurrence argument plus the literature cross-checks above, rather than on a cited
library theorem; an upgrade path (a `PGame` bridge) is documented for when the external library
matches our toolchain.

---

## 8. Threats to validity

We state the residual trusted base precisely.

1. **The n = 18 verdict is cross-validated, not formally certified.** A complete certificate of a
   10¹¹-node search is infeasible; the evidence is (a) two independently configured exhaustive
   searches agreeing on verdict + move + full PV, (b) both leaf evaluators differential-tested
   against an independent scalar reference, (c) an independent raw-mask oracle on thousands of
   high-index subpositions, (d) an integer-width audit, and (e) reproduction of Jenrich's n ≤ 16
   sequence. The single component validated *only* at n ≤ 16 (plus the two-run agreement) is the
   generic high-popcount α-β combination logic over the I9 subtree, above the dense ceiling — it
   is neither differential-tested at n = 18 popcounts nor in scope for the Lean proof. The two
   configurations exercise this logic over different node sets and agree.
2. **The Lean proof covers the leaf evaluator's recurrence semantics, not the whole search.** It
   does not model the bit serialization, the high-popcount α-β, the transposition table, the
   concurrency, or the board→code build, and the Lean↔Rust correspondence is hand-audited
   (Section 7.3).
3. **Benchmark numbers are single-machine and noisy.** All n = 16 wall/node figures carry ≈ ±18 %
   parallel node-count noise; deterministic n = 14 node counts are the noise-free measure.
   Throughput figures depend on a clean, huge-paged, non-throttled box.
4. **The fingerprint table admits probabilistic wrong hits** at ≈ 2⁻⁵⁵ per colliding probe;
   cross-checks against Jenrich's verdicts and the two-run agreement bound the practical risk.

None of these undermines the qualitative result; they delimit what "proved" means at each layer.

---

## 9. Conclusion and future work

We determined that the Non-Attacking Queens game on the 18×18 board is a **first-player win**,
witnessed by the opening move I9 and a 15-ply principal variation, by an exhaustive boolean
game-tree search whose verdict is corroborated by two independently configured runs. The result
extends Jenrich's n ≤ 16 **win/loss** sequence. The n = 18 *outcome* does not by itself contribute
a term to OEIS A344227 (the Sprague–Grundy *nimber*, catalogued only through n = 13), since an
outcome solve does not yield the nimber value — and an even-board first-player win in fact
contradicts that sequence's conjectured even → 0 oscillation. The nimber sequence *is* extended,
separately, by a heap-sum Sprague–Grundy engine that certifies the new terms G(14) = 0, G(15) = 1,
G(16) = 0, and G(17) = 2, so the conjectured even → 0 / odd → 1 oscillation is broken on the odd
side at n = 17 and on the even side by the n = 18 outcome; and an elementary 180°-rotation pairing
argument proves that every even-board first-player winning line must contain a long-diagonal move
— the winning n = 18 opening I9 = (8, 8) is a main-diagonal move, consistent with that theorem
(which constrains winning lines, not first moves; Section 6.6). Two companion theory studies add a
computed geometry of winning openings supporting the Forcing-Root conjecture, root option-value spectra that
mechanically explain the small catalogued nimbers, exhaustive ablations proving that refuting the
central strike requires intruder border tempo at n = 10/12, torus-queens nimbers computed to
n = 10 with a conjectured mod-2 law, and two new proven general laws — Closed-Pairing and
Well-Covered Parity (Sections 6.6–6.8).

The enabling techniques are a dense leaf evaluator that resolves the deepest fifth of the tree
directly from precomputed Node-Kayles tables, isomorphism-aware canonicalisation over a
lockless flat transposition table, dynamic move ordering, and a capacity configuration
(band-skipped transposition work + a 17 GB table) tuned to a single workstation. The leaf
evaluator's recurrence semantics — the component where this solver class has historically had
bugs — are machine-checked in Lean 4, kernel-complete and depending only on standard axioms,
with the bit-level serialization deferred to differential tests.

Directions for further work: computing **G(18)** — the exact oscillation-breaker whose value the
theory predicts to be 1, and each round of which is an n = 18-scale search; closing the residual
trusted base by modelling the u128 code decode in Lean
(removing the serialization from differential-test-only status) and bridging `win`/`grundy` to a
blessed `Impartial`/`grundyValue` once the external game-theory library matches the toolchain;
attacking the heuristic "why n = 18" threshold and the boundedness question (open, with no bound
inherited from general Node-Kayles); exporting third-party-checkable certificates for the computed
nimbers — adjacent work has produced checkable-certificate game solving and an empirical
verification theorem for chess tablebases, but no proof-assistant-verified nimber table exists in
the literature we know of, and our Lean layer (Section 7.3) is positioned to close that gap
[17, 18]; and a resumable, disk-backed transposition tier for n = 20.
The n = 16 search, already carried below 20 s by cross-root killer replies and kernel
micro-optimisation (13.43 s; Section 4.5), continues to reward levers a prior audit had called
terminal — we make no claim that any reported time is a floor.

### Outlook: n = 20 — a registered prediction, not a result

The next open even board is n = 20 (55 root classes mod D₄, of which 10 diagonal). Before any
search is sized or run, the theory of Sections 6.6–6.8 is placed on record as a **falsifiable
registered conjecture**: *n = 20 is a first-player win, with the central diagonal strike
(9, 9) (= J10) as a witness.* This is a structural ranking plus a conjecture, not a probability
claim (the research notes carry quantified priors [11]; the paper does not). The mechanism
reading: (9, 9) tops every geometric ranking simultaneously and by arithmetic fact — the board
maximum 76 deletions (uniquely, as the central class), mirror gap 1 (the diagonal square closest
to its own ρ-image), and the exact embedded-odd-centre structure (it is the centre of the 19×19
sub-board, precisely as I9 was of the 17×17), with a live L-border of 36 squares giving the most
striker-favourable border/area ratio yet (0.125 vs n = 18's 0.143); the intruder's only proven
refutation resource, border tempo (Section 6.7), keeps shrinking relative to the τ-paired
territory; and on every known N-board the most-forcing square has been a winner (Forcing-Root,
Section 6.6). The ranked root schedule an N-hunt should follow: **(9, 9)** first; then **(8, 8)**
(the gap-3 diagonal); then **(8, 9)** (the strongest non-diagonal central neighbour — the n = 6
data show near-central non-diagonal openings can win); then the remaining diagonals descending,
interleaved with the deletion-ordered non-diagonal band. Theorem 3 licenses none of this as a
restriction for a *P-proof* — refuting all 55 root classes remains the obligation if the board is
a second-player win; only a proven Forcing-Root law would collapse the N-decision to one root.
Cheap discriminators worth running before any campaign: the refutation margins and `G(R_n)` at
n = 14 and 16 (does Section 6.6's margin non-monotonicity smooth into a trend?); the Conjecture S
check at n = 12 (all root options G = 1?); a sizing probe for the τ-fold of the `c*` subtree; and
a border-first child-ordering A/B at refutation nodes. The confidence is deliberately tempered:
the refutation-margin puzzle proves the strike's fortunes are not smooth in n, so a second-player
win at n = 20 cannot be excluded by any invariant tested — and would be the single most
informative outcome available. Conditional on a first-player win, the same-mechanism reading
predicts `G(20) = G(18)`. All of this is prediction; none of it is a result.

### Reproducibility

The verdict runs are reproducible (modulo the non-resumable flat table) with the `iso-dense`
solver at the two configurations of Section 5.2 — the primary run sets `dense_k = 17`, the
confirm run `dense_k = 20`, both with the `pc ∈ [18,25]` transposition-skip and a 2.125 × 10⁹-slot
table — on a 26 GB Zen 5 workstation compiled for `znver5`. The validation gates (lineage
agreement, the n = 12 distinct count 1,060,823, the differential tests, the independent-oracle
subposition check) run from the project's standard test target. The nimber terms of Section 6,
including G(14) through G(17), are reproduced by the `queens nimber <n>` command (which prints the
ascending-`k` rounds and the first LOSS), whose own gates check the engine against the full-`mex`
reference for n ≤ 8 and against A344227 for n ≤ 13. The Lean development builds with Lean
v4.32.0-rc1 + mathlib via `lake build`; a
green build with no `sorry` warning is the gate, and `#print axioms` on each theorem yields the
standard axiom triple.

---

## References

1. T. Jenrich. *Successful strategies for a queens placing game on an n x n chess board.*
   arXiv:1312.5135, 2013/2014.
2. OEIS Foundation. *A344227: Sprague–Grundy values of the non-attacking queens placement game.*
   The On-Line Encyclopedia of Integer Sequences.
3. T. J. Schaefer. *On the complexity of some two-person perfect-information games.* Journal of
   Computer and System Sciences 16(2), 1978. (PSPACE-completeness of Node-Kayles.)
4. H. Noon and G. Van Brummelen. *The Non-Attacking Queens Game.* College Mathematics Journal
   37(3), 2006.
5. R. P. Sprague (1935); P. M. Grundy (1939); J. H. Conway, *On Numbers and Games*, 1976.
   (Sprague–Grundy theory of impartial games.)
6. D. Lemire. *Fast random integer generation in an interval.* (The multiply-shift `fastrange`
   table index.)
7. The mathlib Community. *mathlib4*, Lean 4 mathematical library. (`Nat` bitwise theory used by
   the Grundy proofs.)
8. OEIS A002187 (Dawson's chess / Grundy values of path Node-Kayles), used as a Lean adequacy
   cross-check.
9. On the unboundedness of Node-Kayles / vertex-deletion-game Grundy values over general graphs
   (via the "generalisation of Arc-Kayles" line, *International Journal of Game Theory*, 2018;
   arXiv:1709.05219). Cited in Section 6.5 for the standing caveat that no bound on the queen
   family is inherited.
10. K. Wong et al. *Nimber Sequences of Node-Kayles Games.* Journal of Integer Sequences 23,
    2020 (paths, lattices, prisms, hypercubes, generalized Petersen families); N. Songsuwan.
    *Node-Kayles on Trees.* arXiv:2512.24221 (eventual periodicity on regular-tree families).
    Context for Sections 2.2 and 6.5's periodicity/boundedness discussion.
11. *Winning geometry across the even boards, and the n = 20 candidate map.* Project research
    note, 2026-07-03 (`notes/2026-07-03-winning-geometry-n20.md`): winning-opening enumeration,
    root spectra, border ablations, the τ-scar trajectory, refutation margins, the n = 20 root
    rankings, and the quantified priors behind Section 9's registered prediction.
12. *CGT laws and tricks: new theorem directions beyond the mirror theory.* Project research
    note, 2026-07-03 (`notes/2026-07-03-cgt-laws-and-tricks.md`): full proofs of Theorem S1
    (Closed-Pairing), the Well-Covered Parity Law, the True-Twin Deletion Lemma, the
    plane-in-torus embedding, and the round cap; the closed-pairing existence census; the torus
    computation to n = 10; and the correction record for the retracted "non-diagonal openings are
    N-positions" inference.
13. K. Burke, M. Ferland, S.-H. Teng. *Winning the war by (strategically) losing battles:
    settling the complexity of Grundy values in undirected geography.* arXiv:2109.05622 /
    Theoretical Computer Science (nimber-preserving reductions; Generalized Geography is
    Sprague–Grundy-complete for polynomially-short impartial rulesets; the nimber-vs-outcome
    separation).
14. Y. Kobayashi. *On structural parameterizations of Node Kayles.* arXiv:2003.11775;
    T. Hanaka, H. Ono, K. Yoshiwatari. *Colored Node Kayles* (PSPACE-completeness on planar
    max-degree-3 graphs; W[1]-hardness in the number of turns; FPT by vertex cover /
    neighbourhood diversity / twin cover).
15. Z. Liu, Y. Liao, L. Wang. *Statistical mechanics of the N-queens problem.* arXiv:2605.10326,
    2026 (no bulk transition; tensor-network counting); M. Simkin. *The number of n-queens
    configurations.* arXiv:2107.13460 (the counting constant recovered there).
16. D. A. Klarner. *The problem of reflecting queens.* American Mathematical Monthly 74, 1967;
    T. Dai, T. Kelly. *On the existence of reflecting n-queens configurations.* Forum of
    Mathematics, Sigma (arXiv:2407.12742) — existence for all sufficiently large n; the finite
    small-n gap is open.
17. A. Pavlov. *Capture-Quiet Decomposition: a verification theorem for chess endgame
    tablebases.* arXiv:2604.07907, 2026 (empirical validation across three- to six-piece
    tablebases; not mechanised in a proof assistant).
18. K. Takizawa. *Semi-strongly solved: certificate-exporting game solving* (6×6 Othello, 7×6
    Connect Four). arXiv:2411.01029, rev. 2026; I. Shaik et al., QBF strategy validation,
    SAT 2023. Third-party-checkable certificates without formal verification — the adjacent
    lane for Section 9's certified-nimbers direction.
