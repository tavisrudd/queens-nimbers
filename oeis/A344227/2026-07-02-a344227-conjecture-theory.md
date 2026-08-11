# A344227 (Non-Attacking Queens / Node-Kayles on the queen graph) — conjecture theory

**Date**: 2026-07-02
**Scope**: THEORY-ONLY. No solver runs, no builds. Read-only research + this note.
**Author context**: A344227 is known through n=17 — `G(17) = 2`, computed on the box (heap-sum engine). `G(18)`'s exact value is still open.

This note formalizes the open conjectures around A344227, sweeps the literature, offers a
rigorous structural theory of the even/odd split (with proof status marked precisely),
records `G(17) = 2`, predicts the still-open `G(18)`, and lists cheap deferred experiments to discriminate the remaining hypotheses.

---

## 0. The data (established facts this note reasons over)

A344227 = Sprague-Grundy (nimber) values `G(n)` of Node-Kayles on the n×n queen graph
(offset 0; `G=0 ⟺ P-position ⟺ second-player win`).

| n    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|
| G(n) | 0 | 1 | 1 | 2 | 1 | 3 | 1 | 2 | 3 | 1 | 0  | 1  | 0  | 1  | 0  | 1  | 0  | 2  | ≠0 |

- **n=0..13**: the OEIS-listed terms `0,1,1,2,1,3,1,2,3,1,0,1,0,1` (verified against the
  `InnovativeInventor/node-kayles` repo output this session; that repo — Max Fan — is the
  computation behind the sequence and reached n=13, flagging n=10..13 `0,1,0,1` as new).
- **n=14,15,16**: `G = 0, 1, 0` — NEW, this project (heap-sum engine, 2026-07-01), multi-config
  validated; independently equal to the production win/loss verdicts.
- **n=17**: `G = 2` — **verified** (heap-sum engine: k=0 WIN, k=1 WIN, k=2 LOSS). First odd value
  `> 1` since G(7)=2; refutes the conjectured odd→1 continuation.
- **n=18**: `G ≠ 0` — this project's n=18 FIRST-player win (witness opening **I9 = (8,8)**,
  square 152; multi-config, 2026-06). The exact nimber value is open.

**The conjecture on the OEIS page** (as transcribed in this project's handoffs; I could NOT
re-fetch oeis.org this session — it 403s WebFetch on every format — so the *wording* is
project-sourced, but the *terms* are independently repo-verified): for `n ≥ 10`,
`G(n) = 0` if n even, `1` if n odd. The known terms confirm it through n=16; n=18 refutes its
even half (`G(18) ≠ 0`).

**Two things everyone should keep straight (recurring category errors in this project):**
1. A344227 is the **nimber** sequence to n=13. This project's n=14..18 headline results are a
   mix: n=14,15,16 are genuine **nimber** extensions; n=18 is a **win/loss OUTCOME** (nimber ≠ 0,
   value not computed). Only the nimber values extend A344227.
2. Dekking–Shallit–Sloane "Queens in exile" (ELJC 2020) is a **different game** (single token
   on an *infinite* spiral/antidiagonal board, Wythoff-like), NOT Node-Kayles on the finite
   queen graph. Its "every diagonal of the SG table is a permutation" conjecture is a
   community/venue pointer, not a statement about A344227. Do not transfer it.

---

## 1. The open conjectures, formalized (with proof status)

Notation: `B_n` = the initial n×n position; `ρ` = 180° rotation of the board,
`ρ(r,c) = (n-1-r, n-1-c)`; a square is **self-mirroring** if it is queen-adjacent to its own
`ρ`-image. Lemma 1 (§3) shows self-mirroring = "on a long diagonal" (even n) or "on one of the
four central lines" (odd n). `G` = Grundy value; `P` = `G=0`.

### (a) Odd n ⟹ G = 1 for n ≥ 11 (really n ≥ 9)

- **Formal**: `∀ odd n ≥ 9, G(n) = 1`.
- **Proof status**:
  - `G(n) ≥ 1` for ALL odd n is **PROVEN** (Lemma 2, §3: play the center, then mirror — the
    center queen deletes every self-mirroring square, so the responder's pairing strategy is
    globally valid; equivalently the post-center residual is a P-position, giving the root a
    `G=0` option, so `mex ≥ 1`). This is the standard odd-board pairing argument, made precise.
  - `G(n) = 1` **exactly** (not 2, 3, …) is **CONJECTURE**. `mex` equals 1 iff the root has a
    `G=0` option (proven: the center move) AND **no** `G=1` option. The second clause is
    unproven and is exactly where small odd boards differ (n=5→3, n=7→2, n=9→1 all have the
    center move to a P-position, but n=5,7 also reach `G=1`/`G=2` options that lift the mex).
    The conjecture asserts those higher-value options vanish for large odd boards. Data: stable
    at 1 for n=9,11,13,15.

### (b) Even → 0 held n=10..16, broke at 18 — structural reading

- **Formal (the surviving version)**: for even n, `G(n)=0 ⟺` the first player has **no**
  winning long-diagonal (self-mirroring) move — equivalently, second player's mirror strategy
  is never profitably broken. `G(n) ≠ 0 ⟺` some long-diagonal opening (or later diagonal
  deviation) wins for the first player.
- **Proof status**: the **reduction to the diagonals is PROVEN** (Theorem 3, §3): for even n
  the mirror (`ρ`-pairing) strategy wins for the responder against any line that stays off the
  long diagonals; hence a first-player win *requires* a self-mirroring (diagonal) move. The
  n=18 winning move I9 = (8,8) is on the main diagonal (verified by direct arithmetic), and all
  four central cells of any even board lie on a long diagonal (two main, two anti) — so the
  refutation of the even→0 conjecture happens exactly at a central-diagonal square, as the
  theory predicts.
- **What is NOT proven**: *why n=18 and not n≤16*. The framework says the whole even-n question
  is "are the diagonal openings all refutable?"; that they are for n=10..16 and one is not for
  n=18 is an empirical fact with a plausible-but-heuristic reading (§3.4): a central-diagonal
  queen deletes a full row + column + two long diagonals, and its residual asymmetry grows in
  reach with board size; n=18 is where a central-diagonal opening's residual first becomes a
  P-position. This is a heuristic, not a theorem. Note also the even subsequence is **not**
  monotone-to-zero: even values are `G(0,2,4,6,8) = 0,1,1,1,3` then `G(10..16)=0` then
  `G(18)≠0` — a nonzero already occurred at n=8, so "eventually 0" is a fragile empirical
  pattern that this project has now broken, not a law.

### (c) Is G(n) bounded? (all known ≤ 3)

- **Formal candidates, strongest first**:
  - **C1 (aggressive)**: `G(n) ∈ {0,1}` for all `n ≥ 9` — **FALSE**: `G(17) = 2`.
  - **C2 (conservative)**: `G(n) ≤ 3` for all n. Matches every known term (max 3 at n=5, n=8).
  - **C3 (weakest)**: `G(n)` is bounded.
- **Proof status**: all three are **CONJECTURE**. Crucial literature caveat (§2): Node-Kayles
  Grundy values are **unbounded on general graphs** (explicit constructions), so **no** upper
  bound is inherited — any bound on the queen family is a special structural fact that must be
  earned, not assumed. Weak positive evidence: on *structured* graph families (trees, bounded
  neighborhood-diversity / modular-width) Grundy sequences are provably eventually periodic,
  hence bounded per family; queen graphs are dense/irregular and covered by none of these
  theorems, so this is only suggestive.

### (d) Sharper statements I can state and defend

1. **PROVEN**: `G(n) ≥ 1` for all odd n (Lemma 2).
2. **PROVEN**: the only board automorphism that yields a useful pairing strategy is `ρ`=180°
   rotation, and its self-mirroring squares are exactly the long diagonal(s) (Lemma 1 + §3.1).
   The four D4 reflections each have a *full row/column/anti-diagonal* of self-mirroring squares
   (useless); the 90°/270° rotations are not involutions.
3. **PROVEN (Theorem 3)**: for even n, second player's mirror strategy wins against any
   diagonal-free first-player line ⟹ a first-player win requires a diagonal move; the winning
   n=18 line realizes this on move 1 (I9 on the main diagonal).
4. **REFUTED by `G(17) = 2`**: the "unified crisp bet" that `G(n) ∈ {0,1}` for `n ≥ 9` with
   `G(odd) = 1` always. `G(17) = 2` breaks both the `{0,1}` bound and the odd→1 rule at once. What
   survives is C2 (`G(n) ≤ 3`); whether the even values stay `0` except at a sparse
   "diagonal-breakthrough" set (n=18 the first) is still open, with `G(18)` the next probe.

---

## 2. Literature sweep

Legend: **[V]** = verified this session against a primary/secondary source I read;
**[V-sum]** = verified only via a search-engine summary (not the primary text);
**[R]** = recalled / project-sourced, not independently re-verified this session.

### 2.1 The sequence and its game

- **A344227 itself** — Sprague-Grundy values of Node-Kayles on the n-queens graph, terms
  `0,1,1,2,1,3,1,2,3,1,0,1,0,1` (n=0..13). **[V]** terms cross-checked against the
  `InnovativeInventor/node-kayles` repo (which computes exactly this and marks n=10..13 novel).
  Author attribution Max Fan / M. Bardoe (2021) is **[R]** (project note; oeis.org unreachable
  this session — 403 on WebFetch for the page, the `fmt=text` view, and the b-file).
- **The non-attacking queens *game*** (place non-attacking queens, last to move wins = Node-
  Kayles on the queen graph). First proposed/studied by **Noon & Van Brummelen (2006)**; a paper
  literally titled "The Non-Attacking Queens Game" exists (ResearchGate 269909193). **[V-sum]**
  (attribution from search summaries; primary not read). Huggan and Nowakowski appear in the
  adjacent CGT literature (weighted Arc-Kayles etc.). **[V-sum]**
- **Dekking–Shallit–Sloane, "Queens in exile", ELJC 27(1) #P1.52 (2020), doi:10.37236/8905**,
  arXiv:1907.09120. **[V]** exists, authors/venue confirmed. **Different game** (single token,
  infinite board, Tribonacci-word solution; connections to CGT and an SG-table-permutation
  conjecture). Venue/community model for an eventual A344227 write-up, NOT a source of theorems
  about A344227.

### 2.2 Node-Kayles general theory (the load-bearing citations for §1c)

- **PSPACE-completeness**: deciding the winner of Node-Kayles on general graphs is
  **PSPACE-complete** (Schaefer, 1978, "On the complexity of some two-person perfect-information
  games"). **[V-sum]** So there is no expected closed form for A344227 in general; per-family and
  per-n computation is the state of the art. This is why the project's approach (compute
  outcomes/nimbers directly) is the right register.
- **Grundy values unbounded**: Node-Kayles Grundy values are **unbounded over general graphs**
  (established via explicit constructions in the Arc-Kayles / vertex-deletion-game literature,
  e.g. the "A generalization of Arc-Kayles" line, Int. J. Game Theory 2018, arXiv:1709.05219).
  **[V-sum]** (multiple search summaries agree; I could not extract the theorem from the PDF —
  it fetched as unreadable binary). **Consequence for §1c**: any boundedness on the queen family
  is NOT inherited from Node-Kayles in general — it must be a special property of queen graphs.
- **Eventual periodicity on structured families**: "Node-Kayles on Trees" (Songsuwan et al.,
  arXiv:2512.24221, 2025) proves the Grundy sequences of n-regular trees and two-tree-plus-path
  graphs are **eventually periodic**, with computable preperiod/period; explicit formulas/
  recursions. **[V]** (abstract read). Also "Nimber Sequences of Node-Kayles Games" (Brown,
  Daugherty, Fiorini, et al., 2020, NSF PAR 10141270) gives explicit nimber recursions for
  paths/lattices/prisms/cliques/cycles/hypercubes/**generalized Petersen** graphs — but **not**
  the queen graph. **[V]** (page read; queen graph absent). Takeaway: the field gets clean
  eventually-periodic answers on *sparse/structured* families; the queen graph's density and
  irregularity are exactly why it resists and needs brute nimber computation.
- **Structural parameterizations**: Node-Kayles is FPT by **neighborhood diversity /
  modular-width** (Kobayashi; "On Structural Parameterizations of Node Kayles",
  arXiv:2003.11775, 2020/2021). **[V-sum]** Relevant to the solver (the twin/module-reduction
  lever) but the project already measured that queen tail-graphs carry essentially no size-≥3
  modules (2026-06-20 probe), so modular-width is large here — consistent with "queen graphs are
  the hard, dense case."
- **Generalized Geography is Sprague-Grundy-complete** and there are **nimber-preserving
  reductions** between impartial rulesets (arXiv:2109.05622; ScienceDirect S0304397524002512).
  **[V-sum]** Theoretically interesting (any A344227 value is realized by some Geography
  position) but no computational leverage for specific n.

### 2.3 What the literature does NOT give us

No published theory predicts A344227's values, its parity pattern, or a bound. The even→0/odd→1
pattern is a *conjecture attached to the OEIS entry*, not a theorem anywhere — and this project's own `G(17) = 2` refutes its odd half. So the project's
n=14..18 results are genuinely at the frontier, and the theory in §3 is, to my knowledge, new.

---

## 3. Theory: why even n=10..16 are P-positions and n=18 is not

The whole section rests on one clean geometric fact.

### 3.1 Lemma 1 (self-mirroring squares). — PROVEN

Under `ρ(r,c)=(n-1-r,n-1-c)`, a square `s=(r,c)` is queen-adjacent to `ρ(s)` **iff** s lies on
the main diagonal (`r=c`), the anti-diagonal (`r+c=n-1`), the center row (`2r=n-1`), or the
center column (`2c=n-1`).

*Proof.* `s` and `ρ(s)` share a row iff `r=n-1-r`; a column iff `c=n-1-c`; the main diagonal
iff `r-c=(n-1-r)-(n-1-c)=c-r` i.e. `r=c`; the anti-diagonal iff `r+c=(n-1-r)+(n-1-c)` i.e.
`r+c=n-1`. Queen-adjacency is exactly "shares a row, column, or diagonal." ∎

**Even n**: `2r=n-1` and `2c=n-1` have no integer solution, so there is no center row/column and
no `ρ`-fixed cell; the self-mirroring set is exactly the **two long diagonals**.
**Odd n**: the center row, center column, and both diagonals all pass through the single
`ρ`-fixed cell `((n-1)/2,(n-1)/2)`; the self-mirroring set is those **four central lines**.

### 3.2 Why ρ is the only useful pairing symmetry. — PROVEN

A Hamming-style pairing (mirror) strategy needs an **involution** automorphism with **few**
self-mirroring squares (a self-mirroring square is un-pairable: playing it deletes its partner).
The board's symmetry group is D4. Its order-2 elements are `ρ`=180° and the four reflections;
90°/270° are order 4 (no pairing). For a reflection across a row axis, every square shares its
row with its image ⟹ *every* square is self-mirroring (useless); likewise column- and
diagonal-axis reflections have a full line of self-mirroring squares. Only `ρ` has a small
self-mirroring set (Lemma 1). (Aut(queen graph)=D4 for n≥ small is **[R]**; even if a
non-geometric involution existed, the argument below only uses `ρ`.) ∎

### 3.3 The odd/even split — the core results

**Lemma 2 (odd n ⟹ first player wins, G ≥ 1). — PROVEN.**
First player plays the center `c`. The center queen attacks its entire row, column, and both
diagonals = exactly the four central lines = every self-mirroring square (Lemma 1, odd case).
So in the residual `R` there is no available self-mirroring square, and `R` is `ρ`-symmetric.
Now the *responder* (first player) can mirror: for any second-player move `s`, `ρ(s) ≠ s` and
`ρ(s)` is available (its removal would require, by `ρ`-symmetry of the placed set, `s` to be
unavailable). So first player always has a response and moves last ⟹ `R` is a P-position,
`G(R)=0`. Hence the root has a `G=0` option ⟹ `G(B_n) = mex(...) ≥ 1`. ∎

**Theorem 3 (even n: the game reduces to the diagonals). — PROVEN.**
For even n, define a position "symmetric" if it is `ρ`-invariant (`B_n` is). Claim: from a
symmetric position with player X to move, if X ever plays a **non-diagonal** (non-self-mirroring)
square `s`, the opponent can reply `ρ(s)` and return to a symmetric position. (`s` non-diagonal
⟹ `ρ(s)≠s` and, by Lemma 1, `s` does not attack `ρ(s)`, so `ρ(s)` survives X's move; removing
`s`+attacks then `ρ(s)`+attacks is `ρ`-symmetric.) Therefore the mirror strategy is a **valid
winning strategy for the responder against any line that never plays a long-diagonal square** —
the responder always has a reply and moves last.

Consequences:
- **A first-player win for even n requires a self-mirroring (long-diagonal) move.** If first
  never plays one, second mirrors and wins. (Rigorous. The winning move need not be move 1 in
  general, but the win cannot avoid the diagonals entirely; n=18 realizes it on move 1.)
- Symmetric restatement: **`G(B_n)=0` for even n ⟺ every diagonal deviation is refutable**
  (non-diagonal moves are auto-refuted by mirroring). The even-n outcome is decided *entirely*
  by the long-diagonal moves. ∎

This is, I believe, a new and genuinely useful theorem: it collapses the even-n outcome question
onto the `O(n)` long-diagonal squares (≈ n/2 distinct mod D4). It also explains the empirics: the
n=18 winning move **I9=(8,8) is on the main diagonal** (verified), and the four central cells of
*any* even board are all on a long diagonal (two main: (8,8),(9,9); two anti: (8,9),(9,8) — all
verified for n=18). The conjecture's refutation lands exactly where the theory says the only
threats live.

### 3.4 Why n=18 and not n≤16 — HEURISTIC (explicitly not a proof)

Theorem 3 says the even-n question is "is some central/long-diagonal opening winning?" It does
not say *when* one becomes winning. A plausibility argument:

- A central-diagonal queen at `(d,d)` (d≈n/2) deletes its full row, full column, and both long
  diagonals through it — a "cross + X" whose arms scale with n. Its residual is a large board
  minus this cross, and it is **not** `ρ`-symmetric (row d maps to row n-1-d, only one is
  deleted), so second player has no mirror. Second must instead out-play first in a genuinely
  asymmetric ≈P-hunt.
- As n grows, the central-diagonal queen controls a larger absolute swath while the residual
  stays "queen-dense," and at some size the first player's tempo from this central strike first
  outruns the second player's ability to re-establish a losing symmetry. n=18 is empirically that
  size. (n=8's `G=3` shows even boards *can* be first-player wins at small sizes too — the P-band
  n=10..16 is a middle regime, not a permanent law.)
- I have **no** clean invariant (potential function, parity count of maximal independent sets,
  strategy-stealing) that predicts the n=18 threshold. Attempts and why they don't close it:
  - *Parity of maximal independent sets*: the total game length parity would decide the winner if
    all maximal independent sets had the same size parity — they don't for queen graphs, so this
    gives nothing.
  - *Strategy-stealing*: there is no "extra move is never bad" monotonicity in Node-Kayles
    (placing a queen removes options for BOTH players), so standard strategy-stealing does not
    apply; the center-steal works for odd n only because it *deletes the diagonals*, not by
    monotonicity.
  - *Potential function*: none found that is monotone under queen placement and parity-linked.
- **Status**: the reduction (Theorem 3) is proof; the n=18 threshold is heuristic + one data
  point. Treat "even→0 breaks at 18" as an empirical discovery, not an explained one.

### 3.5 The value (not just outcome) — HEURISTIC

- Odd n: `G ≥ 1` proven; `=1` iff the root has no `G=1` option. A central-diagonal-free
  symmetric residual "looks like" a single un-mirrorable degree of freedom, i.e. a `*`-like
  (value-1) game — consistent with n=9,11,13,15 all being 1, and with the conjecture. Not proven.
- Even n with `G≠0` (n=18): the winning structure is "one central-diagonal threat over an
  otherwise mirror-balanced board." A single dominant threat over a P-like remainder smells like
  `*` (value 1) or at most `*2`. This is the basis for the G(18) prior below. Not proven.

---

## 4. Predictions (falsifiable, with confidence + operational guidance)

### 4.1 G(17) = 2 (computed)

- **Value**: `G(17) = 2` (heap-sum engine: k=0 WIN, k=1 WIN, k=2 LOSS) — the first odd value `> 1`
  since G(7) = 2, and the first odd-side failure of the `G(odd) = 1` rule.
- **Why odd→1 broke here**: odd ⟹ `G ≥ 1` is proven, and odd n = 9,11,13,15 were all `1`, but that
  was empirical, not a theorem — it broke the same way even→0 broke at n = 18. `G(17) = 2` stays
  within the observed range `≤ 3` (C2), so the boundedness question is untouched; only the `{0,1}`
  bound and the odd→1 rule fall.

### 4.2 G(18) — the pending multi-day computation

`G(18) ≠ 0` (proven by the first-player win). Distribution I would bet:

| value | P     | reasoning                                                                    |
|-------|-------|------------------------------------------------------------------------------|
| 1     | ~0.55 | single central-diagonal threat over a mirror-balanced remainder ≈ `*`; keeps `G∈{0,1}` for n≥9 (C1) intact — the cleanest world |
| 2     | ~0.30 | two independent threat-parities; still ≤3; breaks C1 but not C2 (`≤3`)       |
| 3     | ~0.12 | matches the historical max (n=5, n=8 both hit 3); even boards have reached 3  |
| ≥4    | ~0.03 | never observed; would refute even C2 and reshape the whole picture           |

### 4.3 Operational recommendation for the ascending-k engine — this saves days

The engine tests rounds `k=0,1,2,…`: round k asks "is `B_18 + *k` a P-position?" (one
n=18-scale search). Key facts about the search economics:
- `B_18 + *k` is a P-position for **exactly one** k (namely `k = G(18)`); it is an N-position for
  every other k. So **any** round that returns **LOSS pins `G` exactly**, regardless of order;
  a round that returns **WIN only excludes that single k**.
- `k=0` is already known WIN (the production first-player win) — do **not** re-run it; `--min-k 1`
  is correctly set.
- Therefore the optimal policy is **test the most-probable value first**. My prior peaks at 1.

**Recommendation: run `k=1` first.**
- ~55% chance it returns LOSS ⟹ `G(18)=1` proven in **one** ~1.5–2 day search. Done.
- If `k=1` returns WIN (⟹ `G ≥ 2`), run `k=2` next (now the mode of the conditional prior),
  then `k=3`. Expected number of searches ≈ `1·0.55 + 2·0.30 + 3·0.12 + 4·0.03 ≈ 1.63`.
- **Do not** speculatively run `k=2` before `k=1`: a `k=2` LOSS would pin `G=2` (fine), but a
  `k=2` WIN tells you nothing you didn't need `k=1` for, and you'd have burned a multi-day search
  on the less-likely branch. Ordering by descending prior is the whole game when each test costs
  days.
- **Load-bearing engine note** (from the nimber handoff): the `h=0` TT-skip band and `bk=20`
  boolean leaf are what make an n=18-scale round converge on this box; the un-skipped n=18 solve
  thrashed. Any k≥1 round must keep those on. No checkpoint/resume exists for the sum engine — a
  multi-day round dies unrecoverably, which *raises* the value of firing the right k first.

**Bottom line**: a confident `G(18)=1` prior is worth ~2 days of box time (one search vs two).
The theory (§3.5: single central-diagonal threat ≈ `*`) and the all-values-≤3 history both point
to 1, so `k=1` first is the recommendation.

---

## 5. Cheap deferred experiments (DO NOT RUN — for a later session with the box free)

All use the existing `queens nimber` engine, mostly on **small boards** or **one ply down**, and
each is chosen to discriminate a specific hypothesis. Ranked by information-per-cost.

1. **Grundy value of each central/long-diagonal opening, even n = 10,12,14,16,18.**
   Compute `G(B_n after placing a central-diagonal cell)`, e.g. `(n/2-1, n/2-1)`.
   - *Predicts*: `G ≥ 1` for every diagonal opening at n≤16 (all refutable ⟹ board is P);
     `G = 0` for the winning diagonal opening(s) at n=18 (I9's residual is a P-position).
   - *Confirms/refutes*: Theorem 3's picture and pins the exact n where a central-diagonal
     opening first becomes winning. Cheap: one ply down, and for the losing cases the round is
     an ordinary sub-position solve.
2. ~~**Validation cross-check: `G(non-diagonal opening) ≥ 1` for even n, all n.**~~
   **RETRACTED 2026-07-03 — the inference was wrong and the claim is FALSE at n=6.**
   Theorem 3 constrains winning *lines* (every winning line contains a long-diagonal move),
   not the *first* move: a non-diagonal opening can win by striking a diagonal later. Computed
   counterexample: at n=6 the non-diagonal openings (1,2)/(0,2) win, i.e. their residuals have
   G=0 — verified on two independent brute-forcers (see the 2026-07-03 winning-geometry and
   cgt-laws notes, which found it independently). Do NOT wire this in as a correctness gate.
   Theorem 3 itself is intact.
3. **Odd-n residual after the center is a P-position.** Compute `G(B_n after center)` for
   n=9,11,13,15,(17).
   - *Predicts*: `= 0` every time (Lemma 2's residual claim). Machine-confirms the proven half of
     the odd→win argument and shows the root's `G=0` option explicitly.
4. **Root option-value multiset for odd n, to probe the exact-1 conjecture.** With a reporting
   hook (one extra pass), list `{G(child)}` at the root for n=7 (G=2) vs n=9,11 (G=1).
   - *Predicts*: for n≥9 the root has a `G=0` option and **no** `G=1` option (⟹ mex=1); n=7
     should show a `G=1` option present (⟹ mex≥2). Directly tests §1a's unproven clause and would
     show *why* the value drops to 1.
5. **Rectangular boards (needs engine extension; currently square-only).** A ρ-symmetric m×n
   board has a center cell iff both m,n odd. Probe m×n with mixed parity:
   - even×even: two "long diagonals" but of a rectangle — does the diagonal-threat mechanism still
     govern? odd×even: a center *row* exists but not a center column — does deleting the center row
     (via a center-row queen) recover an odd-like pairing?
   - *Discriminates*: whether the even/odd split is about **board parity** or specifically about
     **square-board diagonal geometry**. High-value but requires engine work (mark as gated).
6. **G of "board minus a fixed central region."** Pre-remove the 4 central cells (all diagonal)
   and compute the residual Grundy for even n.
   - *Predicts*: if central-diagonal control is the whole story, removing the central cells should
     re-symmetrize the game toward P; comparing n=16 vs n=18 residuals localizes the n=18 effect
     to the center. Cheap (pre-placement + one solve).

Each experiment is small relative to a full n≥17 nimber round and can be delegated to a
background session; none requires the 17 GB production TT.

---

## 6. One-paragraph summary of proof status (for quick reference)

**Proven**: self-mirroring squares = long diagonals (even) / four central lines (odd) (Lemma 1);
`ρ`=180° is the only useful pairing symmetry (§3.2); odd n ⟹ `G ≥ 1` via center-then-mirror
(Lemma 2); for even n the outcome is decided entirely by the long-diagonal moves and a
first-player win requires a diagonal move (Theorem 3) — corroborated by n=18's I9=(8,8) sitting
on the main diagonal. **Refuted by `G(17) = 2`**: odd n ⟹ `G = 1`, and `G(n) ∈ {0,1}` for n≥9.
**Conjecture (unproven, surviving)**: `G(n) ≤ 3` / `G` bounded (standing caveat: Node-Kayles Grundy
values are unbounded in general, so no bound is inherited); the specific n=18 threshold (why the
even breakthrough is at n=18, not n≤16). **Prediction**: `G(18)=1` (~55%, else 2 ~30% / 3 ~12%);
run the pending `G(18)` computation at `k=1` first.
