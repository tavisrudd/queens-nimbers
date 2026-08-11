# Verification

What is checked, how to check it, and what remains trusted. The frozen results and the exact
configurations that produced them are in [RESULTS.md](RESULTS.md).

## The defect class this is built against

The one real bug ever found in this code was a **leaf-decode defect**. Migrating from `n ≤ 16` to
`n = 18` widened square indices from 8 to 16 bits across most of the code but missed the
small-graph / component-canonicalisation path in one module, which still stored board-square
indices in `u8`. At `n ≤ 16` the maximum square index is 255 and fits exactly; at `n ≥ 17` squares
reach 323 and silently truncate (256 → 0, …), corrupting the adjacency rows, hence the code, hence
the looked-up value — a loss↔win flip. It was caught in milliseconds at `pc = 3` by the independent
oracle differential below, and fixed by widening the indices. Everything here is arranged around
that class of failure: the leaf evaluator is where the bugs live, so it is checked several
independent ways.

## Gates

Run the whole Rust suite with `cargo test --release`.

- **Lineage agreement.** Every solver variant matches the memo-less `naive` recurrence's verdict for
  all `n ≤ 9`. `naive` is the ground truth the entire ladder is pinned to
  (`src/queens/mod.rs`, `solver_lineage_agrees`).
- **Exact distinct-position invariants.** `iso-flat` reports its exact distinct-position count: the
  `n = 12` value is **1,060,823** (a second-player win) and the `n = 14` value is about 29.2 M with
  re-expansion about 1.0×. A change in the distinct count signals a lost transposition merge; a jump
  in re-expansion signals an undersized table. Key and table changes must preserve both. The
  about-49.3 M figure sometimes quoted elsewhere is the D₄-distinct count; `iso-flat`'s isomorphism
  key merges below it.
- **Differential tests against a scalar reference** (`src/queens/dense.rs`). The optimised `pext`
  evaluator is compared bit for bit against a plain scalar recurrence across tens of thousands of
  density-spread codes per layer: `graph_wins8_matches_scalar` pins the `pext` W8 build to the
  scalar build; `direct_w9..w16_matches_scalar_recurrence` pins the `getK` layers 9–16 to the
  `u16`-coded scalar reference `wins_rec`; and `direct_w17..w20_matches_scalar_recurrence` pins the
  wide layers — **including the W17/W18 leaves the n = 18 verdict bottoms out on** — to a three-word
  scalar reference `winsw_scalar`. Both n = 18 evaluator configurations are therefore
  scalar-validated. The Grundy layer has the parallel chain
  (`grundy_g9_to_g16_match_reference`, and `G ≠ 0 ⟺ W-win` against the independently built boolean
  tables).
- **Nimber engine against an independent full-mex reference.** The heap-sum engine is checked
  against the full-DAG mex `Nimber` solver for `n ≤ 8` and against the catalogued OEIS terms
  (`src/queens/mod.rs`, `nimber_sum_matches_full_mex_and_oeis`, `nimbers_match_oeis_a344227`).
- **Independent raw-mask oracle.** During the n = 18 migration, about 3,400 18×18 subpositions whose
  live sets span the high-index words were driven through both `iso-dense` and `iso-flat` and
  checked against a *different implementation* — a raw-mask negamax with no `getK`, no
  canonicalisation, and no table. This is the check that caught the truncation defect above. It was
  a migration harness rather than a committed test; the committed differential and lineage gates
  cover the same decode path continuously.
- **Integer-width audit.** A from-scratch read of the value path (mask words, geometry, the
  `d4_bits` bijection, the 128-bit hash, the code build, the wide W17–W20 path) for this bug class
  found no defect able to flip an n = 18 value; the two residual findings were off the proving runs'
  configuration.
- **Reproduction of prior work.** The kernel reproduces Jenrich's full `n ≤ 16` sequence (with
  `n = 16` a second-player win) and agrees with OEIS A344227 across the catalogued range.

## The Lean 4 development

`lean/` is a self-contained Lean 4 + mathlib project, `lake build` green with **no `sorry`**; every
theorem depends only on the three standard mathlib axioms `propext`, `Classical.choice`,
`Quot.sound` — no `sorryAx`, no `native_decide`, no custom axioms. Build it with:

```sh
cd verification/lean && lake exe cache get && lake build
```

The toolchain and the mathlib revision are pinned (`lean-toolchain`, `lakefile.toml`,
`lake-manifest.json`). Do not run `lake update`: it re-floats mathlib to master and breaks the
toolchain pin.

The scope is deliberately **"2-lite"** — prove the recurrence and algorithm semantics in Lean, and
leave the bit-level `pext` and serialization work to the differential tests above, where it is
cheapest to check. Over an abstract finite simple graph `Graph k` on `Fin k` with a `closedNbhd`
operation:

- `win`, the P/N recurrence, is **well-defined and terminating** (well-founded on `|S|`: the played
  vertex lies in its own closed neighbourhood, so the child set strictly shrinks). This mirrors the
  scalar reference `wins_rec`.
- `win_iso` — the value is **invariant under graph isomorphism**, which justifies using any
  labelling of a position's vertices.
- `win_emb` — the value is **invariant under induced-subgraph relabelling**. This is the soundness
  of `projected_code`: the `getK` step that relabels a child's surviving vertices to `0 … k′` and
  reads the smaller `W{k′}` table cannot change the value.
- `buildPred_correct` — the **one-ply build recurrence equals the true value**, mirroring the
  table-build function `graph_wins`, with `not_win_empty` (the empty graph is a loss) as the `W0`
  base case.
- `mex`, `grundy`, and `win_iff_grundy_ne_zero` — the **Grundy characterisation**
  `win ⟺ grundy ≠ 0`.
- `grundy_iso` and `grundy_sum` — Grundy **isomorphism invariance** and the **Sprague–Grundy
  component sum** `grundy(S₁ ∪ S₂) = grundy S₁ ⊕ grundy S₂` when no edges run between the parts.

Adequacy was corroborated against the literature: the path `P₃` computes to Grundy value 2
(Dawson's chess, OEIS A002187), and isolated-vertex parity computes to `n mod 2`.

### What the Lean development does not cover

- The u128 and three-word **code bit-layout and its `pext` decode** (`adj_from_code`,
  `projected_code`), the `pext` W8 build fast path, the generic high-popcount α-β combination logic
  above the dense ceiling, and the board→code build. These are carried by the differential tests.
- The **Lean↔Rust correspondence itself** — that the Lean definitions faithfully mirror the Rust
  functions — is hand-audited, not machine-checked end to end; auto-translation is not viable
  through `pext` intrinsics, const-generic monomorphisation, and unchecked indexing. The move
  polarity is recorded explicitly: Lean's `closedNbhd G v`, the deleted set `{v} ∪ N(v)`,
  corresponds to the Rust `(1<<i) | adj[i]`, and the surviving child `S \ closedNbhd` to its
  complement `full & ¬((1<<i) | adj[i])`.
- `grundy_sum` and `grundy_iso` are **not** on the default `getK`/`iso-dense` path and are not part
  of the n = 18 verdict, which uses only the boolean `win` recurrence. They formalise the principle
  the heap-sum nimber engine relies on rather than the engine's own board-plus-Nim-heap
  instantiation.
- Combinatorial-game theory was, until recently, in mathlib's `SetTheory/Game/`; it has since been
  extracted to a standalone library tracking an older toolchain than this project's, so the Grundy
  layer is built self-contained rather than anchored to a library `Impartial`/`grundyValue`. The
  statement "`win` *is* the game value" therefore rests on the standard-recurrence argument plus the
  literature cross-checks above, not on a cited library theorem.

## Residual trusted base

The n = 18 verdict is **cross-validated, not formally certified**: two independently configured
exhaustive searches, a scalar-validated leaf evaluator, and a search-free re-verification of the
principal variation, but no machine-checkable certificate of the whole search. A full certificate
(a reply book covering every opponent line) is future work. Compiler, hardware, and the
uninstrumented parts of the parallel search remain trusted in the ordinary way.
