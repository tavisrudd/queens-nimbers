import NodeKayles.Basic

/-!
# Phase 4 — the Grundy / Sprague–Grundy layer of the `getK` 2-lite verification (self-contained)

`grundy G S` is the minimal excludant (`mex`) of its children's Grundy values — the standard
impartial-game Grundy function. This file proves, with **no external game-theory dependency**:

* `win_iff_grundy_ne_zero` — the textbook P/N ↔ Grundy characterization (`win ⇔ grundy ≠ 0`);
* `grundy_iso` — the Grundy *value* is invariant under graph isomorphism;
* `mex_xor_union` / `grundy_sum` — the Sprague–Grundy component-XOR sum (disjoint parts XOR),
  built on mathlib's `Nat.lt_xor_cases`.

## Why self-contained — and the upgrade path (decision 2026-06-26)

The natural "blessed semantics" route would anchor `win`/`grundy` to a library's impartial-game
value (`Impartial`, `grundyValue`), so a reviewer trusts a cited, peer-reviewed definition
instead of checking that our recurrence is correct CGT. **That route is currently blocked, not
chosen against:**

* mathlib `v4.32` **removed `SetTheory/Game/`** — `PGame`/`Impartial`/`grundyValue`/`Nimber` were
  extracted to the standalone library `vihdzp/combinatorial-games`. Plain mathlib no longer has them.
* That library, as of 2026-06-09 (commit #419), tracks **Lean `v4.31.0-rc2`**, one minor version
  behind this project's **`v4.32.0-rc1`** (see `lean-toolchain`). Lean `.olean`s are
  toolchain-specific, so depending on it would force a *downgrade* of the whole project (re-pin
  toolchain + mathlib, re-fetch oleans, re-verify these proofs) — undoing a clean, pinned, green
  state for a slightly-older Lean. Not worth it now.

So Phase 4 is built **self-contained**, which is also the documented reason Approach A was chosen
(minimal mathlib footprint, no version churn). The cost: "`win`/`grundy` *are* the game value" is
self-asserted (the standard finite-impartial normal-play recurrence), not a cited theorem — but it
is corroborated against literature values (e.g. the path `P₃` has Grundy `2`, Dawson's chess / OEIS
A002187) and by the Rust differential tests.

**Upgrade path — when `combinatorial-games` ships a bump to Lean ≥ `v4.32` matching our toolchain:**
1. `require` it in `lakefile.toml` (no project downgrade needed once toolchains match).
2. Define `toPGame : Graph k → Finset (Fin k) → PGame` (moves = the `closedNbhd`-deletion children;
   impartial ⇒ Left and Right options coincide), and an `Impartial (toPGame G S)` instance.
3. Prove the bridge `win G S ↔ ¬ (toPGame G S ≈ 0)`, equivalently `grundy G S = grundyValue (toPGame G S)`.
   This **removes the self-asserted adequacy from the trusted base** — the review payoff.
4. `grundy_sum` then also follows from the library's `grundyValue_add` (nim-sum), and `grundy_iso`
   from value-congruence; our self-contained versions stay as the computable-spec layer.

`win_iff_grundy_ne_zero` / `grundy_sum` are the foundation for the component-XOR decomposition
(proposal item 3 — the queens nimber lever; see `grundy_sum`'s docstring for the gated/parked scope).
-/

namespace NodeKayles

variable {k : ℕ}

/-! ## Minimal excludant (`mex`) -/

/-- Minimal excludant of a finite set of naturals: the least `n` not in `T`.
    `Infinite.exists_notMem_finset` supplies the witness `Nat.find` needs — `ℕ` is
    infinite, `T` is finite, so some natural lies outside `T`. -/
def mex (T : Finset ℕ) : ℕ := Nat.find (Infinite.exists_notMem_finset T)

/-- `mex T` is not in `T` (the defining property — `Nat.find_spec`). -/
theorem mex_not_mem (T : Finset ℕ) : mex T ∉ T :=
  Nat.find_spec (Infinite.exists_notMem_finset T)

/-- Every natural below `mex T` *is* in `T` (`mex` is the *least* absentee). -/
theorem lt_mex_mem {T : Finset ℕ} {m : ℕ} (h : m < mex T) : m ∈ T := by
  have := Nat.find_min (Infinite.exists_notMem_finset T) h
  simpa using this

/-- `mex T = 0` iff `0` is absent — the only fact the win/Grundy bridge needs. -/
theorem mex_eq_zero_iff {T : Finset ℕ} : mex T = 0 ↔ (0 : ℕ) ∉ T := by
  constructor
  · intro h hmem
    rw [← h] at hmem
    exact mex_not_mem T hmem
  · intro h
    rcases Nat.eq_zero_or_pos (mex T) with h0 | hpos
    · exact h0
    · exact absurd (lt_mex_mem hpos) h

/-- `mex T ≠ 0` iff `0 ∈ T` — i.e. iff some child is a loss. -/
theorem mex_ne_zero_iff {T : Finset ℕ} : mex T ≠ 0 ↔ (0 : ℕ) ∈ T := by
  rw [ne_eq, mex_eq_zero_iff, not_not]

/-! ## The Grundy value and the win characterization -/

/-- Grundy value of a Node-Kayles position: the `mex` of its children's Grundy values.
    A move on `v ∈ S` deletes `closedNbhd G v`; the child position is `S \ N[v]`. Same
    well-founded recursion as `win` (`termination_by S.card`, via `sdiff_closedNbhd_ssubset`).
    Mirrors the solver's per-node nimber (`grundy`/`mex` over the available moves). -/
def grundy (G : Graph k) (S : Finset (Fin k)) : ℕ :=
  mex (S.attach.image (fun v => grundy G (S \ closedNbhd G v.val)))
termination_by S.card
decreasing_by exact Finset.card_lt_card (sdiff_closedNbhd_ssubset G v.2)

/-- **The Grundy characterization of `win`** (the textbook P/N ↔ Grundy fact): the player
    to move wins iff the position's Grundy value is nonzero. A position is a loss (`win`
    false) exactly when it is a P-position (`grundy = 0`), i.e. every move leads to a
    nonzero-Grundy (winning-for-the-opponent) child.

    Proof: unfold one ply of both sides. `win G S` is "∃ a move to a loss"; `grundy G S ≠ 0`
    is (`mex_ne_zero_iff`) "`0` is among the children's Grundy values", i.e. "∃ a move to a
    `grundy = 0` child". The recursive call on the strictly-smaller child (the IH) turns
    `¬ win (child)` into `grundy (child) = 0`, closing the equivalence move-by-move. -/
theorem win_iff_grundy_ne_zero (G : Graph k) (S : Finset (Fin k)) :
    win G S ↔ grundy G S ≠ 0 := by
  rw [win.eq_def, grundy.eq_def, mex_ne_zero_iff, Finset.mem_image]
  simp only [Finset.mem_attach, true_and]
  refine exists_congr (fun v => ?_)
  rw [win_iff_grundy_ne_zero G (S \ closedNbhd G v.val)]
  simp only [ne_eq, not_not]
termination_by S.card
decreasing_by exact Finset.card_lt_card (sdiff_closedNbhd_ssubset G v.2)

/-- Board-level corollary: the first player wins `G` iff its Grundy value is nonzero — the
    Grundy form of `firstPlayerWins` (what `get(k, code)` returns over the full graph). -/
theorem firstPlayerWins_iff_grundy_ne_zero (G : Graph k) :
    firstPlayerWins G ↔ grundy G Finset.univ ≠ 0 :=
  win_iff_grundy_ne_zero G Finset.univ

/-! ## Phase 4b — the Sprague–Grundy component-XOR sum

The dividend the solver's nimber/component decomposition relies on: a position that splits
into two parts with **no edges between them** (so the induced subgraph is their disjoint
union — a disconnected position) has Grundy value the `Nat.xor` of the parts' values. The
crux is the abstract nim-mex identity `mex_xor_union`, built on `Nat.lt_xor_cases`. -/

/-- `mex` pinned by its two-sided characterisation: if every value below `n` is in `U` and
    `n` itself is not, then `mex U = n`. -/
theorem mex_eq_of {U : Finset ℕ} {n : ℕ} (hlt : ∀ c < n, c ∈ U) (hn : n ∉ U) :
    mex U = n := by
  apply le_antisymm
  · exact Nat.find_min' (Infinite.exists_notMem_finset U) hn
  · by_contra h
    push Not at h
    exact mex_not_mem U (hlt _ h)

/-- `Nat.xor` is self-inverse on the right: `(z ^^^ b) ^^^ b = z`. -/
private theorem xor_cancel (z b : ℕ) : (z ^^^ b) ^^^ b = z := by
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- Right-cancellation for `Nat.xor`. -/
private theorem xor_right_inj {x z b : ℕ} (h : x ^^^ b = z ^^^ b) : x = z := by
  have := congrArg (· ^^^ b) h
  simpa only [xor_cancel] using this

/-- **The abstract nim-mex identity (Sprague–Grundy core).** With `a = mex A`, `b = mex B`,
    the `mex` of the "sum move set" `{x ^^^ b | x ∈ A} ∪ {y ^^^ a | y ∈ B}` is `a ^^^ b`.
    Two obligations: every `c < a ^^^ b` is a sum-move (`Nat.lt_xor_cases` routes `c` into the
    `A`- or `B`-side below the respective `mex`, then `lt_mex_mem`); and `a ^^^ b` itself is
    not (xor-cancellation would force `mex A ∈ A` or `mex B ∈ B`, impossible). -/
theorem mex_xor_union (A B : Finset ℕ) :
    mex (A.image (· ^^^ mex B) ∪ B.image (· ^^^ mex A)) = mex A ^^^ mex B := by
  apply mex_eq_of
  · intro c hc
    rw [Finset.mem_union, Finset.mem_image, Finset.mem_image]
    rcases Nat.lt_xor_cases hc with h | h
    · exact Or.inl ⟨c ^^^ mex B, lt_mex_mem h, xor_cancel c (mex B)⟩
    · exact Or.inr ⟨c ^^^ mex A, lt_mex_mem h, xor_cancel c (mex A)⟩
  · rw [Finset.mem_union, Finset.mem_image, Finset.mem_image]
    push Not
    refine ⟨fun x hx hxb => ?_, fun y hy hya => ?_⟩
    · exact mex_not_mem A (xor_right_inj hxb ▸ hx)
    · rw [Nat.xor_comm (mex A) (mex B)] at hya
      exact mex_not_mem B (xor_right_inj hya ▸ hy)

/-- `grundy` as a `mex` over `S` directly (no `attach` subtype), via `Finset.attach_image_val`.
    The convenient form for the component-sum's image manipulations. -/
theorem grundy_eq_mex_image (G : Graph k) (S : Finset (Fin k)) :
    grundy G S = mex (S.image (fun u => grundy G (S \ closedNbhd G u))) := by
  rw [grundy.eq_def]
  congr 1
  rw [show (fun v : {x // x ∈ S} => grundy G (S \ closedNbhd G v.val))
        = (fun u => grundy G (S \ closedNbhd G u)) ∘ Subtype.val from rfl,
     ← Finset.image_image, Finset.attach_image_val]

/-- **Grundy iso-invariance** — the `grundy` analogue of `win_iso`: transporting the live set
    along a graph isomorphism `e` preserves the Grundy *value* (not just win/loss). This is
    what makes the component-nimber oracle's iso-keyed memoisation sound (a cached nimber may
    be reused across isomorphic components). Proved by mirroring `grundy`'s recursion: the
    children's Grundy multiset is carried across `e` (`closedNbhd_map` + `Finset.map_sdiff`),
    so the two `mex` arguments coincide. -/
theorem grundy_iso (G : Graph k) (H : Graph k') (e : Fin k ≃ Fin k')
    (he : ∀ i j, G.adj i j = H.adj (e i) (e j)) (S : Finset (Fin k)) :
    grundy G S = grundy H (S.map e.toEmbedding) := by
  rw [grundy_eq_mex_image G S, grundy_eq_mex_image H (S.map e.toEmbedding)]
  congr 1
  rw [Finset.map_eq_image, Finset.image_image]
  apply Finset.image_congr
  intro u hu
  have huS : u ∈ S := hu
  dsimp only [Function.comp_apply, Equiv.coe_toEmbedding]
  rw [grundy_iso G H e he (S \ closedNbhd G u)]
  congr 1
  rw [Finset.map_sdiff, ← closedNbhd_map G H e he u, Finset.map_eq_image, Equiv.coe_toEmbedding]
termination_by S.card
decreasing_by exact Finset.card_lt_card (sdiff_closedNbhd_ssubset G huS)

/-- A move played from one side of an edge-disjoint split has its closed neighbourhood
    disjoint from the other side — so that side survives the move intact. -/
private theorem closedNbhd_disjoint_of_noedge (G : Graph k) {S T : Finset (Fin k)} {u : Fin k}
    (hu : u ∈ S) (hd : Disjoint S T) (hno : ∀ a ∈ S, ∀ b ∈ T, G.adj a b = false) :
    Disjoint T (closedNbhd G u) := by
  rw [Finset.disjoint_left]
  intro w hwT hwN
  simp only [closedNbhd, Finset.mem_filter, Finset.mem_univ, true_and] at hwN
  rcases hwN with heq | hadj
  · exact (Finset.disjoint_left.mp hd hu) (heq ▸ hwT)
  · rw [hno u hu w hwT] at hadj; simp at hadj

/-- A child position of an edge-disjoint split is a strict subset of the whole — the
    termination fact for the component-sum recursion. -/
private theorem child_ssubset (G : Graph k) {S₁ S₂ : Finset (Fin k)} {u : Fin k}
    (hu : u ∈ S₁) (hd : Disjoint S₁ S₂) :
    (S₁ \ closedNbhd G u) ∪ S₂ ⊂ S₁ ∪ S₂ := by
  refine (Finset.ssubset_iff_of_subset
    (Finset.union_subset_union Finset.sdiff_subset (Finset.Subset.refl _))).mpr ⟨u, ?_, ?_⟩
  · exact Finset.mem_union_left _ hu
  · rw [Finset.mem_union, Finset.mem_sdiff]
    push Not
    exact ⟨fun _ => self_mem_closedNbhd G u, Finset.disjoint_left.mp hd hu⟩

/-- **Sprague–Grundy component-XOR sum** (proposal item 3): a position that splits into two
    live sets with no edges between them (`hnoedge` — the induced subgraph is their disjoint
    union) has Grundy value the `Nat.xor` of the parts' values. This is the
    soundness of resolving a disconnected position by XOR-ing its components' nimbers instead
    of expanding the product game — the math behind the queens **component-nimber lever**
    (`QUEENS_NIMBER_ORACLE` in `iso_flat.rs`, default-OFF prototype, and the parked
    `queens-component-nimber` branch). NOTE: the default `getK`/iso-dense path and the shipped
    n=18 verdict use only the boolean win/loss recurrence (`win`), not this XOR — so `grundy_sum`
    hardens a *gated/future* lever, not live production code. (Stated for the binary split; the
    N-ary component XOR the oracle uses is the obvious induction on this, and the oracle's
    iso-keyed per-component memo is covered by `grundy_iso`.)

    Proof: by well-founded recursion on `(S₁ ∪ S₂).card`. A move in `S₁` leaves `S₂` intact
    (`closedNbhd_disjoint_of_noedge`), so the child is the edge-disjoint split
    `(S₁ ∖ N[u]) ∪ S₂`; the IH gives its value as `grundy (S₁ ∖ N[u]) ^^^ grundy S₂`.
    Symmetrically for moves in `S₂`. The move set's Grundy values are thus
    `{x ^^^ grundy S₂ | x ∈ children-of-S₁} ∪ {y ^^^ grundy S₁ | y ∈ children-of-S₂}`, whose
    `mex` is `grundy S₁ ^^^ grundy S₂` by `mex_xor_union`. -/
theorem grundy_sum (G : Graph k) (S₁ S₂ : Finset (Fin k))
    (hdisj : Disjoint S₁ S₂) (hnoedge : ∀ a ∈ S₁, ∀ b ∈ S₂, G.adj a b = false) :
    grundy G (S₁ ∪ S₂) = grundy G S₁ ^^^ grundy G S₂ := by
  rw [grundy_eq_mex_image G (S₁ ∪ S₂), grundy_eq_mex_image G S₁, grundy_eq_mex_image G S₂,
      ← mex_xor_union (S₁.image (fun u => grundy G (S₁ \ closedNbhd G u)))
                      (S₂.image (fun u => grundy G (S₂ \ closedNbhd G u)))]
  congr 1
  rw [Finset.image_union]
  congr 1
  · -- moves in S₁: child = (S₁ ∖ N[u]) ∪ S₂
    rw [Finset.image_image]
    apply Finset.image_congr
    intro u hu
    have huS : u ∈ S₁ := hu
    dsimp only [Function.comp_apply]
    have hNd : Disjoint S₂ (closedNbhd G u) :=
      closedNbhd_disjoint_of_noedge G huS hdisj hnoedge
    have h1 : (S₁ ∪ S₂) \ closedNbhd G u = (S₁ \ closedNbhd G u) ∪ S₂ := by
      rw [Finset.union_sdiff_distrib, Finset.sdiff_eq_self_of_disjoint hNd]
    rw [h1, grundy_sum G (S₁ \ closedNbhd G u) S₂
          (Finset.disjoint_of_subset_left Finset.sdiff_subset hdisj)
          (fun a ha b hb => hnoedge a (Finset.sdiff_subset ha) b hb),
        grundy_eq_mex_image G S₂]
  · -- moves in S₂: child = S₁ ∪ (S₂ ∖ N[u])
    rw [Finset.image_image]
    apply Finset.image_congr
    intro u hu
    have huS : u ∈ S₂ := hu
    dsimp only [Function.comp_apply]
    have hno' : ∀ a ∈ S₂, ∀ b ∈ S₁, G.adj a b = false :=
      fun a ha b hb => by rw [G.symm a b]; exact hnoedge b hb a ha
    have hNd : Disjoint S₁ (closedNbhd G u) :=
      closedNbhd_disjoint_of_noedge G huS hdisj.symm hno'
    have h1 : (S₁ ∪ S₂) \ closedNbhd G u = S₁ ∪ (S₂ \ closedNbhd G u) := by
      rw [Finset.union_sdiff_distrib, Finset.sdiff_eq_self_of_disjoint hNd]
    rw [h1, grundy_sum G S₁ (S₂ \ closedNbhd G u)
          (Finset.disjoint_of_subset_right Finset.sdiff_subset hdisj)
          (fun a ha b hb => hnoedge a ha b (Finset.sdiff_subset hb)),
        grundy_eq_mex_image G S₁]
    exact Nat.xor_comm _ _
termination_by (S₁ ∪ S₂).card
decreasing_by
  · exact Finset.card_lt_card (child_ssubset G huS hdisj)
  · rw [Finset.union_comm S₁ (S₂ \ closedNbhd G u), Finset.union_comm S₁ S₂]
    exact Finset.card_lt_card (child_ssubset G huS hdisj.symm)

end NodeKayles
