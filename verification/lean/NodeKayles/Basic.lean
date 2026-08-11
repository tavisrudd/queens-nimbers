import Mathlib

/-!
# Node-Kayles win predicate — Phase 1 (Approach A, the spec backbone)

The 2-lite formal-verification spine for the queens `getK` leaf evaluator. This file is
the Lean mirror of the *scalar reference* in `rust/src/queens/dense.rs`:

| Lean (here)          | Rust (`dense.rs`)                       | meaning                              |
|----------------------|-----------------------------------------|--------------------------------------|
| `win`                | `wins_rec` (`:584`, the `#[cfg(test)]` scalar ref) | `W_K(G) = ∃v · ¬W_{K-1}(G∖N[v])` |
| `closedNbhd G v`     | `(1<<i) \| adj[i]` (the *deleted* set)    | delete `{v} ∪ N(v)` (the move)       |
| `S \ closedNbhd G v` | `full & !((1<<i) \| adj[i])` (the child)  | the surviving live set after the move |
| `firstPlayerWins`    | `get(k, code)` over the full graph       | first player wins this position      |

`win` is a `Prop` here (Node-Kayles normal-play P/N value): clean for the `win_iso`
invariance proof and well-founded recursion. The *computable* `Bool` twin + its `#eval`
cross-check against the Rust `wins_rec` is Phase 3. Phase 2 adds the `code ↔ Graph` decode
(`graphOfCode` ↔ `adj_from_code`) and the W0..W8 build correctness; Phase 4 (optional)
grafts the mathlib `PGame`/Sprague–Grundy bridge.
-/

namespace NodeKayles

variable {k : ℕ}

/-- A finite simple graph on `Fin k`: symmetric, irreflexive adjacency (`Bool`-valued, so
    decidable by construction). The abstract Node-Kayles graph one labelled
    upper-triangular `code` decodes to (Phase 2). -/
structure Graph (k : ℕ) where
  adj    : Fin k → Fin k → Bool
  symm   : ∀ i j, adj i j = adj j i
  irrefl : ∀ i, adj i i = false

/-- Closed neighbourhood of `v`: `v` together with its neighbours — the vertex set a move
    deletes. Placing a queen on a square blocks that square plus every square it attacks,
    i.e. removes `{v} ∪ N(v)` from the live set. -/
def closedNbhd (G : Graph k) (v : Fin k) : Finset (Fin k) :=
  Finset.univ.filter (fun u => u = v ∨ G.adj v u = true)

@[simp] theorem self_mem_closedNbhd (G : Graph k) (v : Fin k) :
    v ∈ closedNbhd G v := by
  simp [closedNbhd]

/-- Node-Kayles win predicate (normal play): the player to move **wins** iff some live
    vertex `v`, when played, leaves the opponent a **loss**. `S` is the set of live
    vertices; a terminal position (`S = ∅`) is a loss for the mover.

    Well-founded on `S.card`: the played `v` lies in both `S` and its own closed
    neighbourhood (`self_mem_closedNbhd`), so the child set `S \ N[v]` is a strict subset
    of `S` and its cardinality strictly drops. Mirrors `wins_rec` (`dense.rs:584`). -/
def win (G : Graph k) (S : Finset (Fin k)) : Prop :=
  -- Bind over the subtype `S` (not `∃ v ∈ S`, a conjunction) so `v.2 : ↑v ∈ S` is in
  -- scope at the recursive call — the witness the termination proof needs.
  ∃ v : S, ¬ win G (S \ closedNbhd G (v : Fin k))
termination_by S.card
decreasing_by
  have hv : (v : Fin k) ∈ S := v.2
  have hssub : S \ closedNbhd G (v : Fin k) ⊂ S :=
    (Finset.ssubset_iff_of_subset Finset.sdiff_subset).mpr
      ⟨(v : Fin k), hv, fun hc => (Finset.mem_sdiff.mp hc).2 (self_mem_closedNbhd G (v : Fin k))⟩
  exact Finset.card_lt_card hssub

/-- "First player wins the game on `G`" — every vertex live. The board-level result
    (e.g. n=18 is a first-player win) is `firstPlayerWins (queenGraph 18)`, where
    `queenGraph` is the Phase-2 board→graph bridge (`att08`/`adj_row_pext`). -/
def firstPlayerWins (G : Graph k) : Prop := win G Finset.univ

/-- A live vertex's move strictly shrinks the position — the termination fact, shared by
    `win`'s recursion and `win_iso`'s. -/
theorem sdiff_closedNbhd_ssubset (G : Graph k) {S : Finset (Fin k)} {v : Fin k}
    (hv : v ∈ S) : S \ closedNbhd G v ⊂ S :=
  (Finset.ssubset_iff_of_subset Finset.sdiff_subset).mpr
    ⟨v, hv, fun hc => (Finset.mem_sdiff.mp hc).2 (self_mem_closedNbhd G v)⟩

/-- `closedNbhd` commutes with a graph isomorphism `e`: the closed neighbourhood of `e v`
    in `H` is the image under `e` of `v`'s closed neighbourhood in `G`. The geometric core
    of `win_iso` / `projected_code`'s relabelling. -/
theorem closedNbhd_map (G : Graph k) (H : Graph k') (e : Fin k ≃ Fin k')
    (he : ∀ i j, G.adj i j = H.adj (e i) (e j)) (v : Fin k) :
    closedNbhd H (e v) = (closedNbhd G v).map e.toEmbedding := by
  ext u
  rw [Finset.mem_map_equiv]
  simp only [closedNbhd, Finset.mem_filter, Finset.mem_univ, true_and,
    Equiv.symm_apply_eq, he v (e.symm u), Equiv.apply_symm_apply]

/-- **Relabeling (isomorphism) invariance.** `win` depends only on the isomorphism class
    of `(G, S)`: transporting the live set along a graph isomorphism `e` preserves the
    value. The mathematical content behind `projected_code` (`dense.rs:516`) — the getK
    recurrence relabels each child's surviving vertices to `0..k'` to index a smaller
    table, and this lemma is what makes that lookup sound.

    Proved by mirroring `win`'s recursion (`termination_by S.card`): `closedNbhd_map` +
    `Finset.map_sdiff` carry each child across `e`, and the recursive `win_iso` call closes
    it on the strictly-smaller child. -/
theorem win_iso (G : Graph k) (H : Graph k') (e : Fin k ≃ Fin k')
    (he : ∀ i j, G.adj i j = H.adj (e i) (e j)) (S : Finset (Fin k)) :
    win G S ↔ win H (S.map e.toEmbedding) := by
  conv_lhs => rw [win.eq_def]
  conv_rhs => rw [win.eq_def]
  constructor
  · rintro ⟨⟨v, hv⟩, hvlose⟩
    have hev : e v ∈ S.map e.toEmbedding := by
      rw [Finset.mem_map_equiv]; simpa using hv
    refine ⟨⟨e v, hev⟩, ?_⟩
    show ¬ win H (S.map e.toEmbedding \ closedNbhd H (e v))
    rw [closedNbhd_map G H e he v, ← Finset.map_sdiff, ← win_iso G H e he]
    exact hvlose
  · rintro ⟨⟨w, hw⟩, hwlose⟩
    rw [Finset.mem_map_equiv] at hw
    refine ⟨⟨e.symm w, hw⟩, ?_⟩
    show ¬ win G (S \ closedNbhd G (e.symm w))
    rw [win_iso G H e he, Finset.map_sdiff, ← closedNbhd_map G H e he, Equiv.apply_symm_apply]
    exact hwlose
termination_by S.card
decreasing_by
  all_goals
    refine Finset.card_lt_card (sdiff_closedNbhd_ssubset G ?_)
    -- forward call: `v ∈ S` directly; backward call: `e.symm w ∈ S` from `w ∈ map e S`.
    first
      | assumption
      | exact Finset.mem_map_equiv.mp (by assumption)

/-- The child of a move carried across an embedding `ι` (the induced-subgraph case): in `H`
    the move at `v` deletes `closedNbhd H v`, and its image under `ι` agrees, on `T.map ι ⊆
    range ι`, with deleting `closedNbhd G (ι v)`. Hence the two children correspond. The
    induced-subgraph analogue of `closedNbhd_map`. -/
theorem childmap_emb (G : Graph k) (H : Graph k') (ι : Fin k' ↪ Fin k)
    (he : ∀ a b, H.adj a b = G.adj (ι a) (ι b)) (T : Finset (Fin k')) (v : Fin k') :
    (T \ closedNbhd H v).map ι = T.map ι \ closedNbhd G (ι v) := by
  rw [Finset.map_sdiff]
  ext x
  simp only [Finset.mem_sdiff, and_congr_right_iff]
  intro hxT
  obtain ⟨u, _huT, rfl⟩ := Finset.mem_map.mp hxT
  rw [not_iff_not, Finset.mem_map]
  constructor
  · rintro ⟨a, ha, hau⟩
    rw [← hau]
    simp only [closedNbhd, Finset.mem_filter, Finset.mem_univ, true_and] at ha ⊢
    rcases ha with h | h
    · exact Or.inl (by rw [h])
    · exact Or.inr (by rw [← he v a]; exact h)
  · intro h
    refine ⟨u, ?_, rfl⟩
    simp only [closedNbhd, Finset.mem_filter, Finset.mem_univ, true_and] at h ⊢
    rcases h with h | h
    · exact Or.inl (ι.injective h)
    · exact Or.inr (by rw [he v u]; exact h)

/-- **Induced-subgraph invariance.** If `H` is (isomorphic to) the subgraph of `G` induced
    on the range of an embedding `ι` (`he`: `ι` carries `H`'s adjacency to `G`'s), then the
    Node-Kayles value of any position `T` of `H` equals that of its image `T.map ι` in `G`.

    This is the soundness of `projected_code` (`dense.rs:516`): the `getK` recurrence
    resolves a child by relabelling its surviving vertices to `0..k'` and reading the
    smaller `W{k'}` table — sound exactly because `win` sees only the induced subgraph.
    Generalises `win_iso` (the `k' = k`, `ι` a bijection case). Proved by mirroring `win`'s
    recursion (`termination_by T.card`), carrying each child across `ι` via `childmap_emb`. -/
theorem win_emb (G : Graph k) (H : Graph k') (ι : Fin k' ↪ Fin k)
    (he : ∀ a b, H.adj a b = G.adj (ι a) (ι b)) (T : Finset (Fin k')) :
    win H T ↔ win G (T.map ι) := by
  conv_lhs => rw [win.eq_def]
  conv_rhs => rw [win.eq_def]
  constructor
  · rintro ⟨⟨v, hv⟩, hvlose⟩
    refine ⟨⟨ι v, Finset.mem_map_of_mem ι hv⟩, ?_⟩
    show ¬ win G (T.map ι \ closedNbhd G (ι v))
    rw [← childmap_emb G H ι he T v, ← win_emb G H ι he]
    exact hvlose
  · rintro ⟨⟨x, hx⟩, hxlose⟩
    obtain ⟨v, hv, rfl⟩ := Finset.mem_map.mp hx
    refine ⟨⟨v, hv⟩, ?_⟩
    show ¬ win H (T \ closedNbhd H v)
    rw [win_emb G H ι he, childmap_emb G H ι he T v]
    exact hxlose
termination_by T.card
decreasing_by
  all_goals exact Finset.card_lt_card (sdiff_closedNbhd_ssubset H (by assumption))

/-! ## Phase 2 (graph-level): the `W_K` build recurrence is correct.

`firstPlayerWins` is the value the dense tables store. `buildPred_correct` shows one ply of
the build (`graph_wins`, `dense.rs:541`) — resolve each move's child by relabelling its
survivors to a smaller induced graph and reading *its* value — equals the true value, and
`not_win_empty` is the `W0` base case. The u128 bit-packing of the code (`adj_from_code` /
`projected_code`) is the serialization layer, deferred to the Rust differential tests. -/

/-- The subgraph of `G` induced on a live set `S`, relabelled to `Fin S.card` by the order
    embedding `S.orderEmbOfFin`. Mirror of decoding a `projected_code` child to a smaller
    labelled graph. -/
def inducedGraph (G : Graph k) (S : Finset (Fin k)) : Graph S.card where
  adj a b := G.adj (S.orderEmbOfFin rfl a) (S.orderEmbOfFin rfl b)
  symm a b := G.symm (S.orderEmbOfFin rfl a) (S.orderEmbOfFin rfl b)
  irrefl a := G.irrefl (S.orderEmbOfFin rfl a)

/-- The order embedding sends the whole index set onto `S`. -/
theorem univ_map_orderEmbOfFin (S : Finset (Fin k)) :
    Finset.univ.map (S.orderEmbOfFin rfl).toEmbedding = S := by
  apply Finset.eq_of_subset_of_card_le
  · intro x hx
    obtain ⟨a, -, rfl⟩ := Finset.mem_map.mp hx
    exact Finset.orderEmbOfFin_mem S rfl a
  · rw [Finset.card_map]; simp

/-- The value of an induced subgraph equals the value of the corresponding position of `G`
    — the bridge from a relabelled child back to `G`, via `win_emb`. -/
theorem firstPlayerWins_inducedGraph (G : Graph k) (S : Finset (Fin k)) :
    firstPlayerWins (inducedGraph G S) ↔ win G S := by
  show win (inducedGraph G S) Finset.univ ↔ win G S
  rw [win_emb G (inducedGraph G S) (S.orderEmbOfFin rfl).toEmbedding (fun _ _ => rfl)
        Finset.univ, univ_map_orderEmbOfFin]

/-- Base case (`W0`): the terminal position (no live vertex) is a loss for the mover. -/
theorem not_win_empty (G : Graph k) : ¬ win G (∅ : Finset (Fin k)) := by
  rw [win.eq_def]
  rintro ⟨⟨v, hv⟩, -⟩
  simp at hv

/-- **The `W_K` build recurrence is correct** (`graph_wins`, `dense.rs:541`): the first
    player wins `G` iff some move leaves the opponent a loss, each child resolved as the
    value of a strictly smaller induced subgraph. With `not_win_empty` (the `W0` base) this
    is the graph-level (one-ply) correctness of the `W_K` build recurrence; the concrete
    table indexing, the flat-arena offset read, and the u128 `code` decode are NOT modelled
    here — they ride on the Rust differential tests (see `TRUST.md`, "Deferred"). -/
theorem buildPred_correct (G : Graph k) :
    firstPlayerWins G ↔
      ∃ i : Fin k, ¬ firstPlayerWins (inducedGraph G (Finset.univ \ closedNbhd G i)) := by
  simp only [firstPlayerWins_inducedGraph]
  show win G Finset.univ ↔ ∃ i : Fin k, ¬ win G (Finset.univ \ closedNbhd G i)
  rw [win.eq_def]
  exact ⟨fun ⟨⟨i, _⟩, hi⟩ => ⟨i, hi⟩, fun ⟨i, hi⟩ => ⟨⟨i, Finset.mem_univ i⟩, hi⟩⟩

end NodeKayles
