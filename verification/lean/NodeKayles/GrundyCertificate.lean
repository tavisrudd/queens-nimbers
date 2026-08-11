import NodeKayles.Grundy

/-!
# Reflected Grundy-book certificates

A book stores every reachable live set once, its claimed Grundy value, and one
move witnessing each smaller value.  The checker also enumerates every legal
move and rejects a book if a child is absent or has the node's claimed value.
This is exactly the two-sided mex contract.
-/

namespace NodeKayles

/-- One row of a Grundy book. `lowerMoves i` witnesses child value `i`. -/
structure GrundyBookNode (k : ℕ) where
  position : Finset (Fin k)
  value : ℕ
  lowerMoves : Fin value → Fin k

/-- A self-contained Grundy DAG with one distinguished root claim. -/
structure GrundyBookData (k : ℕ) where
  root : Finset (Fin k)
  rootValue : ℕ
  nodes : List (GrundyBookNode k)

namespace GrundyBookData

variable {k : ℕ}

/-- First row for a live set. The checker separately validates every listed row. -/
def lookupNode (nodes : List (GrundyBookNode k)) (S : Finset (Fin k)) :
    Option (GrundyBookNode k) :=
  nodes.find? (fun node => node.position = S)

/-- Claimed value for a live set, if the row exists. -/
def lookupValue (nodes : List (GrundyBookNode k)) (S : Finset (Fin k)) : Option ℕ :=
  (lookupNode nodes S).map (·.value)

/-- Local, rules-only mex obligations for one row. -/
def NodeValid (G : Graph k) (nodes : List (GrundyBookNode k))
    (node : GrundyBookNode k) : Prop :=
  (∀ i : Fin node.value,
      let move := node.lowerMoves i
      move ∈ node.position ∧
        lookupValue nodes (node.position \ closedNbhd G move) = some i.val) ∧
    (∀ move : Fin k, move ∈ node.position →
      lookupValue nodes (node.position \ closedNbhd G move) ≠ none ∧
        lookupValue nodes (node.position \ closedNbhd G move) ≠ some node.value)

instance nodeValidDecidable (G : Graph k) (nodes : List (GrundyBookNode k))
    (node : GrundyBookNode k) : Decidable (NodeValid G nodes node) := by
  unfold NodeValid
  infer_instance

/-- Executable reflected checker. -/
def check (G : Graph k) (book : GrundyBookData k) : Bool :=
  decide (lookupValue book.nodes book.root = some book.rootValue) &&
    book.nodes.all (fun node => decide (NodeValid G book.nodes node))

private theorem lookupNode_mem {nodes : List (GrundyBookNode k)}
    {S : Finset (Fin k)} {node : GrundyBookNode k}
    (h : lookupNode nodes S = some node) : node ∈ nodes := by
  unfold lookupNode at h
  exact List.mem_of_find?_eq_some h

private theorem lookupValue_eq_some {nodes : List (GrundyBookNode k)}
    {S : Finset (Fin k)} {g : ℕ}
    (h : lookupValue nodes S = some g) :
    ∃ node, lookupNode nodes S = some node ∧ node.value = g := by
  unfold lookupValue at h
  cases hnode : lookupNode nodes S with
  | none => simp [hnode] at h
  | some node =>
      have hvalue : node.value = g := by simpa [hnode] using h
      exact ⟨node, rfl, hvalue⟩

private theorem lookupNode_position {nodes : List (GrundyBookNode k)}
    {S : Finset (Fin k)} {node : GrundyBookNode k}
    (h : lookupNode nodes S = some node) : node.position = S := by
  unfold lookupNode at h
  have hpred := List.find?_some h
  simpa using hpred

private theorem nodeValid_of_check {G : Graph k} {book : GrundyBookData k}
    (hcheck : check G book = true) {node : GrundyBookNode k}
    (hmem : node ∈ book.nodes) : NodeValid G book.nodes node := by
  unfold check at hcheck
  simp only [Bool.and_eq_true] at hcheck
  have hall : book.nodes.all (fun row => decide (NodeValid G book.nodes row)) = true :=
    hcheck.2
  simp only [List.all_eq_true] at hall
  have hrow : decide (NodeValid G book.nodes node) = true := by
    exact hall node hmem
  exact of_decide_eq_true hrow

/-- Every checked row has its claimed mathematical Grundy value. -/
theorem grundy_eq_value_of_lookup {G : Graph k} {book : GrundyBookData k}
    (hcheck : check G book = true) {S : Finset (Fin k)} {node : GrundyBookNode k}
    (hlookup : lookupNode book.nodes S = some node) :
    grundy G S = node.value := by
  have hmem : node ∈ book.nodes := lookupNode_mem hlookup
  have hvalid := nodeValid_of_check hcheck hmem
  have hposition : node.position = S := lookupNode_position hlookup
  rw [grundy.eq_def]
  apply mex_eq_of
  · intro m hm
    let i : Fin node.value := ⟨m, hm⟩
    let move := node.lowerMoves i
    have hlower := hvalid.1 i
    dsimp only at hlower
    have hmove : move ∈ S := by simpa [move, hposition] using hlower.1
    have hclaim : lookupValue book.nodes (S \ closedNbhd G move) = some m := by
      simpa [move, i, hposition] using hlower.2
    obtain ⟨childNode, hchildLookup, hchildValue⟩ := lookupValue_eq_some hclaim
    rw [Finset.mem_image]
    refine ⟨⟨move, hmove⟩, Finset.mem_attach _ _, ?_⟩
    have hchildGrundy := grundy_eq_value_of_lookup hcheck hchildLookup
    simpa [hchildValue] using hchildGrundy
  · intro hmemValue
    rw [Finset.mem_image] at hmemValue
    obtain ⟨move, _hattach, hgrundy⟩ := hmemValue
    have hmove : (move : Fin k) ∈ S := move.2
    have hclosure := hvalid.2 (move : Fin k) (by rw [hposition]; exact hmove)
    have hsome : lookupValue book.nodes (S \ closedNbhd G (move : Fin k)) ≠ none := by
      simpa [hposition] using hclosure.1
    cases hchild : lookupValue book.nodes (S \ closedNbhd G (move : Fin k)) with
    | none => exact hsome hchild
    | some childValue =>
        have hne : childValue ≠ node.value := by
          intro heq
          apply hclosure.2
          simpa [hposition, heq] using hchild
        obtain ⟨childNode, hchildLookup, hchildValue⟩ :=
          lookupValue_eq_some (by simpa using hchild)
        have hchildGrundy := grundy_eq_value_of_lookup hcheck hchildLookup
        apply hne
        rw [← hchildValue, ← hchildGrundy]
        simpa using hgrundy
termination_by S.card
decreasing_by
  all_goals
    exact Finset.card_lt_card (sdiff_closedNbhd_ssubset G (by assumption))

/-- Soundness direction required by C50. -/
theorem root_grundy_eq_of_check {G : Graph k} {book : GrundyBookData k}
    (hcheck : check G book = true) :
    grundy G book.root = book.rootValue := by
  have hroot : lookupValue book.nodes book.root = some book.rootValue := by
    unfold check at hcheck
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hcheck
    exact hcheck.1
  obtain ⟨node, hlookup, hvalue⟩ := lookupValue_eq_some hroot
  simpa [hvalue] using grundy_eq_value_of_lookup hcheck hlookup

/-- Convert a literal move list of the checked length into lower-value witnesses. -/
def movesOfList {g : ℕ} (moves : List (Fin k)) (h : moves.length = g) : Fin g → Fin k :=
  fun i => moves.get (Fin.cast h.symm i)

end GrundyBookData
end NodeKayles
