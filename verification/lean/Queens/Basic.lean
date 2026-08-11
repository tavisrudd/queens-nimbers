import NodeKayles.Basic
import Mathlib.Tactic

/-!
# Queens boards as Node-Kayles graphs

A flat non-attacking queens position is a Node-Kayles position: vertices are
board squares, and a move deletes the chosen square together with every square
attacked by a queen on that square.
-/

namespace Queens

/-- Row of a flattened `n x n` board index. -/
def row (n : ℕ) (i : Fin (n * n)) : ℕ :=
  i.val / n

/-- Column of a flattened `n x n` board index. -/
def col (n : ℕ) (i : Fin (n * n)) : ℕ :=
  i.val % n

/-- Two flattened board indices are in a queen attack relation. -/
def Attacks (n : ℕ) (i j : Fin (n * n)) : Prop :=
  i ≠ j ∧
    (row n i = row n j ∨
     col n i = col n j ∨
     row n i + col n j = row n j + col n i ∨
     row n i + col n i = row n j + col n j)

instance decidableAttacks (n : ℕ) (i j : Fin (n * n)) : Decidable (Attacks n i j) := by
  unfold Attacks
  infer_instance

theorem attacks_symm (n : ℕ) (i j : Fin (n * n)) :
    Attacks n i j ↔ Attacks n j i := by
  constructor
  · rintro ⟨hij, hline⟩
    refine ⟨hij.symm, ?_⟩
    rcases hline with hrow | hcol | hdiag | hanti
    · exact Or.inl hrow.symm
    · exact Or.inr (Or.inl hcol.symm)
    · exact Or.inr (Or.inr (Or.inl hdiag.symm))
    · exact Or.inr (Or.inr (Or.inr hanti.symm))
  · rintro ⟨hji, hline⟩
    refine ⟨hji.symm, ?_⟩
    rcases hline with hrow | hcol | hdiag | hanti
    · exact Or.inl hrow.symm
    · exact Or.inr (Or.inl hcol.symm)
    · exact Or.inr (Or.inr (Or.inl hdiag.symm))
    · exact Or.inr (Or.inr (Or.inr hanti.symm))

@[simp] theorem not_attacks_self (n : ℕ) (i : Fin (n * n)) :
    ¬ Attacks n i i := by
  intro h
  exact h.1 rfl

/-- The `n x n` queen graph for Node-Kayles. -/
def queenGraph (n : ℕ) : NodeKayles.Graph (n * n) where
  adj i j := decide (Attacks n i j)
  symm i j := by
    by_cases hij : Attacks n i j
    · have hji : Attacks n j i := (attacks_symm n i j).1 hij
      simp [hij, hji]
    · have hji : ¬ Attacks n j i := by
        intro h
        exact hij ((attacks_symm n i j).2 h)
      simp [hij, hji]
  irrefl i := by
    have h : ¬ Attacks n i i := not_attacks_self n i
    exact decide_eq_false_iff_not.mpr h

theorem queenGraph_adj_eq_true_iff {n : ℕ} {i j : Fin (n * n)} :
    (queenGraph n).adj i j = true ↔ Attacks n i j := by
  by_cases h : Attacks n i j <;> simp [queenGraph, h]

/-- A coordinate square before flattening. -/
abbrev Square (n : ℕ) := Fin n × Fin n

/-- Flatten a square into the `Fin (n*n)` indexing used by `NodeKayles.Graph`. -/
def indexOf {n : ℕ} (s : Square n) : Fin (n * n) :=
  ⟨s.1.val * n + s.2.val, by
    have hc : s.2.val < n := s.2.isLt
    have hr : s.1.val < n := s.1.isLt
    have h1 : s.1.val * n + s.2.val < s.1.val * n + n :=
      Nat.add_lt_add_left hc _
    have h2 : s.1.val * n + n = (s.1.val + 1) * n := by
      rw [Nat.add_mul, one_mul]
    have h3 : (s.1.val + 1) * n ≤ n * n :=
      Nat.mul_le_mul_right n (Nat.succ_le_of_lt hr)
    exact lt_of_lt_of_le (by simpa [h2] using h1) h3⟩

/-- Flatten an explicit coordinate with proofs that it lies on the board. -/
def index (n r c : ℕ) (hr : r < n) (hc : c < n) : Fin (n * n) :=
  indexOf (n := n) (⟨r, hr⟩, ⟨c, hc⟩)

@[simp] theorem row_index {n r c : ℕ} (hr : r < n) (hc : c < n) :
    row n (index n r c hr hc) = r := by
  have hn : 0 < n := Nat.lt_of_le_of_lt (Nat.zero_le r) hr
  simp [row, index, indexOf, Nat.mul_comm r n, Nat.mul_add_div hn,
    Nat.div_eq_of_lt hc]

@[simp] theorem col_index {n r c : ℕ} (hr : r < n) (hc : c < n) :
    col n (index n r c hr hc) = c := by
  simp [col, index, indexOf, Nat.mul_comm r n, Nat.mod_eq_of_lt hc]

theorem index_inj {n r₁ c₁ r₂ c₂ : ℕ}
    {hr₁ : r₁ < n} {hc₁ : c₁ < n} {hr₂ : r₂ < n} {hc₂ : c₂ < n} :
    index n r₁ c₁ hr₁ hc₁ = index n r₂ c₂ hr₂ hc₂ ↔ r₁ = r₂ ∧ c₁ = c₂ := by
  constructor
  · intro h
    constructor
    · have hrow := congrArg (row n) h
      simpa using hrow
    · have hcol := congrArg (col n) h
      simpa using hcol
  · rintro ⟨rfl, rfl⟩
    apply Fin.ext
    simp [index, indexOf]

theorem attacks_index_iff {n r₁ c₁ r₂ c₂ : ℕ}
    {hr₁ : r₁ < n} {hc₁ : c₁ < n} {hr₂ : r₂ < n} {hc₂ : c₂ < n} :
    Attacks n (index n r₁ c₁ hr₁ hc₁) (index n r₂ c₂ hr₂ hc₂) ↔
      ¬ (r₁ = r₂ ∧ c₁ = c₂) ∧
        (r₁ = r₂ ∨
         c₁ = c₂ ∨
         r₁ + c₂ = r₂ + c₁ ∨
         r₁ + c₁ = r₂ + c₂) := by
  simp [Attacks, index_inj]

end Queens
