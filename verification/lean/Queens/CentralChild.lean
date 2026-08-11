import Queens.Basic
import NodeKayles.Certificate

/-!
# Central-child targets for the n=20 plan

The n=20 lucky-first-win plan reduces the hoped-for root win to proving that
the child after `J10 = (9,9)` is a P-position.  This file gives that statement
and the n=18 calibration target precise Lean names.
-/

namespace Queens

/-- The `I9 = (8,8)` square on the `18 x 18` board, using zero-based indices. -/
def I9 : Fin (18 * 18) :=
  index 18 8 8 (by decide) (by decide)

/-- The `J10 = (9,9)` square on the `20 x 20` board, using zero-based indices. -/
def J10 : Fin (20 * 20) :=
  index 20 9 9 (by decide) (by decide)

@[simp] theorem row_I9 : row 18 I9 = 8 := by
  simp [I9]

@[simp] theorem col_I9 : col 18 I9 = 8 := by
  simp [I9]

@[simp] theorem row_J10 : row 20 J10 = 9 := by
  simp [J10]

@[simp] theorem col_J10 : col 20 J10 = 9 := by
  simp [J10]

/-- Live set after a fixed root move. -/
def centralChildLive (n : ℕ) (root : Fin (n * n)) : Finset (Fin (n * n)) :=
  NodeKayles.child (queenGraph n) Finset.univ root

@[simp] theorem root_not_mem_centralChildLive (n : ℕ) (root : Fin (n * n)) :
    root ∉ centralChildLive n root := by
  simp [centralChildLive, NodeKayles.child]

/-- The root-child P-position target for a concrete first move. -/
def RootChildIsP (n : ℕ) (root : Fin (n * n)) : Prop :=
  ¬ NodeKayles.win (queenGraph n) (centralChildLive n root)

/-- Calibration target from the plan: prove the child after `I9` on `18 x 18` is P. -/
def N18I9CalibrationTarget : Prop :=
  RootChildIsP 18 I9

/-- Lucky n=20 target: prove the child after `J10` on `20 x 20` is P. -/
def N20J10LuckyTarget : Prop :=
  RootChildIsP 20 J10

/-- A certified P child after a root move proves the whole board is an N-position. -/
theorem firstPlayerWins_of_rootChildIsP {n : ℕ} {root : Fin (n * n)}
    (h : RootChildIsP n root) : NodeKayles.firstPlayerWins (queenGraph n) := by
  exact NodeKayles.firstPlayerWins_of_move_to_not_win (G := queenGraph n) (v := root) h

/-- The `19 x 19` core inside the `20 x 20` central-child geometry. -/
def N20Core : Finset (Fin (20 * 20)) :=
  Finset.univ.filter (fun i => row 20 i < 19 ∧ col 20 i < 19)

/-- The row/column-19 border from the n=20 central-child plan. -/
def N20Border : Finset (Fin (20 * 20)) :=
  Finset.univ.filter (fun i => row 20 i = 19 ∨ col 20 i = 19)

/--
The tau relation on the `19 x 19` core: `(r,c)` is paired with
`(18-r,18-c)`.  It is a relation for now rather than a function, avoiding any
premature proof that every flattened index lies in the intended core.
-/
def N20TauRelated (i j : Fin (20 * 20)) : Prop :=
  row 20 j = 18 - row 20 i ∧ col 20 j = 18 - col 20 i

/-! ## Certificate targets

These aliases connect the generic Node-Kayles reply-book scaffold to the two
central-child positions used by the n=20 certificate plan.  The certificate
validity predicate is the checker contract; unresolved extractor leaves are not
accepted by `NodeKayles.CertificateArtifact.Valid`.
-/

/-- Final certificate type for the known n=18 `I9` calibration child. -/
abbrev N18I9Certificate :=
  NodeKayles.FinalCertificate (queenGraph 18) (centralChildLive 18 I9)

/-- Final certificate type for the n=20 `J10` lucky child. -/
abbrev N20J10Certificate :=
  NodeKayles.FinalCertificate (queenGraph 20) (centralChildLive 20 J10)

/-- Extractor artifact type for the n=20 `J10` lucky child, including unresolved drafts. -/
abbrev N20J10Artifact :=
  NodeKayles.CertificateArtifact (queenGraph 20) (centralChildLive 20 J10)

/-- A valid final certificate proves the n=18 calibration child is P. -/
theorem N18I9CalibrationTarget_of_certificate (cert : N18I9Certificate)
    (hvalid : cert.Valid) : N18I9CalibrationTarget := by
  exact cert.isP hvalid

/-- A valid final certificate proves the n=20 lucky child is P. -/
theorem N20J10LuckyTarget_of_certificate (cert : N20J10Certificate)
    (hvalid : cert.Valid) : N20J10LuckyTarget := by
  exact cert.isP hvalid

/-- A valid extractor artifact proves the n=20 lucky child is P; unresolved leaves are rejected. -/
theorem N20J10LuckyTarget_of_artifact (artifact : N20J10Artifact)
    (hvalid : artifact.Valid) : N20J10LuckyTarget := by
  exact artifact.isP hvalid

/-- A valid n=20 `J10` child certificate proves the full `20 x 20` board is first-player winning. -/
theorem firstPlayerWins20_of_N20J10Certificate (cert : N20J10Certificate)
    (hvalid : cert.Valid) : NodeKayles.firstPlayerWins (queenGraph 20) := by
  exact firstPlayerWins_of_rootChildIsP (N20J10LuckyTarget_of_certificate cert hvalid)

end Queens
