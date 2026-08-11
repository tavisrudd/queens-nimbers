import NodeKayles.Basic

/-!
# Small P-position certificates for Node-Kayles

The production solvers may eventually emit reply books.  This file gives the
basic kernel theorem such a book should satisfy: if every legal opponent move
has a certified reply to a P child, then the current position is P.
-/

namespace NodeKayles

variable {k : ℕ}

/-- The live set after playing `v` from `S`. -/
def child (G : Graph k) (S : Finset (Fin k)) (v : Fin k) : Finset (Fin k) :=
  S \ closedNbhd G v

/--
A one-ply P-certificate for a position: every move has a reply whose child is
already known to be P.
-/
def ReplyCertificate (G : Graph k) (S : Finset (Fin k)) : Prop :=
  ∀ v : S, ∃ w : child G S v, ¬ win G (child G (child G S v) w)

/-- A reply certificate proves that `S` is a P-position. -/
theorem not_win_of_replyCertificate {G : Graph k} {S : Finset (Fin k)}
    (hcert : ReplyCertificate G S) : ¬ win G S := by
  rw [win.eq_def]
  rintro ⟨v, hvlose⟩
  rcases hcert v with ⟨w, hwP⟩
  apply hvlose
  rw [win.eq_def]
  exact ⟨w, hwP⟩

/-- Board-level certificate for a first move: one P child proves the root is N. -/
theorem firstPlayerWins_of_move_to_not_win {G : Graph k} {v : Fin k}
    (hchild : ¬ win G (child G Finset.univ v)) : firstPlayerWins G := by
  rw [firstPlayerWins, win.eq_def]
  exact ⟨⟨v, Finset.mem_univ v⟩, hchild⟩

/-! ## Compact reply-book artifact vocabulary

The n=20 queens plan needs a checker-facing datatype for compressed reply
books: automatic paired-core replies, explicit border/scar exception tables,
and terminal leaves.  The structures below are intentionally statement-level:
they record the artifact shape, while `Valid` is the semantic checker contract
that must be discharged by generated proofs or later refined sub-certificates.
-/

/-- A semantic P-claim for a live set. -/
abbrev IsP (G : Graph k) (S : Finset (Fin k)) : Prop :=
  ¬ win G S

/-- A checked reply choice for one opponent move. -/
structure CertifiedReply (G : Graph k) (S : Finset (Fin k)) (v : S) where
  reply : child G S (v : Fin k)
  childP : IsP G (child G (child G S (v : Fin k)) (reply : Fin k))

/-- A checked reply for every legal opponent move from `S`. -/
abbrev ReplyStrategyCertificate (G : Graph k) (S : Finset (Fin k)) : Type :=
  ∀ v : S, CertifiedReply G S v

/-- A strategy-style reply certificate proves that `S` is a P-position. -/
theorem not_win_of_replyStrategyCertificate {G : Graph k} {S : Finset (Fin k)}
    (hcert : ReplyStrategyCertificate G S) : IsP G S := by
  exact not_win_of_replyCertificate (fun v =>
    let r := hcert v
    ⟨r.reply, r.childP⟩)

/-- Metadata for one explicit exception-table entry. -/
structure ExceptionEntryData (k : ℕ) where
  move : Fin k
  reply : Fin k

/-- Metadata for unresolved extractor output. Unresolved nodes are not final certificates. -/
structure UnresolvedLeafData (k : ℕ) where
  live : Finset (Fin k)
  triedReplies : List (ExceptionEntryData k) := []
  note : String := ""

/-- Terminal claim kinds allowed by the central-child certificate plan. -/
inductive TerminalClaim (k : ℕ) where
  | s1Leaf (pairs : Finset (Fin k × Fin k))
  | tauSymmetricLeaf (tau : Fin k → Fin k)
  | solvedLeaf (nodes : ℕ)

namespace TerminalClaim

/--
Checker contract for terminal leaves.  The current scaffold records the final
semantic obligation; later versions can replace each case with concrete S1
pairing, tau-pairing, or dense-leaf checkers without changing the outer
certificate theorem.
-/
def Valid (G : Graph k) (S : Finset (Fin k)) (_claim : TerminalClaim k) : Prop :=
  IsP G S

end TerminalClaim

/-- Metadata for the automatic tau paired-core node kind. -/
structure PairedCoreData (k : ℕ) where
  core : Finset (Fin k)
  border : Finset (Fin k)
  tau : Fin k → Fin k
  borderEntries : List (ExceptionEntryData k) := []
  scarEntries : List (ExceptionEntryData k) := []

/-- Metadata for an explicit exception-table node. -/
structure ExceptionTableData (k : ℕ) where
  entries : List (ExceptionEntryData k)

/-- Final certificate nodes. Unresolved leaves are intentionally excluded. -/
inductive FinalCertificate (G : Graph k) (S : Finset (Fin k)) where
  | pairedCore (data : PairedCoreData k)
  | exceptionTable (data : ExceptionTableData k)
  | terminal (claim : TerminalClaim k)

namespace FinalCertificate

/--
Semantic checker contract for a final certificate.  Non-terminal compressed
nodes must provide a checked reply for every legal opponent move; terminal
nodes must discharge their terminal P-claim.
-/
def Valid {G : Graph k} {S : Finset (Fin k)} : FinalCertificate G S → Prop
  | pairedCore _ => ReplyCertificate G S
  | exceptionTable _ => ReplyCertificate G S
  | terminal claim => claim.Valid G S

/-- A valid final certificate proves the certified position is P. -/
theorem isP {G : Graph k} {S : Finset (Fin k)} (cert : FinalCertificate G S)
    (hvalid : cert.Valid) : IsP G S := by
  cases cert with
  | pairedCore data =>
      exact not_win_of_replyCertificate hvalid
  | exceptionTable data =>
      exact not_win_of_replyCertificate hvalid
  | terminal claim =>
      exact hvalid

end FinalCertificate

/--
Extractor output may contain unresolved leaves during development.  A final
checker must reject them by requiring `CertificateArtifact.Valid`.
-/
inductive CertificateArtifact (G : Graph k) (S : Finset (Fin k)) where
  | final (cert : FinalCertificate G S)
  | unresolved (leaf : UnresolvedLeafData k)

namespace CertificateArtifact

/-- Final validity rejects unresolved leaves. -/
def Valid {G : Graph k} {S : Finset (Fin k)} : CertificateArtifact G S → Prop
  | final cert => cert.Valid
  | unresolved _ => False

/-- A valid artifact proves the certified position is P. -/
theorem isP {G : Graph k} {S : Finset (Fin k)} (artifact : CertificateArtifact G S)
    (hvalid : artifact.Valid) : IsP G S := by
  cases artifact with
  | final cert =>
      exact cert.isP hvalid
  | unresolved leaf =>
      cases hvalid

end CertificateArtifact

end NodeKayles
