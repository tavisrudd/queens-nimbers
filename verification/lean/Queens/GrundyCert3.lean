import NodeKayles.GrundyCertificate
import Queens.Basic

/-! Generated C50 prototype: a reflected Grundy book for 3×3 queens. -/

namespace Queens.GrundyCert3

abbrev V := Fin 9

def live (mask : ℕ) : Finset V :=
  Finset.univ.filter fun i => mask.testBit i.val

def mv (i : ℕ) (h : i < 9 := by omega) : V := ⟨i, h⟩

open NodeKayles
open NodeKayles.GrundyBookData

def nodes : List (GrundyBookNode 9) := [
  { position := live 0x1ff, value := 2,
    lowerMoves := movesOfList [mv 4, mv 0] (by decide) },
  { position := live 0x005, value := 1,
    lowerMoves := movesOfList [mv 0] (by decide) },
  { position := live 0x00a, value := 1,
    lowerMoves := movesOfList [mv 1] (by decide) },
  { position := live 0x022, value := 1,
    lowerMoves := movesOfList [mv 1] (by decide) },
  { position := live 0x041, value := 1,
    lowerMoves := movesOfList [mv 0] (by decide) },
  { position := live 0x088, value := 1,
    lowerMoves := movesOfList [mv 3] (by decide) },
  { position := live 0x0a0, value := 1,
    lowerMoves := movesOfList [mv 5] (by decide) },
  { position := live 0x104, value := 1,
    lowerMoves := movesOfList [mv 2] (by decide) },
  { position := live 0x140, value := 1,
    lowerMoves := movesOfList [mv 6] (by decide) },
  { position := live 0x000, value := 0,
    lowerMoves := movesOfList [] (by decide) }
]

def book : GrundyBookData 9 where
  root := live 0x1ff
  rootValue := 2
  nodes := nodes

/-- The generated artifact passes the reflected rules-only checker. -/
theorem check_book : book.check (queenGraph 3) = true := by decide

/-- End-to-end kernel-checked nimber for 3×3 queens. -/
theorem queen3_grundy : grundy (queenGraph 3) (live 0x1ff) = 2 := by
  exact root_grundy_eq_of_check check_book

theorem live_full : live 0x1ff = Finset.univ := by decide

/-- Board-level spelling of the certified 3×3 queens nimber. -/
theorem queen3_grundy_full : grundy (queenGraph 3) Finset.univ = 2 := by
  rw [← live_full]
  exact queen3_grundy

#print axioms queen3_grundy_full

end Queens.GrundyCert3
