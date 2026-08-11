# OEIS A344227 — the extension package

[A344227](https://oeis.org/A344227) records the Sprague–Grundy values of the Non-Attacking Queens
game, which is Node Kayles on the n-queens graph. As catalogued it runs to `a(13)`. The solver in
this repository computes four further terms:

```
a(14) = 0,  a(15) = 1,  a(16) = 0,  a(17) = 2
```

`a(17) = 2` refutes the conjecture recorded in the entry's comments, that the sequence oscillates
between 1 and 0 after the ninth term. The oscillation holds through `a(16)` and breaks at `a(17)`;
the even side was already known to break, since the 18×18 game is a first-player win and therefore
`G(18) ≠ 0`.

| File                    | Contents                                                                     |
|-------------------------|------------------------------------------------------------------------------|
| `submission-package.md` | every field proposed for the entry — DATA, b-file, comments, links, extensions, crossrefs — with the consistency check against the catalogued terms and the answers to the questions a reviewer of a game-value sequence tends to ask |
| `b344227.txt`           | the b-file for `a(0)..a(17)`, offset-aligned                                  |
| `conjecture-theory.md`  | the conjectures around the sequence, a literature survey, and a structural theory of the even/odd split with the proof status of each step marked |

The run configurations behind the new terms are in
[../../verification/RESULTS.md](../../verification/RESULTS.md), and the validation gates behind the
solver are in [../../verification/README.md](../../verification/README.md).
