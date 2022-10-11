from z3 import *

_n, _t0, _t1, i = Ints('_n _t0 _t1 i')
e = And((_n >= 0), Or((_t0 <= 319998), (_t0 > 319999), (_t1 < i)))
sim = Tactic('ctx-solver-simplify')
e2 = Or(_t0 <= 319998, _t0 > 319999)
print(sim(e2))