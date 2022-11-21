from z3 import *
_t0 = Int('_t0')
expr = And((_t0 <= 31999), Or((_t0 == 0), (_t0 >= 31999)))
print(simplify(expr))
sim = Repeat(Then('nnf', 'ctx-solver-simplify'))
print(sim(expr))