from z3 import *

_d = Int('_d')
_k0 = Int('_k0')
_t0 = Int('_t0')
_n = Int('_n')
# e = # And(And(True,
    #     ForAll(_d,
    #            Implies(And(1*_d + 0 >= 0, 1*_d + 0 < 1*_k0),
    #                    -1 ==
    #                    If(0 == 1*_d + 0,
    #                       _t0,
    #                       _n + -1*(1*_d + 0)) +
    #                    If(0 == 1*_d + 0, 0, 1*_d + 0) +
    #                    -1*_n)),
    #     _k0 >= 1))# ,
e = ForAll(_d,
           Implies(And(_d >= 0),
                   Not(-1 ==
                       If(0 == _k0, _t0, _n + -1*_k0) +
                       _d +
                       -1*_k0 +
                       -1*_n)))
s = Solver()
s.add(e)
s.add(_k0 >= 1)
res = s.check()
print(res)