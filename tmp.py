from sympy import *
s = Symbol('s')
e = (s > 0) & (s < 1)
print(simplify(~e))
from utils import to_sympy
from z3 import *

_t0 = Int('_t0')
_n = Int('_n')
i = Int('i')
_t1 = Int('_t1')
e = Not(And(_t0 <= 319999,
        Not(And(Not(_t0 <= 319998),
                Not(319999 <= _t0),
                Not(319999 == _t0))),
        -1 <= _t0,
        _n + i + -1*_t1 >= 1))
e1 = Not(And(Not(_t0 <= 319998),
                Not(319999 <= _t0),
                Not(319999 == _t0)))
to_sympy(e1)


s = Solver()
res = s.check((And(Not(_t0 <= 319998),
                Not(319999 <= _t0),
                Not(319999 == _t0))))

# _t0 > 319999 | (t0 > 319998 & t0 < 319999 & t0 != 319999)
# (_t0 > 319999) | (_t0 < -1) | ((_t0 >= 319998) & (_t0 < 319999)) | (_n - _t1 + i < 1)