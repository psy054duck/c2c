from functools import reduce
from webbrowser import BackgroundBrowser
import sympy as sp
from sympy.logic.boolalg import Boolean

class Recurrence:
    def __init__(self, conditions: list[Boolean], transitions: list[dict[sp.Symbol, sp.Expr]]):
        self.conditions = conditions
        self.transitions = transitions
        self.variables = reduce(set.union, [set(t.keys()) for t in transitions])

    def __str__(self):
        res = []
        for c, t in zip(self.conditions, self.transitions):
            res.append('(%s, %s)' % (c, t))
        return ', '.join(res)

    def kth_values(self, inits: dict[sp.Symbol, sp.Integer | int], k: int):
        if self.variables != set(inits.keys()):
            raise Exception('Not all initial values are provided')
        cur = inits.copy()
        for _ in range(k):
            for cond, tran in zip(self.conditions, self.transitions):
                if cond.subs(cur) == sp.S.true:
                    cur = {var: tran.setdefault(var, var).subs(cur) for var in tran}
                    break
        return cur

    def solve_with_inits(self, inits: dict[sp.Symbol, sp.Integer | int]):
        l = 2
        BOUND = 1000
        cur_inits = inits
        while l < BOUND:
            index_seq = self.get_index_seqs(cur_inits, l)

    def get_index_seq(self, inits: dict[sp.Symbol, sp.Integer | int], l):
        index_seq = []
        cur = {k: sp.Integer(inits[k]) for k in inits}
        for _ in range(l):
            for i, (cond, tran) in enumerate(zip(self.conditions, self.transitions)):
                if cond.subs(cur) == sp.S.true:
                    index_seq.append(i)
                    cur = {var: tran.setdefault(var, var).subs(cur) for var in tran}
                    break
        return index_seq

    def solve_periodic(self, periodic_seg: list[int]):
        linearized_transitions = Recurrence.to_linear(self.transitions)

    @staticmethod
    def solve_linear_rec(linear_transition: dict[sp.Symbol, sp.Expr]):
        matrix_form = Recurrence.linear2matrix(linear_transition)

    @staticmethod
    def to_linear(transitions):
        pass

    @staticmethod
    def linear2matrix(linear_transition: dict[sp.Symbol, sp.Expr]) -> sp.Matrix:
        pass
