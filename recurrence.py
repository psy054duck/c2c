from functools import reduce
from webbrowser import BackgroundBrowser
import sympy as sp

class Recurrence:
    def __init__(self, conditions, transitions):
        self.conditions = conditions
        self.transitions = transitions
        self.variables = reduce(set.union, [set(t.keys()) for t in transitions])

    def __str__(self):
        res = []
        for c, t in zip(self.conditions, self.transitions):
            res.append('(%s, %s)' % (c, t))
        return ', '.join(res)

    def kth_values(self, inits, k):
        if self.variables != set(inits.keys()):
            raise Exception('Not all initial values are provided')
        cur = inits.copy()
        for _ in range(k):
            for cond, tran in zip(self.conditions, self.transitions):
                if cond.subs(cur) == sp.S.true:
                    cur = {var: tran.setdefault(var, var).subs(cur) for var in tran}
                    break
        return cur

    def solve_with_inits(self, inits):
        l = 2
        BOUND = 1000
        cur_inits = inits
        while l < BOUND:
            index_seq = self.get_index_seqs(cur_inits, l)

    def get_index_seq(self, inits, l):
        index_seq = []
        cur = inits.copy()
        for _ in range(l):
            for i, (cond, tran) in enumerate(zip(self.conditions, self.transitions)):
                if cond.subs(cur) == sp.S.true:
                    index_seq.append(i)
                    cur = {var: tran.setdefault(var, var).subs(cur) for var in tran}
                    break
        return index_seq

    def solve_periodic(self, periodic_seg):
        linearized_transitions = self.to_linear(self.transitions)

    def to_linear(self, transitions):
        pass