from functools import reduce
import re
from webbrowser import BackgroundBrowser
import sympy as sp
from sympy.logic.boolalg import Boolean

class Recurrence:

    inductive_var = sp.Symbol('_n', Integer=True)

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
        periodic_transition = {var: var for var in self.variables}
        for i in periodic_seg:
            cur_transition = linearized_transitions[i]
            periodic_transition = Recurrence.symbolic_transition(periodic_transition, cur_transition)
        periodic_closed_form = Recurrence.solve_linear_rec(periodic_transition)
        mod_closed_form = [periodic_closed_form]
        for i in periodic_seg[:-1]:
            cur_transition = linearized_transitions[i]
            cur_value = mod_closed_form[-1]
            mod_closed_form.append(Recurrence.symbolic_transition(cur_value, cur_transition))
        for i in range(len(mod_closed_form)):
            mod_closed_form[i] = {v: mod_closed_form[i][v].subs({Recurrence.inductive_var: (Recurrence.inductive_var - i)/len(periodic_seg)}) for v in mod_closed_form[i]}
        print(mod_closed_form)

    @staticmethod
    def symbolic_transition(cur_value: dict[sp.Symbol, sp.Expr], transition: dict[sp.Symbol, sp.Expr]):
        return {var: transition.setdefault(var, var).subs(cur_value) for var in cur_value}

    @staticmethod
    def solve_linear_rec(linear_transition: dict[sp.Symbol, sp.Expr]):
        ordered_vars, matrix_form = Recurrence.linear2matrix(linear_transition)
        matrix_closed_form = sp.MatPow(matrix_form, Recurrence.inductive_var, evaluate=True)
        assert(not isinstance(matrix_closed_form, sp.MatPow))
        return Recurrence.matrix2linear(ordered_vars, matrix_closed_form)

    @staticmethod
    def to_linear(transitions: list[dict[sp.Symbol, sp.Expr]]):
        return transitions

    @staticmethod
    def matrix2linear(ordered_vars: list[sp.Symbol], matrix: sp.Matrix):
        res = {}
        for i, v1 in enumerate(ordered_vars):
            res |= {v1: sum([matrix[i, j]*v2 for j, v2 in enumerate(ordered_vars)]) + matrix[i, -1]}
        return res

    @staticmethod
    def linear2matrix(linear_transition: dict[sp.Symbol, sp.Expr]) -> tuple[list[sp.Symbol], sp.Matrix]:
        ordered_vars = list(linear_transition.keys())
        dim = len(ordered_vars)
        res = sp.eye(dim + 1)
        for i, v1 in enumerate(ordered_vars):
            const_term = linear_transition[v1].subs({v: sp.Integer(0) for v in ordered_vars})
            res[i, dim] = const_term
            for j, v2 in enumerate(ordered_vars):
                mask_dict = {v: sp.Integer(0) for v in ordered_vars} | {v2: sp.Integer(1)}
                res[i, j] = linear_transition[v1].subs(mask_dict) - const_term
        return ordered_vars, res

