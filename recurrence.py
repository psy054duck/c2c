from cgitb import small
from functools import reduce
import re
import sympy as sp
from sympy.logic.boolalg import Boolean
import random
import z3

class Recurrence:

    inductive_var = sp.Symbol('_n', Integer=True)

    def __init__(self, conditions: list[Boolean], transitions: list[dict[sp.Symbol, sp.Expr]]):
        self.conditions = conditions
        self.conditions = [conditions[0]]
        for i, cond in enumerate(conditions[1:]):
            self.conditions.append(sp.And(sp.Not(self.conditions[-1]), cond))
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

    def solve(self):
        cur_initals = {var: sp.Integer(random.randint(-10, 10)) for var in self.variables}
        _, index_seq = self.solve_with_inits(cur_initals)
        ks = [z3.Int('_k%d' % i) for i in range(len(cur_initals) - 1)]


    def solve_with_inits(self, inits: dict[sp.Symbol, sp.Integer | int]):
        l = 10
        BOUND = 1000
        cur_inits = inits
        res = []
        smallest = 0
        res_index_seq = []
        while l < BOUND:
            l *= 2
            index_seq, values = self.get_index_seq(cur_inits, l)
            threshold, periodic_seg = generate_periodic_seg(index_seq)
            non_periodic_closed_form = None
            cur_threshold = threshold

            non_periodic_res_index = []
            non_periodic_res = []
            prev_threshold = threshold
            while cur_threshold != 0:
                cur_threshold, cur_periodic_seg = generate_periodic_seg(index_seq[:prev_threshold])
                if len(cur_periodic_seg) == 0:
                    non_periodic_res_index = [index_seq[:cur_threshold]] + non_periodic_res_index
                    cur_closed_form = self.solve_periodic(index_seq[:cur_threshold])
                else:
                    non_periodic_res_index.append(cur_periodic_seg)
                    cur_closed_form = self.solve_periodic(cur_periodic_seg)
                if len(cur_periodic_seg) == 0:
                    cur_closed_form = [{v: closed_form[v].subs(values[0]) for v in closed_form} for closed_form in cur_closed_form]
                    non_periodic_res = [(Recurrence.inductive_var < cur_threshold + smallest, cur_closed_form)] + non_periodic_res
                    break
                else:
                    cur_closed_form = [{v: closed_form[v].subs(values[cur_threshold]) for v in closed_form} for closed_form in cur_closed_form]
                    non_periodic_res.append((Recurrence.inductive_var < smallest + threshold, cur_closed_form))
                # res.append((Recurrence.inductive_var < threshold + smallest, non_periodic_closed_form))
                prev_threshold = cur_threshold
            res += non_periodic_res
            res_index_seq.extend(non_periodic_res_index)
            periodic_closed_form = self.solve_periodic(periodic_seg)
            periodic_closed_form = [{v: closed_form[v].subs(values[threshold] | {Recurrence.inductive_var: Recurrence.inductive_var - threshold}) for v in closed_form} for closed_form in periodic_closed_form]
            cur_smallest = self.validate(periodic_closed_form, periodic_seg)
            res_index_seq.append(periodic_seg)
            if cur_smallest == -1:
                return res + [(sp.S.true, periodic_closed_form)], res_index_seq
            else:
                res += [(Recurrence.inductive_var < smallest + cur_smallest, periodic_closed_form)]
            cur_inits = self.kth_values(values[smallest], cur_smallest)
            smallest = smallest + cur_smallest

    def validate(self, closed_form: list[dict[sp.Symbol, sp.Expr]], periodic_seg: list[int]):
        smallest = []
        non_neg_interval = sp.Interval(0, sp.oo)
        for i, cond in enumerate(self.conditions):
            for j, p in enumerate(periodic_seg):
                to_validate = cond.subs(Recurrence._transform_closed_form(closed_form[j], {Recurrence.inductive_var: len(periodic_seg)*Recurrence.inductive_var + j}))
                to_validate = sp.simplify(to_validate)
                # to_validate = cond.subs(closed_form[j].subs({Recurrence.inductive_var: Recurrence.inductive_var + j}).subs({Recurrence.inductive_var: len(periodic_seg)*Recurrence.inductive_var}))
                # if to_validate != sp.S.true and to_validate != sp.S.false:
                #     raise Exception('fail to simplify the validation condition %s' % to_validate)
                if (i == p and to_validate != sp.S.true):
                    interval = to_validate.as_set().intersect(non_neg_interval)
                    if not interval.is_superset(non_neg_interval):
                        if interval.boundary.contains(0):
                            int_in_boundary = {sp.floor(v) for v in interval.boundary if interval.contains(v)}
                            min_v = min(int_in_boundary - {sp.Integer(0)})
                        else:
                            min_v = sp.Integer(0)
                        smallest.append(min_v*len(periodic_seg) + j)
                elif (i != p and to_validate != sp.S.false):
                    interval = to_validate.as_set().intersect(non_neg_interval)
                    if not interval.is_empty:
                        min_v = sp.ceiling(interval.inf)
                        if not interval.contains(min_v):
                            min_v += 1
                        smallest.append(min_v*len(periodic_seg) + j)
        return min(smallest, default=-1)

    @staticmethod
    def _transform_closed_form(closed_form, pairs):
        return {k: closed_form[k].subs(pairs) for k in closed_form}

    def get_index_seq(self, inits: dict[sp.Symbol, sp.Integer | int], l):
        index_seq = []
        cur = {k: sp.Integer(inits[k]) for k in inits}
        values = [cur]
        for _ in range(l):
            for i, (cond, tran) in enumerate(zip(self.conditions, self.transitions)):
                if cond.subs(cur) == sp.S.true:
                    index_seq.append(i)
                    cur = cur | {var: tran.setdefault(var, var).subs(cur) for var in tran}
                    values.append(cur)
                    break
        return index_seq, values

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
        return mod_closed_form

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

def generate_periodic_seg(index_seq):
    best_periodic_seg = []
    best_threshold = 0
    reversed_index_seq = list(reversed(index_seq))
    for i in range(1, len(index_seq)//2 + 1):
        candidate = reversed_index_seq[:i]
        for j in range(len(reversed_index_seq)):
            if candidate[j % i] != reversed_index_seq[j]:
                if j >= 2*i and j > best_threshold:
                    best_periodic_seg = reversed_index_seq[j - i:j]
                    best_threshold = j
                break
        else:
            return 0, reversed_index_seq[j - i:j]
    return len(index_seq) - best_threshold, list(reversed(best_periodic_seg))

if __name__ == '__main__':
    print(generate_periodic_seg([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]))