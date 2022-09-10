from functools import reduce
from itertools import product
import re
import sympy as sp
from sympy.logic.boolalg import Boolean
import random
import z3
from closed_form import Closed_form

class Recurrence:

    inductive_var = sp.Symbol('_n', integer=True)

    def __init__(self, inits: dict[sp.Symbol, sp.Expr], conditions: list[Boolean], transitions: list[dict[sp.Symbol, sp.Expr]]):
        self.inits = inits
        self.conditions = [conditions[0]]
        for i, cond in enumerate(conditions[1:]):
            self.conditions.append(sp.And(sp.Not(self.conditions[-1]), cond))
        self.transitions = transitions
        self.variables = reduce(set.union, [set(k for k in t.keys()) for t in transitions])
        self.arity = {}
        for var in self.variables:
            for trans in self.transitions:
                for t in trans:
                    if var.name == t.name:
                        self.arity[var] = len(t.args)
                        break

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

    def solve_array(self):
        self.extract_scalar_part()

    def extract_scalar_part(self):
        scalar_vars = {var for var in self.arity if self.arity[var] == 0}
        scalar_transitions = [{var: trans[var] for var in scalar_vars} for trans in self.transitions]
        print(scalar_transitions)

    def solve(self):
        # cur_initals = {var: sp.Integer(random.randint(-10, 10)) for var in self.variables}
        solver = z3.Solver()
        solver.add(*[to_z3(var) == to_z3(self.inits[var]) for var in self.inits])
        tot_closed_form = []
        while solver.check() == z3.sat:
            m = solver.model()
            cur_initals = {var: m.eval(z3.Int(str(var)), model_completion=True).as_long() for var in self.variables}
            _, index_seq = self.solve_with_inits(cur_initals)
            ks = [sp.Symbol('_k%d' % i, integer=True) for i in range(len(index_seq) - 1)]
            closed_forms = []
            prev_closed_form = None
            prev_k = None
            for seq, k in zip(index_seq, ks):
                cur_closed_form = self.solve_periodic(seq)
                cur_closed_form = self.periodic_closed_form2sympy(cur_closed_form)
                if prev_closed_form is not None:
                    init_values = {var: prev_closed_form[var].subs(Recurrence.inductive_var, prev_k) for var in prev_closed_form}
                    cur_closed_form = {var: cur_closed_form[var].subs(init_values).subs(Recurrence.inductive_var, Recurrence.inductive_var - prev_k) for var in cur_closed_form}
                prev_closed_form = cur_closed_form
                prev_k = k
                closed_forms.append(cur_closed_form)
            cur_closed_form = self.solve_periodic(index_seq[-1])
            cur_closed_form = self.periodic_closed_form2sympy(cur_closed_form)
            if prev_closed_form is not None:
                init_values = {var: prev_closed_form[var].subs(Recurrence.inductive_var, prev_k) for var in prev_closed_form}
                cur_closed_form = {var: cur_closed_form[var].subs(init_values).subs(Recurrence.inductive_var, Recurrence.inductive_var - prev_k) for var in cur_closed_form}
            closed_forms.append(cur_closed_form)
            z3_qe = z3.Then('qe', 'ctx-solver-simplify', 'ctx-simplify')
            # z3_qe = z3.Tactic('qe')
            z3_ind_var = z3.Int(str(Recurrence.inductive_var))
            validation_conditions = True
            z3_ks = [to_z3(k) for k in ks]
            for i, k in enumerate(z3_ks):
                cur_closed_form = closed_forms[i]
                if i > 0:
                    prev_k = z3_ks[i-1]
                else:
                    prev_k = 0
                if i > 0:
                    cur_closed_form = {var: sp.cancel(cur_closed_form[var].subs(Recurrence.inductive_var, Recurrence.inductive_var - sp.Symbol(str(prev_k), integer=True))) for var in cur_closed_form}
                else:
                    cur_closed_form = {var: sp.cancel(cur_closed_form[var]) for var in cur_closed_form}
                seq = index_seq[i]
                for j, s in enumerate(seq):
                    shifted_ind_var = len(seq)*z3_ind_var + j
                    shifted_closed_form = {to_z3(var): z3.substitute(to_z3(cur_closed_form[var]), (z3_ind_var, shifted_ind_var)) for var in cur_closed_form}
                    validation_cond = z3.substitute(to_z3(self.conditions[s]), *[(var, shifted_closed_form[var]) for var in shifted_closed_form])
                    validation_cond = z3.ForAll(z3_ind_var, z3.Implies(z3.And(0 <= shifted_ind_var, shifted_ind_var< k), validation_cond))
                    validation_conditions = z3.And(validation_conditions, validation_cond, k >= 1)
            prev_k = z3_ks[-1] if len(z3_ks) > 0 else 0
            if len(z3_ks) > 0:
                cur_closed_form = {var: sp.cancel(closed_forms[-1][var].subs(Recurrence.inductive_var, Recurrence.inductive_var - sp.Symbol(str(prev_k), integer=True))) for var in closed_forms[-1]}
            else:
                cur_closed_form = {var: sp.cancel(closed_forms[-1][var]) for var in closed_forms[-1]}
            seq = index_seq[-1]
            for j, s in enumerate(seq):
                shifted_ind_var = len(seq)*Recurrence.inductive_var + j
                shifted_closed_form = {to_z3(var): to_z3(cur_closed_form[var].subs(Recurrence.inductive_var, shifted_ind_var)) for var in cur_closed_form}
                validation_cond = z3.substitute(to_z3(self.conditions[s]), *[(var, shifted_closed_form[var]) for var in shifted_closed_form])
                validation_cond = z3.ForAll(z3_ind_var, z3.Implies(z3.And(0 <= to_z3(shifted_ind_var)), validation_cond))
                validation_conditions = z3.And(validation_conditions, validation_cond)
            cnf = z3_qe(z3.simplify(validation_conditions))[0]
            res_ks = [solve_k(cnf, k) for k in z3_ks]
            constraint = [z3.substitute(c, *[(k, v) for k, v in zip(z3_ks, res_ks)]) for c in cnf]
            constraint = [z3.substitute(c, *[(to_z3(var), to_z3(self.inits[var])) for var in self.inits]) for c in constraint]
            constraint = z3.simplify(z3.And(*constraint))
            solver.add(z3.Not(constraint))
            subs_pairs1 = {k: to_sympy(k_z3) for k, k_z3 in zip(ks, res_ks)}
            subs_pairs2 = {var: self.inits[var] for var in self.inits}
            for i in range(len(closed_forms)):
                closed_form = {var: closed_forms[i][var].subs(subs_pairs1, simultaneous=True).subs(subs_pairs2, simultaneous=True) for var in closed_forms[i]}
                closed_forms[i] = closed_form
            res_ks_sympy = [subs_pairs1[var].subs(subs_pairs2, simultaneous=True) for var in ks]
            tot_closed_form.append((closed_forms, to_sympy(constraint), res_ks_sympy))
        res = Closed_form(tot_closed_form, Recurrence.inductive_var)
        return res
        
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
                prev_threshold = cur_threshold
            res += non_periodic_res
            res_index_seq.extend(non_periodic_res_index)
            periodic_closed_form = self.solve_periodic(periodic_seg)
            validate_closed_form = [{v: closed_form[v].subs(values[threshold]) for v in closed_form} for closed_form in periodic_closed_form]
            cur_smallest = self.validate(validate_closed_form, periodic_seg)
            periodic_closed_form = [{v: closed_form[v].subs(values[threshold] | {Recurrence.inductive_var: Recurrence.inductive_var - threshold}) for v in closed_form} for closed_form in periodic_closed_form]
            res_index_seq.append(periodic_seg)
            if cur_smallest == -1:
                return res + [(sp.S.true, periodic_closed_form)], res_index_seq
            else:
                res += [(Recurrence.inductive_var < smallest + cur_smallest + threshold, periodic_closed_form)]
            cur_inits = self.kth_values(values[smallest], cur_smallest)
            smallest = smallest + cur_smallest

    def validate(self, closed_form: list[dict[sp.Symbol, sp.Expr]], periodic_seg: list[int]):
        smallest = []
        non_neg_interval = sp.Interval(0, sp.oo)
        for i, cond in enumerate(self.conditions):
            for j, p in enumerate(periodic_seg):
                to_validate = cond.subs(Recurrence._transform_closed_form(closed_form[j], {Recurrence.inductive_var: len(periodic_seg)*Recurrence.inductive_var + j}))
                to_validate = sp.simplify(to_validate)
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

    def periodic_closed_form2sympy(self, closed_form: list[dict[sp.Symbol, sp.Expr]]):
        res = {}
        if len(closed_form) == 1:
            return closed_form[0]
        for var in self.variables:
            res[var] = sp.Piecewise(*[(closed[var], sp.Eq(Recurrence.inductive_var % len(closed_form), i)) for i, closed in enumerate(closed_form)])
        return res

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

def to_z3(sp_expr):
    self = sp.factor(sp_expr)
    if isinstance(self, sp.Add):
        res = sum([to_z3(arg) for arg in self.args])
    elif isinstance(self, sp.Mul):
        res = 1
        for arg in reversed(self.args):
            if arg.is_number and not arg.is_Integer:
                res = (res*arg.numerator)/arg.denominator
            else:
                res = res * to_z3(arg)
        return z3.simplify(res)
        # return reduce(lambda x, y: x*y, [to_z3(arg) for arg in reversed(self.args)])
    elif isinstance(self, sp.Piecewise):
        cond  = to_z3(self.args[0][1])
        res = z3.If(cond, to_z3(self.args[0][0]), to_z3(self.args[1][0]))
    elif isinstance(self, sp.And):
        res = z3.And(*[to_z3(arg) for arg in self.args])
    elif isinstance(self, sp.Or):
        res = z3.Or(*[to_z3(arg) for arg in self.args])
    elif isinstance(self, sp.Not):
        res = z3.Not(*[to_z3(arg) for arg in self.args])
    elif isinstance(self, sp.Gt):
        res = to_z3(self.lhs) > to_z3(self.rhs)
    elif isinstance(self, sp.Ge):
        res = to_z3(self.lhs) >= to_z3(self.rhs)
    elif isinstance(self, sp.Lt):
        res = to_z3(self.lhs) < to_z3(self.rhs)
    elif isinstance(self, sp.Le):
        res = to_z3(self.lhs) <= to_z3(self.rhs)
    elif isinstance(self, sp.Eq):
        res = to_z3(self.lhs) == to_z3(self.rhs)
    elif isinstance(self, sp.Integer) or isinstance(self, int):
        res = z3.IntVal(int(self))
    elif isinstance(self, sp.Symbol):
        res = z3.Int(str(self))
    elif isinstance(self, sp.Rational):
        # return z3.RatVal(self.numerator, self.denominator)
        res = z3.IntVal(self.numerator) / z3.IntVal(self.denominator)
    elif isinstance(self, sp.Mod):
        res = to_z3(self.args[0]) % to_z3(self.args[1])
    else:
        raise Exception('Conversion for "%s" has not been implemented yet' % type(self))
    return z3.simplify(res)

def to_sympy(expr):
    if z3.is_int_value(expr):
        res = expr.as_long()
    elif z3.is_const(expr):
        res = sp.Symbol(str(expr), integer=True)
    elif z3.is_add(expr):
        res = sum([to_sympy(arg) for arg in expr.children()])
    elif z3.is_sub(expr):
        children = expr.children()
        assert(len(children) == 2)
        res = to_sympy(children[0]) - to_sympy(children[1])
    elif z3.is_mul(expr):
        children = expr.children()
        res = reduce(lambda x, y: x*y, [to_sympy(ch) for ch in children])
    elif z3.is_gt(expr):
        children = expr.children()
        res = to_sympy(children[0]) > to_sympy(children[1])
    elif z3.is_lt(expr):
        children = expr.children()
        res = to_sympy(children[0]) < to_sympy(children[1])
    elif z3.is_ge(expr):
        children = expr.children()
        res = to_sympy(children[0]) >= to_sympy(children[1])
    elif z3.is_le(expr):
        children = expr.children()
        res = to_sympy(children[0]) <= to_sympy(children[1])
    elif z3.is_not(expr):
        children = expr.children()
        res = sp.Not(to_sympy(children[0]))
    elif z3.is_and(expr):
        children = expr.children()
        res = sp.And(*[to_sympy(ch) for ch in children])
    else:
        raise Exception('conversion for type "%s" is not implemented' % type(expr))
    return res

def my_simplify_sp(expr):
    return sp.simplify(expr)
    if isinstance(expr, sp.Piecewise):
        new_args = [(sp.simplify(e), sp.simplify(c)) for e, c in expr.args]
        print(new_args)
        return sp.Piecewise(*new_args)
    return sp.simplify(expr)

def solve_k(constraints, k):
    solver = z3.Solver()
    solver.add(*constraints)
    all_vars = list(reduce(lambda x, y: x.union(y), [z3_all_vars(expr) for expr in constraints]) - {k})
    all_vars_1 = all_vars + [z3.IntVal(1)]
    # det_vars = [z3.Real('c_%d' % i) for i in range(len(all_vars_1))]
    det_vars = [z3.Int('c_%d' % i) for i in range(len(all_vars_1))]
    template = sum([c*v for c, v in zip(det_vars, all_vars_1)])
    eqs = []
    while solver.check() == z3.sat:
        linear_solver = z3.Solver()
        for _ in range(len(all_vars_1)):
            solver.check()
            m = solver.model()
            eq = (m[k] == m.eval(template))
            eqs.append(eq)
            solver.add(*[z3.Not(var == m[var]) for var in all_vars + [k]])
        linear_solver.add(*eqs)
        linear_solver.check()
        m_c = linear_solver.model()
        cur_sol = m_c.eval(template)
        solver.add(z3.Not(k == cur_sol))
    return cur_sol

def z3_all_vars(expr):
    if z3.is_const(expr):
        if z3.is_int_value(expr):
            return set()
        else:
            return {expr}
    else:
        try:
            return reduce(lambda x, y: x.union(y), [z3_all_vars(ch) for ch in expr.children()])
        except:
            return set()




if __name__ == '__main__':
    # print(generate_periodic_seg([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]))
    # a = sp.Symbol('a')
    # k = sp.Symbol('k')
    # e = a + 1 - sp.Piecewise((a/2 + 1, sp.Eq(k % 10, 1)), (a + 100, sp.Eq(k % 10, 0)))
    # print(z3_all_vars(to_z3(e)))
    k = z3.Int('k')
    x, y = z3.Ints('x y')
    constraints = [k >= 1, z3.Not(x <= y + k), x <= 1 + y + k]
    # solve_k(constraints, k)
    print([to_sympy(c) for c in constraints])
