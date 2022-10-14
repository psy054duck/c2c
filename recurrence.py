from functools import reduce
from os import scandir
import random
import sympy as sp
from sympy.logic.boolalg import Boolean, true, false
import z3
from closed_form import Closed_form
from utils import check_conditions_consistency, to_z3, to_sympy, get_app_by_var, solve_k, z3_all_vars, my_sp_simplify

class Recurrence:

    inductive_var = sp.Symbol('_n', integer=True)
    neg_ind_var = sp.Symbol('_d', integer=True)

    def __init__(self, inits: dict[sp.Symbol, sp.Expr], conditions: list[Boolean], transitions: list[dict[sp.Symbol, sp.Expr]], ind_var=sp.Symbol('_n', integer=True), acc_transitions=None, e_transitions=None, bounded_vars=None):
        self.ind_var = ind_var
        self.bounded_vars = bounded_vars
        self.conditions = [conditions[0]]
        for cond in conditions[1:]:
            self.conditions.append(sp.simplify(sp.And(sp.Not(self.conditions[-1]), cond)))
        all_cond = sp.simplify(sp.Or(*self.conditions))

        self.transitions = transitions
        self.variables = reduce(set.union, [set(k for k in t.keys()) for t in transitions])
        self.variables = self.variables.union(reduce(set.union, [cond.free_symbols for cond in self.conditions]))
        self.acc_transitions = acc_transitions
        self.e_transitions = e_transitions
        self._combine_branches()
        self.inits = {var: inits[var] for var in inits if var in self.variables}
        self.sum_end = sp.Symbol('sum_end', integer=True)
        # self.variables = self.variables - {self.ind_var}
        self.arity = {}
        for var in self.variables:
            for trans in self.transitions:
                for t in trans:
                    if var.name == t.name:
                        self.arity[var] = len(t.args)
                        break
        self.constraints = []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def print(self):
        for cond, tran in zip(self.conditions, self.transitions):
            print(cond)
            print('\t%s' % tran)

    def to_file(self, filename):
        with open(filename, 'w') as fp:
            s = ''
            for var in self.inits:
                s += '%s = %s;\n' % (var, self.inits[var])
            for i, (cond, trans) in enumerate(zip(self.conditions, self.transitions)):
                if i > 0:
                    s += 'else '
                s += 'if (%s) {\n' % cond
                for k in trans:
                    s += '\t%s = %s;\n' % (k, trans[k])
                s += '} '
            fp.write(s)

    def _combine_branches(self):
        new_conditions = []
        new_transitions = []
        new_acc_transitions = []
        for i in range(len(self.conditions)):
            cond1, trans1 = self.conditions[i], self.transitions[i]
            acc1 = self.acc_transitions[i] if self.acc_transitions is not None else 0
            if (self.acc_transitions is None and trans1 in new_transitions) or (trans1 in new_transitions and self.acc_transitions is not None and acc1 in new_acc_transitions): continue
            new_condition = cond1
            for j in range(i + 1, len(self.conditions)):
                cond2, trans2 = self.conditions[j], self.transitions[j]
                acc2 = 0 if self.acc_transitions is None else self.acc_transitions[j]
                if (self.acc_transitions is None and trans2 in new_transitions) or (trans2 in new_transitions and self.acc_transitions is not None and acc2 in new_acc_transitions): continue
                if trans1 == trans2 and acc1 == acc2:
                    new_condition = sp.Or(cond1, cond2)
            new_conditions.append(sp.simplify(new_condition))
            new_transitions.append(trans1)
            if self.acc_transitions is not None:
                new_acc_transitions.append(acc1)
        self.conditions = new_conditions
        self.transitions = new_transitions
        self.acc_transitions = new_acc_transitions if self.acc_transitions is not None else None

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
        # t_list = self._prepare_t()
        scalar_rec = self.to_scalar()
        scalar_closed_form = scalar_rec.solve()
        scalar_closed_form.simplify()
        neg_scalar_closed_form_sp = scalar_closed_form.subs({self.ind_var: self.ind_var - Recurrence.neg_ind_var - 1}).to_sympy()
        # new_conditions = [cond.subs(neg_scalar_closed_form_sp) for cond in self.conditions]
        new_rec, t_list, acc, e, array_var = self._t_transitions(neg_scalar_closed_form_sp)
        array_func = array_var.func
        # new_rec.add_constraint(to_z3(self.ind_var >= 0))
        new_rec.to_file(filename='tmp.txt')
        arr_part_closed_form = new_rec.solve()
        arr_part_closed_form = arr_part_closed_form.subs({arr_part_closed_form.ind_var: self.ind_var})
        # mid = sp.Symbol('mid', integer=True)
        # arr_part_closed_form.add_constraint(sp.And(*[t >= 0 for t in t_list]))
        # res = res.subs({self.ind_var: 800})
        # for cond in arr_part_closed_form.conditions:
        #     t0 = sp.Symbol('_t0', integer=True)
        #     t1 = sp.Symbol('_t1', integer=True)
        #     print(sp.simplify(sp.And(t0 >= 0, t1 >= 0, t0 < 320000, t1 < 320000, cond.subs({sp.Symbol('_n', integer=True): 320000, sp.Symbol('i', integer=True): 0}))))
        # print('*'*10)
        # for closed in arr_part_closed_form.closed_forms:
        #     print(closed)
        arr_part_closed_form.simplify()
        array_closed_form = self._form_array_closed_form(arr_part_closed_form, array_func, t_list, acc)
        return scalar_closed_form, array_closed_form

    def _form_array_closed_form(self, part_closed_form, arr_var, t_list, acc):
        arr_closed_forms = []
        e = sp.Symbol('_e', integer=True)
        for closed, cond in zip(part_closed_form.closed_forms, part_closed_form.conditions):
            arr_app = closed[e]*arr_var(*[closed[t] for t in t_list]) + closed[acc]
            arr_closed_forms.append({arr_var(*t_list): arr_app})
        return Closed_form(part_closed_form.conditions, arr_closed_forms, self.ind_var, self.sum_end, t_list)

    def extract_scalar_part(self):
        scalar_vars = {var for var in self.arity if self.arity[var] == 0}
        scalar_transitions = [{var: trans[var] for var in scalar_vars} for trans in self.transitions]
        return scalar_transitions

    def to_scalar(self):
        scalar_transitions = self.extract_scalar_part()
        scalar_rec = Recurrence(self.inits, self.conditions, scalar_transitions)
        return scalar_rec

    def _prepare_t(self):
        array_var = [var for var in self.arity if self.arity[var] >= 1][0] # assume there is only one array
        t_list = [sp.Symbol('_t%d' % i, integer=True) for i in range(self.arity[array_var])]
        return t_list, array_var

    def _t_transitions(self, scalar_closed_form):
        t_list, array_var = self._prepare_t()
        acc = sp.Symbol('_acc', integer=True)
        e = sp.Symbol('_e', integer=True)
        transitions = []
        new_conditions = []
        d = sp.Symbol('d_p', integer=True)
        scalar_closed_form = {var: scalar_closed_form[var].subs(Recurrence.neg_ind_var, d) for var in scalar_closed_form}
        acc_transitions = []
        # e_transitions = []
        for cond, trans in zip(self.conditions, self.transitions):
            t_trans = {var: get_app_by_var(var.func, trans[var]) for var in trans if self.arity.get(var, 0) >= 1}
            t_trans = {var: t_trans[var] if t_trans[var] is not None else var for var in t_trans}
            e_trans = {var: trans[var].coeff(t_trans[var]) for var in t_trans}
            acc_terms = {var: trans[var] - e_trans[var]*t_trans[var] for var in t_trans}
            # e_trans = {var: 1 for var in t_trans}
            # e_trans = {var: e_trans[var] if e_trans[var] != 0 else 3 for var in e_trans}
            # acc_terms = {var: acc_terms[var].subs(scalar_closed_form) for var in acc_terms}
            for app in t_trans:
                cond_arr = sp.And(*[sp.Eq(t, arg.subs(scalar_closed_form, simultaneous=True)) for t, arg in zip(t_list, app.args)])
                new_conditions.append(sp.simplify(sp.And(cond.subs(scalar_closed_form), cond_arr)))
                # new_conditions.append(sp.And(cond.subs(scalar_closed_form), cond_arr))
                new_trans = {t: self._expr2involving_t(t, arg1, arg2).subs(scalar_closed_form, simultaneous=True) for arg1, arg2, t in zip(app.args, t_trans[app].args, t_list)}
                # transitions.append(new_trans | {d: d + 1, acc: acc + acc_terms[app]})
                transitions.append(new_trans | {d: d + 1} | {e: e_trans[app]*e})
                # acc_transitions.append({acc: acc + acc_terms[app].subs(scalar_closed_form, simultaneous=True)})
                acc_transitions.append(acc_terms[app].subs(scalar_closed_form, simultaneous=True).subs(d, Recurrence.neg_ind_var))
                # e_transitions.append(e_trans[app])
                # transitions.append(new_trans | {d: d + 1})
        new_conditions.append(sp.simplify(sp.Not(sp.Or(*[cond for cond in new_conditions]))))

        transitions.append({t: t for t in t_list} | {d: d + 1} | {e: e})
        acc_transitions.append(0)
        # e_transitions.append(1)
        # transitions.append({t: t for t in t_list} | {d: d + 1})
        return Recurrence({d: 0, acc: 0, e: 1} | self.inits, new_conditions, transitions, ind_var=Recurrence.neg_ind_var, acc_transitions=acc_transitions), t_list, acc, e, array_var
        # return Recurrence({d: 0}, new_conditions, transitions, ind_var=Recurrence.neg_ind_var), t_list

    def _expr2involving_t(self, t, t_expr, expr):
        '''if t = t_expr, then make expr to be an equivalent expression involving t'''
        vars = t_expr.free_symbols
        vars_p = expr.free_symbols
        intersec_vars = vars.intersection(vars_p)
        if len(intersec_vars) == 0:
            return expr
        var = list(intersec_vars)[0]
        sol = sp.solve(t - t_expr, var, dict=True)
        res = expr
        if len(sol) == 1:
            res = expr.subs(sol[0])
        return res

    def solve(self):
        solver = z3.Solver()
        solver.add(*self.constraints)
        solver.add(*[to_z3(var) == to_z3(self.inits[var]) for var in self.inits])
        tot_closed_form = []
        while solver.check() == z3.sat:
            m = solver.model()
            cur_initals = {var: m.eval(z3.Int(str(var)), model_completion=True).as_long() for var in self.variables}
            _, index_seq = self.solve_with_inits(cur_initals)
            ks = [sp.Symbol('_k%d' % i, integer=True) for i in range(len(index_seq) - 1)]
            closed_forms = []
            acc_k = 0
            inits = self.inits
            for i, (seq, k) in enumerate(zip(index_seq, ks + [sp.oo])):
                cur_closed_form = self.solve_periodic(seq)
                cur_closed_form = self.periodic_closed_form2sympy(cur_closed_form)
                cur_closed_form = {var: cur_closed_form[var].subs(inits).subs(self.ind_var, self.ind_var - acc_k) for var in cur_closed_form}
                acc_k += k
                inits = {var: cur_closed_form[var].subs(self.ind_var, acc_k) for var in cur_closed_form}
                closed_forms.append(cur_closed_form)

            z3_qe = z3.Then('qe', 'ctx-solver-simplify', 'ctx-simplify')
            z3_ind_var = z3.Int(str(self.ind_var))

            validation_conditions = z3.BoolVal(True)
            z3_ks = [to_z3(k) for k in ks]
            for i, k in enumerate(z3_ks):
                cur_closed_form = closed_forms[i]
                if i > 0:
                    acc_k = sum(len(index_seq[j])*ks[j] for j in range(i))
                    cur_closed_form = {var: sp.cancel(cur_closed_form[var].subs(self.ind_var, self.ind_var + acc_k)) for var in cur_closed_form}
                else:
                    cur_closed_form = {var: sp.cancel(cur_closed_form[var]) for var in cur_closed_form}
                seq = index_seq[i]
                for j, s in enumerate(seq):
                    shifted_ind_var = len(seq)*z3_ind_var + j
                    shifted_closed_form = {to_z3(var): z3.substitute(to_z3(cur_closed_form[var]), (z3_ind_var, shifted_ind_var)) for var in cur_closed_form}
                    validation_cond = z3.substitute(to_z3(self.conditions[s]), *[(var, shifted_closed_form[var]) for var in shifted_closed_form])
                    validation_cond = z3.ForAll(z3_ind_var, z3.Implies(z3.And(0 <= shifted_ind_var, shifted_ind_var < len(seq)*k), validation_cond))
                    validation_conditions = z3.And(validation_conditions, validation_cond, k >= 1)
            if len(z3_ks) > 0:
                acc_k = sum(len(index_seq[j])*ks[j] for j in range(len(z3_ks)))
                cur_closed_form = {var: sp.cancel(closed_forms[-1][var].subs(self.ind_var, self.ind_var + acc_k)) for var in closed_forms[-1]}
            else:
                cur_closed_form = {var: sp.cancel(closed_forms[-1][var]) for var in closed_forms[-1]}
            seq = index_seq[-1]
            for j, s in enumerate(seq):
                shifted_ind_var = len(seq)*self.ind_var + j
                shifted_closed_form = {to_z3(var): to_z3(cur_closed_form[var].subs(self.ind_var, shifted_ind_var)) for var in cur_closed_form}
                validation_cond = z3.substitute(to_z3(self.conditions[s]), *[(var, shifted_closed_form[var]) for var in shifted_closed_form])
                validation_cond = z3.ForAll(z3_ind_var, z3.Implies(z3.And(0 <= to_z3(shifted_ind_var)), validation_cond))
                validation_conditions = z3.And(validation_conditions, validation_cond)
            cnf = z3_qe(z3.simplify(validation_conditions))[0]

            res_ks = [solve_k(cnf, k, z3_ks) for k in z3_ks]
            constraint = [z3.substitute(c, *[(k, v) for k, v in zip(z3_ks, res_ks)]) for c in cnf]
            constraint = [z3.substitute(c, *[(to_z3(var), to_z3(self.inits[var])) for var in self.inits]) for c in constraint]
            constraint = z3.simplify(z3.And(*(constraint + self.constraints)))
            solver.add(z3.Not(constraint))
            subs_pairs1 = {k: to_sympy(k_z3) for k, k_z3 in zip(ks, res_ks)}
            subs_pairs2 = {var: self.inits[var] for var in self.inits}
            for i in range(len(closed_forms)):
                closed_form = {var: closed_forms[i][var].subs(subs_pairs1, simultaneous=True).subs(subs_pairs2, simultaneous=True) for var in closed_forms[i]}
                closed_forms[i] = closed_form
            res_ks_sympy = [sp.simplify(subs_pairs1[var].subs(subs_pairs2, simultaneous=True)) for var in ks]
            constraint_sp = to_sympy(constraint)
            debug_s = z3.Solver()
            res1 = debug_s.check(constraint, z3.Not(to_z3(to_sympy(constraint))))
            res2 = debug_s.check(z3.Not(constraint), to_z3(to_sympy(constraint)))
            if res1 == z3.sat or res2 == z3.sat:
                print(constraint)
                print(to_sympy(constraint))

            tot_closed_form.append((closed_forms, constraint_sp, res_ks_sympy, index_seq, self.acc_transitions))
        res = self._tot_closed_form2class(tot_closed_form)
        return res
        
    def _tot_closed_form2class(self, tot_closed_forms):
        conditions = []
        res_closed_forms = []
        for closed_forms, cond, ks, index_seq, acc_transition in tot_closed_forms:
            # prev_k = 0
            # prev_seq = []
            acc_k = 0
            acc = sp.Integer(0)
            for closed, k, seq in zip(closed_forms, ks + [sp.oo], index_seq):
                conditions.append(sp.simplify(sp.And(cond, acc_k <= self.ind_var, self.ind_var < acc_k + len(seq)*k)))
                if acc_transition is not None:
                    assert(len(seq) == 1)
                    for i in seq:
                        e = closed[sp.Symbol('_e', integer=True)]
                        if k is not sp.oo and k.is_constant():
                            # acc += sp.summation(acc_transition[i], (self.ind_var, acc_k, (acc_k + len(seq)*k - 1) if k is not sp.oo else self.sum_end - 1))
                            acc = acc + sum(e.subs(self.ind_var, acc_k + j)*acc_transition[i].subs(self.ind_var, acc_k + j) for j in range(len(seq)*k))
                        else:
                            acc = acc + sp.summation(e*acc_transition[i], (self.ind_var, acc_k, (acc_k + len(seq)*k - 1) if k is not sp.oo else self.sum_end - 1))
                    res_closed_forms.append(closed | {sp.Symbol('_acc', integer=True): acc.doit()})
                else:
                    res_closed_forms.append(closed)
                acc_k += len(seq)*k
        return Closed_form(conditions, res_closed_forms, self.ind_var, self.sum_end)
        
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
                # cur_threshold, cur_periodic_seg = generate_periodic_seg(index_seq[:prev_threshold])
                cur_threshold, cur_periodic_seg = split_non_periodic_seg(index_seq[:prev_threshold])
                if len(cur_periodic_seg) == 0:
                    non_periodic_res_index = [index_seq[:cur_threshold]] + non_periodic_res_index
                    cur_closed_form = self.solve_periodic(index_seq[:cur_threshold])
                else:
                    # non_periodic_res_index.append(cur_periodic_seg)
                    non_periodic_res_index = [cur_periodic_seg] + non_periodic_res_index
                    cur_closed_form = self.solve_periodic(cur_periodic_seg)
                if len(cur_periodic_seg) == 0:
                    cur_closed_form = [{v: closed_form[v].subs(values[0]) for v in closed_form} for closed_form in cur_closed_form]
                    non_periodic_res = [(self.ind_var < cur_threshold + smallest, cur_closed_form)] + non_periodic_res
                    break
                else:
                    cur_closed_form = [{v: closed_form[v].subs(values[cur_threshold]) for v in closed_form} for closed_form in cur_closed_form]
                    # non_periodic_res.append((self.ind_var < smallest + threshold, cur_closed_form))
                    non_periodic_res = [(self.ind_var < smallest + threshold, cur_closed_form)] + non_periodic_res
                prev_threshold = cur_threshold
            res += non_periodic_res
            res_index_seq.extend(non_periodic_res_index)
            periodic_closed_form = self.solve_periodic(periodic_seg)
            validate_closed_form = [{v: closed_form[v].subs(values[threshold]) for v in closed_form} for closed_form in periodic_closed_form]
            cur_smallest = self.validate(validate_closed_form, periodic_seg)
            periodic_closed_form = [{v: closed_form[v].subs(values[threshold] | {self.ind_var: self.ind_var - threshold}) for v in closed_form} for closed_form in periodic_closed_form]
            res_index_seq.append(periodic_seg)
            if cur_smallest == -1:
                return res + [(sp.S.true, periodic_closed_form)], res_index_seq
            else:
                res += [(self.ind_var < smallest + cur_smallest + threshold, periodic_closed_form)]
            cur_inits = self.kth_values(values[smallest], cur_smallest)
            smallest = smallest + cur_smallest

    def validate(self, closed_form: list[dict[sp.Symbol, sp.Expr]], periodic_seg: list[int]):
        smallest = []
        non_neg_interval = sp.Interval(0, sp.oo)
        for i, cond in enumerate(self.conditions):
            for j, p in enumerate(periodic_seg):
                to_validate = cond.subs(Recurrence._transform_closed_form(closed_form[j], {self.ind_var: len(periodic_seg)*self.ind_var + j}))
                to_validate = sp.simplify(to_validate)
                if (i == p and to_validate != sp.S.true):
                    s = z3.Solver()
                    z3_ind_var = to_z3(self.ind_var)
                    N = z3.Int('N')
                    s.add(z3.ForAll(z3_ind_var, z3.Implies(z3.And(0 <= z3_ind_var, z3_ind_var < N), to_z3(to_validate))))
                    s.add(z3.Not(z3.substitute(to_z3(to_validate), (z3_ind_var, N))))
                    s.add(N >= 0)
                    if s.check() == z3.sat:
                        m = s.model()
                        min_v = to_sympy(m[N])
                        smallest.append(min_v*len(periodic_seg) + j)
                    # else:
                    #     raise Exception('It should not be unsat')
                    # interval = to_validate.as_set().intersect(non_neg_interval)
                    # if not interval.is_superset(non_neg_interval):
                    #     if interval.boundary.contains(0):
                    #         int_in_boundary = {sp.floor(v) for v in interval.boundary if interval.contains(v)}
                    #         print(int_in_boundary)
                    #         min_v = min(int_in_boundary - {sp.Integer(0)})
                    #     else:
                    #         min_v = sp.Integer(0)
                    #     smallest.append(min_v*len(periodic_seg) + j)
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
        periodic_closed_form = Recurrence.solve_linear_rec(periodic_transition, self.ind_var)
        mod_closed_form = [periodic_closed_form]
        for i in periodic_seg[:-1]:
            cur_transition = linearized_transitions[i]
            cur_value = mod_closed_form[-1]
            mod_closed_form.append(Recurrence.symbolic_transition(cur_value, cur_transition))
        for i in range(len(mod_closed_form)):
            mod_closed_form[i] = {v: mod_closed_form[i][v].subs({self.ind_var: (self.ind_var - i)/len(periodic_seg)}) for v in mod_closed_form[i]}
        return mod_closed_form

    def periodic_closed_form2sympy(self, closed_form: list[dict[sp.Symbol, sp.Expr]]):
        res = {}
        if len(closed_form) == 1:
            return closed_form[0]
        for var in self.variables:
            res[var] = sp.Piecewise(*([(closed[var], sp.Eq(self.ind_var % len(closed_form), i)) for i, closed in enumerate(closed_form)] + [(-1, True)]))
        return res

    @staticmethod
    def symbolic_transition(cur_value: dict[sp.Symbol, sp.Expr], transition: dict[sp.Symbol, sp.Expr]):
        return {var: transition.setdefault(var, var).subs(cur_value) for var in cur_value}

    @staticmethod
    def solve_linear_rec(linear_transition: dict[sp.Symbol, sp.Expr], ind_var):
        ordered_vars, matrix_form = Recurrence.linear2matrix(linear_transition)
        # matrix_closed_form = sp.MatPow(matrix_form, ind_var, evaluate=True)
        P, J = matrix_form.jordan_form()
        _, cells = J.jordan_cells()
        cells = [sp.MatPow(m, ind_var, evaluate=True) if not m.is_zero_matrix else m for m in cells]
        shape = J.shape
        middle = sp.zeros(*shape)
        x, y = 0, 0
        for cell in cells:
            cell_x, cell_y = cell.shape
            for i in range(cell_x):
                for j in range(cell_y):
                    middle[x + i, y + j] = cell[i, j]
            x += cell_x
            y += cell_y
        # matrix_closed_form = (P*middle*P.inv()).subs(3**ind_var, 3).subs(0**ind_var, 0)
        matrix_closed_form = P*middle*P.inv()
        if 0**ind_var in matrix_closed_form:
            matrix_closed_form = matrix_closed_form.subs(0**ind_var, 0)
        linear_form = Recurrence.matrix2linear(ordered_vars, matrix_closed_form)
        if matrix_form.rank() != matrix_form.shape[0]:
            # linear_form = {var: sp.Piecewise((var, sp.Eq(ind_var, 0)), (linear_form[var], True)) if linear_form[var].subs(ind_var, 0) != var else linear_form[var] for var in linear_form}
            linear_form = {var: sp.Piecewise((var, sp.Eq(ind_var, 0)), (linear_form[var], True)) for var in linear_form}
        return linear_form


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
    best_periodic_seg = [index_seq[-1]]
    best_threshold = 1
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

def split_non_periodic_seg(index_seq):
    if len(index_seq) == 2:
        return len(index_seq) - 1, [index_seq[1]]
    elif len(index_seq) == 1:
        return 0, [index_seq[0]]
    else:
        return generate_periodic_seg(index_seq)




def my_simplify_sp(expr):
    return sp.simplify(expr)
    if isinstance(expr, sp.Piecewise):
        new_args = [(sp.simplify(e), sp.simplify(c)) for e, c in expr.args]
        print(new_args)
        return sp.Piecewise(*new_args)
    return sp.simplify(expr)







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
