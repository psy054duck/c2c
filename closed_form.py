from functools import reduce
import sympy as sp
import z3
from utils import to_z3, to_sympy, z3_deep_simplify, expr2c, get_app_by_var, my_sp_simplify
import pycparser.c_ast as c_ast
from pycparser import c_generator
import recurrence

class Closed_form:
    def __init__(self, conditions, closed_forms, ind_var, sum_end, bounded_vars=None):
        self.conditions = [sp.S.true] if len(conditions) == 1 else conditions
        self.closed_forms = closed_forms
        self.ind_var = ind_var
        self.bounded_vars = bounded_vars
        self.sum_end = sum_end
        # self._simplify_conditions()

    def keep_live_vars(self, live_vars):
        for i in range(len(self.closed_forms)):
            closed = self.closed_forms[i]
            closed = {var: closed[var] for var in closed if var.name in live_vars}
            self.closed_forms[i] = closed

    def remove_vars(self, vars):
        for i in range(len(self.closed_forms)):
            closed = self.closed_forms[i]
            closed = {var: closed[var] for var in closed if var not in vars}
            self.closed_forms[i] = closed

    def pp_print(self):
        print('*'*100)
        for cond, closed_form in zip(self.conditions, self.closed_forms):
            for var in closed_form:
                print('{:<100}{}'.format('%s = %s' % (var, closed_form[var]), cond))
        print('*'*100)
                # print('%s = %s\t%s' % (var, closed_form[var], cond))

    def subs(self, subs_dict):
        new_conditions = [cond.subs(subs_dict, simultaneous=True) for cond in self.conditions]
        new_closed_forms = [{var: closed[var].subs(subs_dict, simultaneous=True) for var in closed} for closed in self.closed_forms]
        return Closed_form(new_conditions, new_closed_forms, self.ind_var, self.sum_end, self.bounded_vars)

    def add_constraint(self, constraint):
        self.conditions = [sp.simplify(sp.And(cond, constraint)) for cond in self.conditions]
        # self._simplify_conditions()

    def _simplify_conditions(self):
        sim = z3.Tactic('ctx-solver-simplify')
        for i in range(len(self.conditions)):
            # z3_cond = z3.And(z3.BoolVal(True), *(sim(to_z3(self.conditions[i]))[0]))
            deep_simplified_cond = z3_deep_simplify(to_z3(self.conditions[i]))
            z3_cond_list = list(sim(deep_simplified_cond)[0])
            remain_indicator = [True] * len(z3_cond_list)
            indicator = set()
            for j, cond in enumerate(z3_cond_list):
                s = z3.Solver()
                # s.add(*(z3_cond_list[:j] + z3_cond_list[j+1:]))
                s.add(*[c for k, c in enumerate(z3_cond_list) if k not in indicator.union({j})])
                s.add(z3.Not(cond))
                if s.check() == z3.unsat:
                    indicator.add(j)
                    remain_indicator.append(False)
                else:
                    remain_indicator.append(True)
            z3_cond = z3.And(z3.BoolVal(True), *[cond for i, cond in enumerate(z3_cond_list) if i not in indicator])
            new_cond = to_sympy(z3_cond)
            self.conditions[i] = new_cond
        new_closed_forms = []
        new_conditions = []
        for closed, cond in zip(self.closed_forms, self.conditions):
            if cond is not sp.S.false:
                new_closed_forms.append(closed)
                new_conditions.append(cond)
        self.closed_forms = new_closed_forms
        self.conditions = new_conditions

        merged = {}
        for i in range(len(self.conditions)):
            absorbed = reduce(set.union, (merged[idx] for idx in merged), set())
            if i in absorbed: continue
            merged[i] = set()
            # if i in merged: continue
            # cur_closed_form = self.closed_forms[i]
            # cur_condition = self.conditions[i]
            for j in range(len(self.conditions)):
                # print(merged)
                absorbed = reduce(set.union, (merged[idx] for idx in merged))
                # print(absorbed)
                if i == j or j in absorbed: continue
                s = z3.Solver()
                for var in self.closed_forms[i]:
                    s.add(to_z3(self.conditions[j]))
                    s.push()
                    s.add(z3.Not(to_z3(self.closed_forms[i][var]) == to_z3(self.closed_forms[j][var])))
                    # print(self.conditions[j])
                    # print(z3.Not(to_z3(self.closed_forms[i][var]) == to_z3(self.closed_forms[j][var])))
                    # print('*'*10)
                    check_res = s.check()
                    s.pop()
                    if check_res == z3.sat:
                        break
                else:
                    # new_closed_forms.append(self.closed_forms[i])
                    # new_conditions.append(z3.Or(self.conditions[i], self.conditions[j]))
                    # cur_condition = to_sympy(z3_deep_simplify(z3.And(z3.BoolVal(True), *sim(to_z3(sp.Or(cur_condition, self.conditions[j])))[0])))
                    # merged.add(j)
                    merged[i].add(j)
            # new_closed_forms.append(cur_closed_form)
            # new_conditions.append(cur_condition)

        new_closed_forms = []
        new_conditions = []
        absorbed = reduce(set.union, (merged[idx] for idx in merged))
        for i in merged:
            cur_condition = to_sympy(z3_deep_simplify(z3.And(z3.BoolVal(True), *sim(to_z3(sp.Or(self.conditions[i], *[self.conditions[j] for j in merged[i]])))[0])))
            if i not in absorbed:
                new_closed_forms.append(self.closed_forms[i])
                new_conditions.append(cur_condition)
            else:
                self.conditions[i] = cur_condition
        self.closed_forms = new_closed_forms
        self.conditions = new_conditions
        if len(absorbed) != 0:
            self._simplify_conditions()

    def simplify(self):
        self._simplify_conditions()
        for i, cond in enumerate(self.conditions):
            closed = self.closed_forms[i]
            new_closed = {var: my_sp_simplify(closed[var], cond) for var in closed}
            self.closed_forms[i] = new_closed

    def set_ind_var(self, ind_var):
        self.ind_var = ind_var

    def get_ind_var(self):
        return self.inv_var

    def to_sympy(self):
        res_tmp = {}
        for cond, closed in zip(self.conditions, self.closed_forms):
            cond_p = sp.S.true if sp.Or(cond, self.ind_var < 0) is sp.S.true else cond
            for var in closed:
                res_tmp.setdefault(var, []).append((closed[var], cond_p))
        res = {var: sp.Piecewise(*(res_tmp[var] + [(-1, True)])) for var in res_tmp}
        return res

    def eval(self, values):
        for cond, (closed_forms, ks) in zip(self.conditions, self.closed_forms_ks):
            if sp.simplify(cond.subs(values)) == sp.S.true:
                for closed_form, k in zip(closed_forms, ks + [sp.oo]):
                    if values[self.ind_var] < k.subs(values):
                        return {var: closed_form[var].subs(values) for var in closed_form}
        raise Exception('fail')

    def to_c(self, symbol_table):
        self.keep_live_vars(symbol_table.get_vars())
        self._reorder_conditions()
        # self.pp_print()
        if self.is_splitable() and self.bounded_vars is not None:
            res = self.to_c_split()
        elif self.bounded_vars is None:
            res = self.to_c_scalar()
        else:
            res = self.to_c_general(symbol_table)
        return res
    
    def to_c_scalar(self):
        block_items = []
        for cond in self.conditions:
            if cond is sp.S.true:
                for closed_form in self.closed_forms:
                    for var in closed_form:
                        if len(var.args) == 0:
                            lhs = expr2c(var)
                            rhs = expr2c(closed_form[var])
                            assignment = c_ast.Assignment('=', lhs, rhs)
                            block_items.append(assignment)
                # res = c_ast.Compound(block_items)
        return block_items

    def is_splitable(self):
        try:
            _ = [cond.as_set() for cond in self.conditions]
            return len(self.conditions) >= 1
        except:
            return False

    def _reorder_conditions(self):
        if self.is_splitable() and len(self.conditions) > 1:
            t = list(self.conditions[0].free_symbols)[0]
            closed_forms_conditions = sorted(zip(self.closed_forms, self.conditions), key=lambda x: x[1].as_set().sup)
            self.closed_forms = [cc[0] for cc in closed_forms_conditions]
            self.conditions = [cc[1].as_set().as_relational(t) for cc in closed_forms_conditions]

    def to_c_split(self):
        loops = []
        for closed, cond in zip(self.closed_forms, self.conditions):
            interval = cond.as_set()
            left = interval.inf
            right = interval.sup
            if not interval.contains(left):
                left = left + 1
            if interval.contains(right):
                right = right + 1

            for var in closed:
                if len(var.args) > 0:
                    # generator = c_generator.CGenerator()
                    assert(len(var.args) == 1)
                    t = var.args[0]
                    decl = c_ast.Decl(str(t), None, None, None, None, c_ast.TypeDecl(str(t), [], None, c_ast.IdentifierType(['int'])), c_ast.Constant('int', str(left)), None)
                    init = c_ast.DeclList([decl])
                    cond = c_ast.BinaryOp('<', c_ast.ID(str(t)), c_ast.Constant('int', str(right)))
                    nex = c_ast.UnaryOp('p++', c_ast.ID(str(t)))
                    assignment = c_ast.Assignment('=', c_ast.ArrayRef(c_ast.ID(str(var.func)), c_ast.ID(str(t))), expr2c(closed[var]))
                    stmt = c_ast.Compound([assignment])
                    for_loop = c_ast.For(init, cond, nex, stmt)
                    loops.append(for_loop)
        return loops

    def to_c_general(self, symbol_table):
        assert(len(self.conditions) == 1)
        closed, _ = self.closed_forms[0], self.conditions[0]
        # stmt_list = []
        loops = []
        for var in closed:
            # idx = [c_ast.ID(str(t)) for t in var.args]
            # array_ref = c_ast.ArrayRef(c_ast.ID(str(var.func)), idx[0])
            # for i in range(1, len(idx)):
            #     array_ref = c_ast.ArrayRef(array_ref, idx[i])
            stmt = c_ast.Assignment('=', expr2c(var), expr2c(closed[var]))
            # stmt_list.append(assignment)
            # stmt = c_ast.Compound(stmt_list)
            for t, bnd in reversed(list(zip(var.args, symbol_table.q_dim_bnd(str(var.func))))):
                stmt = c_ast.Compound([stmt])
                decl = c_ast.Decl(str(t), None, None, None, None, c_ast.TypeDecl(str(t), [], None, c_ast.IdentifierType(['int'])), c_ast.Constant('int', '0'), None)
                init = c_ast.DeclList([decl])
                cond = c_ast.BinaryOp('<', c_ast.ID(str(t)), c_ast.Constant('int', str(bnd)))
                nex = c_ast.UnaryOp('p++', c_ast.ID(str(t)))
                stmt = c_ast.For(init, cond, nex, stmt)
            loops.append(stmt)
        return loops





        # for var in closed:
        #     if len(var.args) > 0:
        #         for t in var.args:
        #             decl = c_ast.Decl(str(t), None, None, None, None, c_ast.TypeDecl(str(t), [], None, c_ast.IdentifierType(['int'])), c_ast.Constant('int', str(0)), None)
        #             init = c_ast.DeclList([decl])
        #             cond = c_ast.BinaryOp('<', c_ast.ID(str(t)), c_ast.Constant('int', str(dim_info[t])))
        #             nex = c_ast.UnaryOp('p++', c_ast.ID(str(t)))
        #             # assignment = c_ast.Assignment('=', c_ast.ArrayRef(c_ast.ID(str(var.func)), c_ast.ID(str(t))), expr2c(closed[var]))
        #             stmt = c_ast.Compound([assignment])
        #             for_loop = c_ast.For(init, cond, nex, stmt)




    def to_rec(self, scalar_closed_forms):
        assert(self.bounded_vars is not None)
        Rec = recurrence.Recurrence
        d = sp.Symbol('d_p', integer=True)
        scalar_closed_form = {var: scalar_closed_form[var].subs(Rec.neg_ind_var, d) for var in scalar_closed_form}
        transitions = []
        acc_transitions = []
        acc_symbol = sp.Symbol('_acc', integer=True)
        for cond, trans in zip(self.conditions, self.closed_forms):
            assert(len(trans) == 1)
            left_app = list(trans)[0]
            right_app = get_app_by_var(left_app.func, trans[left_app])
            # app_trans = {var: get_app_by_var(var.func, trans[var]) for var in trans}
            t_trans = {arg: arg_p for arg, arg_p in zip(left_app.args, right_app.args)}
            transitions.append(t_trans)
            acc = trans[left_app] - right_app
            acc_transitions.append(acc)
        inits = {Rec.neg_ind_var: 0, acc_symbol: 0}
        return Rec(inits, self.conditions, transitions, Rec.neg_ind_var, acc_transitions)