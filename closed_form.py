import sympy as sp
import z3
from utils import to_z3, to_sympy

class Closed_form:
    def __init__(self, conditions, closed_forms, ind_var):
        self.conditions = [sp.S.true] if len(conditions) == 1 else conditions
        self.closed_forms = closed_forms
        self.ind_var = ind_var
        self._simplify_conditions()

    def pp_print(self):
        for cond, closed_form in zip(self.conditions, self.closed_forms):
            for var in closed_form:
                print('{:<100}{}'.format('%s = %s' % (var, sp.refine(closed_form[var], cond)), cond))
                # print('%s = %s\t%s' % (var, closed_form[var], cond))

    def subs(self, subs_dict):
        new_conditions = [cond.subs(subs_dict, simultaneous=True) for cond in self.conditions]
        new_closed_forms = [{var: closed[var].subs(subs_dict, simultaneous=True) for var in closed} for closed in self.closed_forms]
        return Closed_form(new_conditions, new_closed_forms, self.ind_var)

    def add_constraint(self, constraint):
        self.conditions = [sp.simplify(sp.And(cond, constraint)) for cond in self.conditions]
        self._simplify_conditions()

    def _simplify_conditions(self):
        sim = z3.Tactic('ctx-solver-simplify')
        for i in range(len(self.conditions)):
            z3_cond = z3.And(z3.BoolVal(True), *(sim(to_z3(self.conditions[i]))[0]))
            new_cond = to_sympy(z3_cond)
            self.conditions[i] = new_cond
        new_closed_forms = []
        new_conditions = []
        for closed, cond in zip(self.closed_forms, self.conditions):
            if cond:
                new_closed_forms.append(sp.refine(closed, cond))
                new_conditions.append(cond)
        self.closed_forms = new_closed_forms
        self.conditions = new_conditions

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
        res = {var: sp.Piecewise(*res_tmp[var]) for var in res_tmp}
        return res

    def eval(self, values):
        for cond, (closed_forms, ks) in zip(self.conditions, self.closed_forms_ks):
            if sp.simplify(cond.subs(values)) == sp.S.true:
                for closed_form, k in zip(closed_forms, ks + [sp.oo]):
                    if values[self.ind_var] < k.subs(values):
                        return {var: closed_form[var].subs(values) for var in closed_form}
        raise Exception('fail')
