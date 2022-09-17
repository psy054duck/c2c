import sympy as sp

class Closed_form:
    def __init__(self, conditions, closed_forms, ind_var):
        self.conditions = [sp.S.true] if len(conditions) == 1 else conditions
        self.closed_forms = closed_forms
        self.ind_var = ind_var

    def pp_print(self):
        for cond, closed_form in zip(self.conditions, self.closed_forms):
            for var in closed_form:
                print('{:<100}{}'.format('%s = %s' % (var, closed_form[var]), cond))
                # print('%s = %s\t%s' % (var, closed_form[var], cond))

    def subs(self, subs_dict):
        new_conditions = [cond.subs(subs_dict, simultaneous=True) for cond in self.conditions]
        new_closed_forms = [{var: closed[var].subs(subs_dict, simultaneous=True) for var in closed} for closed in self.closed_forms]
        return Closed_form(new_conditions, new_closed_forms, self.ind_var)

    def set_ind_var(self, ind_var):
        self.ind_var = ind_var

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
