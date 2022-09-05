import sympy as sp

class Closed_form:
    def __init__(self, closed_forms, ind_var):
        self.conditions = []
        self.closed_forms_ks = []
        self.ind_var = ind_var
        for closed_form, cond, ks in closed_forms:
            self.conditions.append(cond)
            self.closed_forms_ks.append((closed_form, ks))

    def pp_print(self):
        for cond, (closed_forms, ks) in zip(self.conditions, self.closed_forms_ks):
            prev_k = 0
            for closed_form, k in zip(closed_forms, ks + [sp.oo]):
                for var in closed_form:
                    print('%s = %s\t%s' % (var, closed_form[var], sp.And(cond, prev_k <= self.ind_var, self.ind_var < k)))
                prev_k = k


    def eval(self, values):
        for cond, (closed_forms, ks) in zip(self.conditions, self.closed_forms_ks):
            if sp.simplify(cond.subs(values)) == sp.S.true:
                for closed_form, k in zip(closed_forms, ks + [sp.oo]):
                    if values[self.ind_var] < k.subs(values):
                        return {var: closed_form[var].subs(values) for var in closed_form}
        raise Exception('fail')
