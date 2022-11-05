from tkinter import W


class SymbolTable:

    def __init__(self):
        self.tb = {}
        self._stack = []

    def push(self):
        self._stack.append(self.tb.copy())

    def pop(self):
        self.tb = self._stack.pop()

    def insert_record(self, var, **args):
        self.tb[var] = args

    def add_attribute(self, var, **args):
        self.tb[var] |= args

    def q_dim(self, var):
        try:
            res = self.tb[var]['dim']
        except:
            res = 0
        return res

    def q_type(self, var):
        return self.tb[var]['type']

    def q_dim_bnd(self, var):
        return self.tb[var]['bound']

    def q_value(self, var):
        try:
            return self.tb[var]['value']
        except:
            return None

    def get_vars(self):
        return set(self.tb)

    def print(self):
        for var in self.tb:
            print('%s: %s' % (var, self.tb[var]))
        print('*'*100)

if __name__ == '__main__':
    from sympy import *
    a = Symbol('a')
    aa = Function('aa')
    tb = SymbolTable()
    tb.insert_record(a, 'int')
    tb.print()
    tb.push()
    tb.insert_record(aa, 'int', dim=2, bnd=(100, 200))
    tb.print()
    tb.pop()
    tb.print()
    tb.insert_record(aa, 'double', dim=2, bnd=(100, 200))
    tb.print()
    

