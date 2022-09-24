from ast import Constant
from lib2to3.pgen2.pgen import generate_grammar
import sympy as sp
from sympy.logic.boolalg import true, false
import z3
from functools import reduce
import pycparser.c_ast as ast

def z3_deep_simplify(expr):
    # print(expr)
    sim = z3.Tactic('ctx-solver-simplify')
    cond_list = list(sim(expr)[0])
    # print(cond_list)
    if len(cond_list) == 0:
        return z3.BoolVal(True)
    elif len(cond_list) == 1:
        return cond_list[0]
    else:
        return z3.And(*[z3_deep_simplify(cond) for cond in cond_list])

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
        if len(self.args) == 1:
            res = to_z3(self.args[0][0])
        else:
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
    elif isinstance(self, sp.Ne):
        res = to_z3(self.lhs) != to_z3(self.rhs)
    elif isinstance(self, sp.Integer) or isinstance(self, int):
        res = z3.IntVal(int(self))
    elif isinstance(self, sp.Symbol):
        res = z3.Int(str(self))
    elif isinstance(self, sp.Rational):
        # return z3.RatVal(self.numerator, self.denominator)
        res = z3.IntVal(self.numerator) / z3.IntVal(self.denominator)
    elif isinstance(self, sp.Pow):
        if self.base == 0: res = z3.IntVal(0)
        else: raise Exception('%s' % self)
    elif isinstance(self, sp.Mod):
        res = to_z3(self.args[0]) % to_z3(self.args[1])
    elif isinstance(self, sp.Abs):
        res = z3.Abs(to_z3(self.args[0]))
    elif self is true:
        res = z3.BoolVal(True)
    elif self is false:
        res = z3.BoolVal(False)
    elif self.is_Function:
        func = self.func
        args = self.args
        z3_func = z3.Function(func.name, *([z3.IntSort()]*(len(args) + 1)))
        res = z3_func(*[to_z3(arg) for arg in args])
    else:
        raise Exception('Conversion for "%s" has not been implemented yet: %s' % (type(self), self))
    return z3.simplify(res)

def to_sympy(expr):
    if z3.is_int_value(expr):
        res = expr.as_long()
    elif z3.is_const(expr) and z3.is_bool(expr):
        res = sp.S.true if z3.is_true(expr) else sp.S.false
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
    elif z3.is_mod(expr):
        children = expr.children()
        res = to_sympy(children[0]) % to_sympy(children[1])
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
    elif z3.is_eq(expr):
        children = expr.children()
        res = sp.Eq(to_sympy(children[0]), to_sympy(children[1]))
    elif z3.is_not(expr):
        children = expr.children()
        res = sp.Not(to_sympy(children[0]))
    elif z3.is_and(expr):
        children = expr.children()
        res = sp.And(*[to_sympy(ch) for ch in children])
    elif z3.is_or(expr):
        children = expr.children()
        res = sp.Or(*[to_sympy(ch) for ch in children])
    elif len(expr.children()) == 3 and z3.is_bool(expr.children()[0]):
        children = expr.children()
        cond = to_sympy(children[0])
        res = sp.Piecewise((to_sympy(children[1]), cond), (to_sympy(children[2]), sp.S.true))
    else:
        raise Exception('conversion for type "%s" is not implemented: %s' % (type(expr), expr))
    return sp.simplify(res)

def get_app_by_var(var, expr):
    '''for sympy'''
    if expr.func is var:
        return expr
    for arg in expr.args:
        res = get_app_by_var(var, arg)
        if res is not None:
            return res
    return None

def expr2c(expr: sp.Expr):
    if isinstance(expr, sp.Add):
        assert(len(expr.args) == 2)
        res = ast.BinaryOp('+', expr2c(expr.args[0]), expr2c(expr.args[1]))
    elif isinstance(expr, sp.Mul):
        assert(len(expr.args) == 2)
        res = ast.BinaryOp('*', expr2c(expr.args[0]), expr2c(expr.args[1]))
    elif isinstance(expr, sp.Integer) or isinstance(expr, int):
        res = ast.Constant('int', str(expr))
    elif expr.is_Function:
        assert(len(expr.args) == 1)
        arg = expr.args[0]
        res = ast.ArrayRef(ast.ID(str(expr.func)), expr2c(arg))
    elif isinstance(expr, sp.Symbol):
        res = ast.ID(str(expr))
    else:
        raise Exception('conversion for type "%s" is not implemented: %s' % (type(expr), expr))
    return res



if __name__ == '__main__':
    a = sp.Function('a')
    i = sp.Symbol('i', integer=True)
    e = 2*a(i+1) + 1
    from pycparser import c_generator
    generator = c_generator.CGenerator()
    print(generator.visit(expr2c(e)))

    # print(get_app_by_var(a, e))
