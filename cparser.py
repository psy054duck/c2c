from copy import deepcopy
from logging import exception
from pycparser import parse_file, c_generator
from pycparser.c_ast import *
import sympy as sp

class Vectorizer:
    def __init__(self):
        self.symbol_table = {}

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        return getattr(self, method, self.unimplemented_visit)(node)

    def visit_FileAST(self, node):
        for ext in node.ext:
            res = self.visit(ext)
            if isinstance(ext, Decl):
                self.symbol_table[res[0]] = res[1]
        return node


    def visit_Decl(self, node):
        t = self.visit(node.type)
        init = self.visit(node.init)
        return node.name, {'type': t, 'init': init}
        # self.symbol_table[node.name] = {'type': t, 'init': init}

    def visit_ArrayDecl(self, node):
        t = self.visit(node.type)
        dim = self.visit(node.dim)
        return (t[0], [dim] + t[1])

    def visit_TypeDecl(self, node):
        return self.visit(node.type)

    def visit_IdentifierType(self, node):
        return (node.names, [])

    def visit_Constant(self, node):
        if node.type == 'float' or node.type == 'double':
            return float(node.value)
        elif node.type == 'int':
            return int(node.value)
        else:
            raise exception('Type "%s" is not implemented' % node.type)

    def visit_FuncDecl(self, node):
        args = self.visit(node.args)
        t = self.visit(node.type)
        return t, args

    def visit_FuncDef(self, node):
        old_symbol_table = deepcopy(self.symbol_table)
        func_name = node.decl.name
        t, args = self.visit_FuncDecl(node.decl.type)
        self.visit(node.body)
        self.symbol_table = old_symbol_table

    def visit_Compound(self, node):
        res = []
        for block_item in node.block_items:
            b = self.visit(block_item)
            res.append(b)
        return res

    def visit_BinaryOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if node.op == '+':
            return left + right
        elif node.op == '-':
            return left - right
        elif node.op == '*':
            return left * right
        elif node.op == '/' and any(isinstance(operand, float) for operand in [left, right]):
            return left / right
        elif node.op == '/' and all(isinstance(operand, int) for operand in [left, right]):
            return left // right
        elif node.op == '/':
            return left / right
        elif node.op == '<':
            return left < right
        else:
            raise Exception('Operation "%s" is not implemented' % node.op)


    def visit_UnaryOp(self, node):
        expr = self.visit(node.expr)
        if node.op == 'p++' or node.op == '++':
            return (expr, expr + 1)
        else:
            raise Exception('Operation "%s" is not implemented' % node.op)

    def visit_DeclList(self, node):
        res = []
        for decl in node.decls:
            res.append(self.visit(decl))
        return res

    def visit_For(self, node):
        init = self.visit(node.init)
        cond = self.visit(node.cond)
        nex = self.visit(node.next)
        stmt = self.visit(node.stmt)
        for2rec(init, nex, stmt)
    
    def visit_If(self, node):
        cond = self.visit(node.cond)
        iftrue = self.visit(node.iftrue)
        iffalse = self.visit(node.iffalse)
        return (cond, iftrue, iffalse)

    def visit_Assignment(self, node):
        op = node.op
        lvalue = self.visit(node.lvalue)
        rvalue = self.visit(node.rvalue)
        if op == '=':
            return (lvalue, rvalue)
        elif op == '+=':
            return (lvalue, lvalue + rvalue)
        elif op == '-=':
            return (lvalue, lvalue - rvalue)
        elif op == '*=':
            return (lvalue, lvalue * rvalue)
        elif op == '/=':
            return (lvalue, lvalue / rvalue) # type
        else:
            raise Exception('operation "%s" is no implemented' % op)

    def visit_ArrayRef(self, node):
        name = sp.Function(str(self.visit(node.name)))
        subscript = self.visit(node.subscript)
        return name(subscript)

    def visit_Return(self, node):
        expr = self.visit(node.expr)

    def visit_ID(self, node):
        return sp.Symbol(node.name)

    def visit_ParamList(self, node):
        res = []
        for p in node.params:
            res.append(self.visit(p))
        return res

    def visit_NoneType(self, node):
        '''A special visitor for None'''
        return None

    def unimplemented_visit(self, node):
        raise Exception('visitor for "%s" is not implemented' % type(node))

def for2rec(init, nex, body):
    print(flat_body(body + [nex]))

def flat_body(body):
    res_cond = [True]
    res_stmt = [[]]
    is_if = lambda x: len(x) > 0 and (x[0].is_Boolean or x[0].is_Relational)
    if len(body) == 1:
        stmt = body[0]
        if is_if(stmt):
            return [stmt[0], sp.Not(stmt[0])], [stmt[1], stmt[2]]
        return [True], [[stmt]]
    elif len(body) == 0:
        return [], []

    for item in body:
        if is_if(item):
            cond, iftrue, iffalse = item
        else:
            cond = True
            iftrue = [item]
            iffalse = []

        t_conds, t_recs = flat_body(iftrue)
        f_conds, f_recs = flat_body(iffalse)

        t_conds = [sp.And(cond, c) for c in t_conds]
        f_conds = [sp.And(sp.Not(cond), c) for c in f_conds]
        stmt = t_recs + f_recs
        cur_cond = []
        cur_stmt = []
        for i, cond1 in enumerate(res_cond):
            for j, cond2 in enumerate(t_conds + f_conds):
                subs_pairs = res_stmt[i]
                cur_cond.append(sp.And(cond1, cond2.subs(subs_pairs)))
                cur_stmt.append(res_stmt[i] + [(var if isinstance(var, sp.Symbol) else var.subs(subs_pairs), s.subs(subs_pairs)) for var, s in stmt[j]])
        res_cond = cur_cond
        res_stmt = cur_stmt
    return res_cond, res_stmt

if __name__ == '__main__':
    c_ast = parse_file('test.c', use_cpp=True, cpp_path='clang-cpp-10')
    vectorizer = Vectorizer()
    new_ast = vectorizer.visit(c_ast)
    generator = c_generator.CGenerator()
    generator.visit(new_ast)