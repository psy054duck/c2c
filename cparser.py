from logging import exception
from pycparser import parse_file
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
        func_name = node.decl.name
        t, args = self.visit_FuncDecl(node.decl.type)
        self.visit(node.body)

    def visit_Compound(self, node):
        for block in node.block_items:
            res = self.visit(block)

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


if __name__ == '__main__':
    c_ast = parse_file('test.c', use_cpp=True)
    vectorizer = Vectorizer()
    vectorizer.visit(c_ast)