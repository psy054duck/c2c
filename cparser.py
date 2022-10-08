from copy import deepcopy
from logging import exception
from pycparser import parse_file, c_generator
from pycparser.c_ast import *
import sympy as sp
from parser import parse
from utils import compute_N
from closed_form import Closed_form
from recurrence import Recurrence

class Vectorizer:
    def __init__(self):
        self.symbol_table = {}
        self.loop_stack = []

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
        new_blocks = []
        for i in range(len(node.block_items)):
            block_item = node.block_items[i]
            b = self.visit(block_item)
            if isinstance(block_item, Decl):
                self.symbol_table[b[0]] = b[1]
            elif isinstance(block_item, Assignment) and isinstance(b[1], int):
                self.symbol_table[str(b[0])] = {'init': b[1]}# b[1]
            else:
                res.append(b)
            
            if isinstance(block_item, For) and (len(self.loop_stack) == 0 or i == len(node.block_items) - 1 or b[1] is None):
                pass
                # new_blocks.extend(b[0])
            else:
                new_blocks.append(block_item)
        node.block_items = new_blocks
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
        self.loop_stack.append(1)
        init = self.visit(node.init)
        old_table = self.symbol_table.copy()
        # self.symbol_table = {}
        for var_info in init:
            self.symbol_table[var_info[0]] = var_info[1]
        cond = self.visit(node.cond)
        nex = self.visit(node.next)
        stmt = self.visit(node.stmt)
        self.loop_stack.pop()
        # try:
        filename = 'rec.txt'
        considered = set()
        rec = for2rec(init, nex, stmt, filename=filename)
        if rec is None:
            rec = parse(filename)
        scalar_cf, array_cf = rec.solve_array()
        scalar_cf = scalar_cf.subs({sp.Symbol(var, integer=True): self.symbol_table[var]['init'] for var in self.symbol_table if self.symbol_table[var]['init'] is not None})
        # scalar_cf = scalar_cf.subs({sp.Symbol(var, integer=True): self.symbol_table[var] for var in self.symbol_table if self.symbol_table[var] is not None})
        num_iter = compute_N(cond, scalar_cf)
        print('#'*10)
        rec.print()
        print(cond)
        print(self.symbol_table)
        scalar_cf.pp_print()
        print(num_iter)
        print('#'*10)
        scalar_cf = scalar_cf.subs({scalar_cf.ind_var: num_iter})
        scalar_cf.pp_print()
        array_cf = array_cf.subs({sp.Symbol(var, integer=True): self.symbol_table[var]['init'] for var in set(self.symbol_table) - set(old_table) if self.symbol_table[var]['init'] is not None})
        array_cf = array_cf.subs({array_cf.ind_var: num_iter, array_cf.sum_end: num_iter})
        print('*'*10)
        print(set(self.symbol_table))
        print(set(old_table))
        considered = set()
        if 'i' in set(self.symbol_table) - set(old_table):
            for closed_forms in array_cf.closed_forms:
                not_considered = set(closed_forms) - considered
                for var in not_considered:
                    considered.add(var)
                    for bnd_var, bnd in zip(array_cf.bounded_vars, self.symbol_table[str(var.func)]['type'][1]):
                        array_cf.add_constraint(bnd_var < bnd)
                        array_cf.add_constraint(bnd_var >= 0)
            array_cf.simplify()
        res = scalar_cf, array_cf
        array_cf.pp_print()
        self.symbol_table = old_table
        return res
    
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
            ret = (lvalue, rvalue)
        elif op == '+=':
            ret = (lvalue, lvalue + rvalue)
        elif op == '-=':
            ret = (lvalue, lvalue - rvalue)
        elif op == '*=':
            ret = (lvalue, lvalue * rvalue)
        elif op == '/=':
            ret = (lvalue, lvalue / rvalue) # type
        else:
            raise Exception('operation "%s" is no implemented' % op)
        return ret

    def visit_ArrayRef(self, node):
        name = sp.Function(str(self.visit(node.name)))
        subscript = self.visit(node.subscript)
        return name(subscript)

    def visit_Return(self, node):
        expr = self.visit(node.expr)

    def visit_ID(self, node):
        return sp.Symbol(node.name, integer=True)

    def visit_ParamList(self, node):
        res = []
        for p in node.params:
            res.append(self.visit(p))
        return res

    def visit_Typedef(self, node):
        return node

    def visit_FuncCall(self, node):
        return node

    def visit_NoneType(self, node):
        '''A special visitor for None'''
        return None

    def unimplemented_visit(self, node):
        raise Exception('visitor for "%s" is not implemented' % type(node))

def for2rec(init, nex, body, filename=None):
    if len(body) == 1 and all(isinstance(i, Closed_form) for i in body[0]):
        return flat_body_inner_arr(body[0], nex)
    else:
        conds, stmts = flat_body_compound(body + [nex])
    if filename is None:
        pass
    else:
        with open(filename, 'w') as fp:
            s = ''.join('%s = %s;\n' % (var, info['init']) for var, info in (init))
            s += 'if (%s) {\n' % conds[0]
            s += '%s\n' % _transform_stmt(stmts[0])
            s += '} '
            for cond, stmt in zip(conds[1:], stmts[1:]):
                s += 'else if (%s) {\n' % cond
                s += '%s\n' % _transform_stmt(stmt)
                s += '} '
            fp.write(s)

def _transform_stmt(stmts):
    return '\n'.join(['\t%s = %s;' % (var, expr) for var, expr in stmts])

def flat_body_inner_arr(body, nex):
    assert(all(isinstance(b, Closed_form) for b in body))
    scalar_cf, arr_cf = body
    # arr_cf = arr_cf.subs()

    conditions = []
    transitions = []
    variables = set()
    for scalar_cond, scalar_form in zip(scalar_cf.conditions, scalar_cf.closed_forms):
        for arr_cond, arr_form in zip(arr_cf.conditions, arr_cf.closed_forms):
            conditions.append(sp.And(scalar_cond, arr_cond))
            transitions.append(scalar_form | arr_form | {nex[0]: nex[1]})
            variables = variables.union(set(scalar_form))
    all_cond = sp.simplify(sp.Or(*conditions))
    if all_cond is not sp.logic.true:
        conditions.append(sp.Not(all_cond))
        transitions.append({var: var for var in variables} | {nex[0]: nex[1]})
    conditions = [cond.subs(Recurrence.neg_ind_var, sp.Symbol('_d0', integer=True)) for cond in conditions]
    transitions = [{var: t[var].xreplace({Recurrence.neg_ind_var: sp.Symbol('_d0', integer=True)}) for var in t} for t in transitions]
    print(transitions)
    print('h'*100)
    rec = Recurrence({}, conditions, transitions, bounded_vars=arr_cf.bounded_vars)
    return rec

def flat_body_compound(body):
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

        t_conds, t_recs = flat_body_compound(iftrue)
        f_conds, f_recs = flat_body_compound(iffalse)

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
    # c_ast = parse_file('test.c', use_cpp=True, cpp_path='clang-cpp-10', cpp_args='-I./fake_libc_include')
    # c_ast = parse_file('test.c', use_cpp=True)
    c_ast = parse_file('test.c', use_cpp=True, cpp_args='-I./fake_libc_include')
    vectorizer = Vectorizer()
    new_ast = vectorizer.visit(c_ast)
    generator = c_generator.CGenerator()
    generator.visit(new_ast)