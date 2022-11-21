from array import array
import sympy as sp
import ply.yacc as yacc
from lexer import lexer, tokens
from recurrence import Recurrence


def p_recurrence(p):
    '''recurrence : inits if_seq'''
    inits = p[1]
    conditions = [s[0] for s in p[2]]
    transitions = [s[1] for s in p[2]]
    p[0] = Recurrence(inits, conditions, transitions)


def p_inits(p):
    '''inits : transitions'''
    p[0] = p[1]


def p_if_seq_1(p):
    '''if_seq : if_statement'''
    p[0] = [p[1]]


def p_if_seq_2(p):
    '''if_seq : if_statement ELSE if_seq'''
    p[0] = [p[1]] + p[3]


def p_if_seq_3(p):
    '''if_seq : if_statement ELSE LBRACK transitions RBRACK'''
    p[0] = [p[1], (sp.S.true, p[4])]


def p_if_statement(p):
    '''if_statement : IF LPAREN cond RPAREN LBRACK transitions RBRACK'''
    p[0] = (p[3], p[6])


def p_cond_1(p):
    '''cond : condo'''
    p[0] = p[1]


def p_condo_1(p):
    '''condo : conda OR condo'''
    p[0] = sp.Or(p[1], p[3])


def p_condo_2(p):
    '''condo : conda'''
    p[0] = p[1]


def p_conda_1(p):
    '''conda : atom_cond AND conda'''
    p[0] = sp.And(p[1], p[3])


def p_conda_2(p):
    '''conda : atom_cond'''
    p[0] = p[1]


def p_atom_cond_1(p):
    '''atom_cond : expr EQ expr'''
    p[0] = sp.Eq(p[1], p[3])


def p_atom_cond_2(p):
    '''atom_cond : expr GT expr'''
    p[0] = p[1] > p[3]


def p_atom_cond_3(p):
    '''atom_cond : expr LT expr'''
    p[0] = p[1] < p[3]


def p_atom_cond_4(p):
    '''atom_cond : expr GE expr'''
    p[0] = p[1] >= p[3]


def p_atom_cond_5(p):
    '''atom_cond : expr LE expr'''
    p[0] = p[1] <= p[3]


def p_atom_cond_6(p):
    '''atom_cond : TRUE'''
    p[0] = sp.S.true


def p_atom_cond_7(p):
    '''atom_cond : FALSE'''
    p[0] = sp.S.false


def p_transitions_1(p):
    '''transitions : transition transitions'''
    p[0] = p[1] | p[2]


def p_transitions_2(p):
    '''transitions : '''
    p[0] = {}


def p_transition_1(p):
    '''transition : VARIABLE ASSIGN expr SEMI'''
    p[0] = {sp.Symbol(p[1], integer=True): p[3]}


def p_transition_2(p):
    '''transition : array_ref ASSIGN expr SEMI'''
    p[0] = {p[1]: p[3]}


def p_array_ref(p):
    '''array_ref : VARIABLE array_index'''
    f = sp.Function(p[1])
    p[0] = f(*p[2])


def p_array_index_1(p):
    '''array_index : LPAREN expr RPAREN array_index'''
    p[0] = [p[2]] + p[4]


def p_array_index_2(p):
    '''array_index : LPAREN expr RPAREN'''
    p[0] = [p[2]]


def p_expr_1(p):
    '''expr : factor PLUS expr'''
    p[0] = p[1] + p[3]


def p_expr_2(p):
    '''expr : factor MINUS expr'''
    p[0] = p[1] - p[3]


def p_expr_3(p):
    '''expr : factor'''
    p[0] = p[1]


def p_factor_1(p):
    '''factor : factor TIMES operand'''
    p[0] = p[1] * p[3]


def p_factor_2(p):
    '''factor : factor DIV operand'''
    p[0] = p[1] / p[3]


def p_factor_3(p):
    '''factor : factor MOD operand'''
    p[0] = p[1] % p[3]


def p_factor_4(p):
    '''factor : operand'''
    p[0] = p[1]


def p_operand_1(p):
    '''operand : VARIABLE'''
    p[0] = sp.Symbol(p[1], integer=True)


def p_operand_2(p):
    '''operand : NUMBER'''
    p[0] = sp.Integer(p[1])


def p_operand_3(p):
    '''operand : array_ref'''
    p[0] = p[1]


def p_operand_4(p):
    '''operand : LPAREN expr RPAREN'''
    p[0] = p[2]


def p_error(p):
    print(p)


parser = yacc.yacc()


def parse(filename):
    with open(filename) as fp:
        recurrence = parser.parse(fp.read())
        return recurrence


if __name__ == '__main__':
    with open('rec.txt') as fp:
        recurrence = parser.parse(fp.read())
        x = sp.Symbol('x', integer=True)
        y = sp.Symbol('y', integer=True)
        # res = recurrence.solve_array()
        # mid = sp.Symbol('mid', integer=True)
        # res = res.subs({res.ind_var: 6400, mid: 3200})
        # res.pp_print()
        # recurrence.solve_periodic([0, 1])
        # res = recurrence.solve_with_inits({x: sp.Integer(-200), y: sp.Integer(0)})
        scalar_closed_form, array_closed_form = recurrence.solve_array()
        scalar_closed_form.pp_print()
        # for t in array_closed_form.bounded_vars:
        #     array_closed_form.add_constraint(t < 6400)
        # array_closed_form = array_closed_form.subs({array_closed_form.ind_var: 6400})
        array_closed_form._simplify_conditions()
        # array_closed_form.to_c()
        array_closed_form.pp_print()
        # res.pp_print()
        # values = {x: -10, y: 10, Recurrence.inductive_var: 11}
        # print(res.eval(values))
