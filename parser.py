import sympy as sp
import ply.yacc as yacc
from lexer import lexer, tokens
from recurrence import Recurrence

def p_recurrence(p):
    '''recurrence : if_seq'''
    conditions = [s[0] for s in p[1]]
    transitions = [s[1] for s in p[1]]
    p[0] = Recurrence(conditions, transitions)

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
    '''cond : atom_cond'''
    p[0] = p[1]

def p_atom_cond_1(p):
    '''atom_cond : expr EQ expr'''
    p[0] = p[1] == p[3]

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

def p_transitions_1(p):
    '''transitions : transition transitions'''
    p[0] = p[1] | p[2]

def p_transitions_2(p):
    '''transitions : '''
    p[0] = {}

def p_transition(p):
    '''transition : VARIABLE ASSIGN expr SEMI'''
    p[0] = {sp.Symbol(p[1]): p[3]}

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
    '''factor : operand'''
    p[0] = p[1]

def p_operand_1(p):
    '''operand : VARIABLE'''
    p[0] = sp.Symbol(p[1])

def p_operand_2(p):
    '''operand : NUMBER'''
    p[0] = sp.Integer(p[1])

def p_error(p):
    print(p)

parser = yacc.yacc()

if __name__ == '__main__':
    with open('test.txt') as fp:
        recurrence = parser.parse(fp.read())
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        # recurrence.solve_periodic([0, 1])
        res = recurrence.solve_with_inits({x: sp.Integer(-10), y: sp.Integer(0)})
        print(res)