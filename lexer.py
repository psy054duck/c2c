import sympy as sp
import ply.lex as lex

keywords = (
    'if', 'else', 'True', 'False'
)

tokens = (
    'VARIABLE','NUMBER',
    'PLUS','MINUS','TIMES','DIV','EQ', 'MOD',
    'LPAREN','RPAREN', 'LBRACK', 'RBRACK',
    'SEMI', 'ASSIGN',
    'GT', 'LT', 'GE', 'LE',
    'AND', 'OR',
    ) + tuple(k.upper() for k in keywords)


t_PLUS = r'\+'
t_MINUS = r'\-'
t_TIMES = r'\*'
t_DIV = r'/'
t_MOD = r'%'
t_IF = r'if'
t_ELSE = r'else'
t_LBRACK = r'{'
t_RBRACK = r'}'
t_LPAREN = r'\('
t_RPAREN = r'\)'
# t_LREC = r'\['
# t_RREC = r'\]'
t_SEMI = r';'
t_ASSIGN = r'='
t_EQ = r'=='
t_LT = r'<'
t_GT = r'>'
t_LE = r'<='
t_GE = r'>='
t_AND = r'&&'
t_OR = r'\|\|'
t_TRUE = r'True'
t_FALSE = r'False'

t_ignore = ' \t'

def t_NUMBER(t):
    r'\-?\d+\.?'
    t.value = int(t.value[:-1] if t.value[-1] == '.' else t.value)
    return t

def t_VARIABLE(t):
    r'[a-zA-Z_]+'
    if t.value in keywords:
        t.type = t.value.upper()
    return t

def t_newline(t):
    r'\n'
    t.lexer.lineno += t.value.count('\n')

def t_error(t):
    print("Illegal character '%s'" % t.value[0])

lexer = lex.lex()