"""
PROGRAMA: Cálculo de Derivadas de Segunda Ordem e Matriz Hessiana
OBJETIVO: Calcular as derivadas parciais de segunda ordem de uma função
          multivariável f(x, y) e montar a matriz Hessiana.

CONCEITO MATEMÁTICO:
  - Derivada parcial de segunda ordem: mede a taxa de variação da taxa de
    variação de uma função em relação a uma variável.
  - Matriz Hessiana: matriz quadrada de todas as derivadas parciais de segunda
    ordem de uma função. É fundamental em otimização (método de Newton, teste
    da segunda derivada para máximos/mínimos).
    Para f(x,y): H = | f_xx  f_xy |
                     | f_yx  f_yy |
    Onde f_xy = f_yx (pela simetria das derivadas mistas, teorema de Schwarz).
"""

# ─── BIBLIOTECA ────────────────────────────────────────────────────────────────
# SymPy é uma biblioteca de matemática simbólica em Python.
# Ela permite manipular expressões matemáticas de forma analítica (não numérica),
# ou seja, ela calcula derivadas exatamente, sem aproximações por diferenças finitas.
import sympy as sp


# ─── PASSO 1: DEFINIÇÃO DAS VARIÁVEIS SIMBÓLICAS ──────────────────────────────
# sp.symbols cria variáveis que o SymPy trata como símbolos matemáticos,
# não como valores numéricos. Isso permite derivar, integrar e simplificar
# expressões algebraicas de forma exata.
x, y = sp.symbols('x y')


# ─── PASSO 2: DEFINIÇÃO DA FUNÇÃO f(x, y) ─────────────────────────────────────
# A função escolhida é: f(x, y) = x³·y + sin(x)·e^y
#
# COMO A FUNÇÃO FUNCIONA:
#   - Termo 1: x³·y → produto de x³ com y. A derivada em relação a x traz 3x²·y;
#     a derivada em relação a y traz x³.
#   - Termo 2: sin(x)·e^y → produto de seno de x com o exponencial de y.
#     O seno oscila (varia entre -1 e 1), e o exponencial cresce/decresce
#     rapidamente. A derivada em relação a x traz cos(x)·e^y;
#     a derivada em relação a y traz sin(x)·e^y.
f = x**3 * y + sp.sin(x) * sp.exp(y)


print("--- DERIVADAS DE SEGUNDA ORDEM INDIVIDUAIS ---")

# ─── DERIVADA PARCIAL DE SEGUNDA ORDEM EM RELAÇÃO A x ─────────────────────────
# Notação: ∂²f / ∂x²
# Primeiro derivamos f em relação a x (duas vezes).
# sp.diff(f, x, 2) significa: derivar f em relação a x, de ordem 2.
#
# PROCESSO MATEMÁTICO:
#   f_x = ∂f/∂x = 3x²·y + cos(x)·e^y
#   f_xx = ∂²f/∂x² = 6x - sin(x)·e^y  (derivar novamente em relação a x)
f_xx = sp.diff(f, x, 2)
print(f"f_xx: {f_xx}")

# ─── DERIVADA PARCIAL DE SEGUNDA ORDEM EM RELAÇÃO A y ─────────────────────────
# Notação: ∂²f / ∂y²
# Primeiro derivamos f em relação a y (duas vezes).
#
# PROCESSO MATEMÁTICO:
#   f_y = ∂f/∂y = x³ + sin(x)·e^y
#   f_yy = ∂²f/∂y² = 0 + sin(x)·e^y = sin(x)·e^y
f_yy = sp.diff(f, y, 2)
print(f"f_yy: {f_yy}")

# ─── DERIVADA PARCIAL MISTA ────────────────────────────────────────────────────
# Notação: ∂²f / (∂x ∂y)
# Primeiro derivamos em relação a x, depois o resultado em relação a y.
# sp.diff(f, x, y) significa: derivar f em relação a x primeiro, depois em
# relação a y (ou vice-versa, pelo Teorema de Schwarz o resultado é o mesmo).
#
# PROCESSO MATEMÁTICO:
#   f_x = 3x²·y + cos(x)·e^y
#   f_xy = 3x² + cos(x)·e^y  (derivando f_x em relação a y)
f_xy = sp.diff(f, x, y)
print(f"f_xy: {f_xy}")


print("--- MATRIZ HESSIANA COMPLETA ---")

# ─── MATRIZ HESSIANA ───────────────────────────────────────────────────────────
# A Matriz Hessiana é uma matriz quadrada que agrupa todas as derivadas
# parciais de segunda ordem:
#
#         | ∂²f/∂x²   ∂²f/∂x∂y |
#   H  =  |                      |
#         | ∂²f/∂y∂x  ∂²f/∂y²   |
#
# sp.hessian(f, (x, y)) monta essa matriz automaticamente:
#   - A variável (x, y) define a ordem das linhas e colunas da matriz.
#   - hessian[0,0] = f_xx, hessian[0,1] = f_xy
#   - hessian[1,0] = f_yx (= f_xy por simetria), hessian[1,1] = f_yy
#
# UTILIDADE NA OTIMIZAÇÃO:
#   - Se o determinante da Hessiana > 0 e f_xx > 0 → ponto é MÍNIMO local.
#   - Se o determinante da Hessiana > 0 e f_xx < 0 → ponto é MÁXIMO local.
#   - Se o determinante da Hessiana < 0 → ponto de CADEIRA (sela).
#   - Se o determinante da Hessiana = 0 → teste é inconclusivo.
hessian = sp.hessian(f, (x, y))

# sp.pprint imprime a matriz de forma formatada (bonita no terminal),
# com alinhamento de colunas e símbolos matemáticos legíveis.
sp.pprint(hessian)
