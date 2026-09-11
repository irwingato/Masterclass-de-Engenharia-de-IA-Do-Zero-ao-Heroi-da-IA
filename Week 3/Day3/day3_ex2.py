"""
PROGRAMA: Derivadas Parciais de uma Função de Duas Variáveis

OBJETIVO: Calcular as derivadas parciais de primeira ordem da função
          f(x, y) = x² + 3y² − 4xy, usando SymPy.
"""

# ─── BIBLIOTECA ────────────────────────────────────────────────────────────────
import sympy as sp


# ─── PASSO 1: DEFINIÇÃO DAS VARIÁVEIS SIMBÓLICAS ──────────────────────────────
# sp.symbols('x y') cria dois símbolos simultaneamente — x e y.
x, y = sp.symbols('x y')

# ─── PASSO 2: DEFINIÇÃO DA FUNÇÃO ──────────────────────────────────────────────
# f(x, y) = x² + 3y² − 4xy
#
# MATEMÁTICA: esta é uma função quadrática bivariável.
#   - x²       → parábola na direção de x (estufa para cima em x)
#   - 3y²      → parábola estreita na direção de y (estufa 3× mais rápido)
#   - −4xy     → termo CRUZADO (mistura as duas variáveis).
#                  Cria uma "sela" (forma de cadeira) em vez de um fundo:
#                  o mínimo em x pode ser máximo em y ao mesmo tempo.
#
# Ponto crítico (gradiente = 0):
#   ∂f/∂x = 2x − 4y = 0  →  x = 2y
#   ∂f/∂y = 6y − 4x = 0  →  6y = 4x → y = (2/3)x
#   Substituindo: x = 2·(2/3)x → x = (4/3)x → só verdadeira se x = 0, y = 0.
#   Hessiana: det(H) = (2)(6) − (−4)² = 12 − 16 = −4 < 0 → PONTO DE SELA em (0,0)
f = x**2 + 3*y**2 - 4*x*y


# ─── PASSO 3: DERIVADAS PARCIAIS ───────────────────────────────────────────────
# sp.diff(f, x) → derivada parcial em relação a x (trata y como constante):
#   ∂f/∂x = 2x − 4y
grad_x = sp.diff(f, x)

# sp.diff(f, y) → derivada parcial em relação a y (trata x como constante):
#   ∂f/∂y = 6y − 4x
grad_y = sp.diff(f, y)


# ─── SAÍDA ─────────────────────────────────────────────────────────────────────
print("Gradients:")
print("Grad X:", grad_x)
print("Grad Y:", grad_y)
