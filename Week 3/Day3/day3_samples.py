"""
PROGRAMA: Exemplos de Derivadas Simples com SymPy

OBJETIVO: Dois exemplos didáticos de cálculo de derivadas usando computação
          simbólica — um de função de uma variável e outro de duas variáveis.
"""

# ─── BIBLIOTECA ────────────────────────────────────────────────────────────────
import sympy as sp


# ══════════════════════════════════════════════════════════════════════════════
# EXEMPLO 1 (comentado, inativo): Derivada de uma função de uma variável
# ══════════════════════════════════════════════════════════════════════════════
# x = sp.Symbol('x')          # Cria o símbolo x para operações matemáticas
# f = x**2                     # Define f(x) = x²
# derivative = sp.diff(f, x)  # Calcula df/dx = 2x
#
# MATEMÁTICA:
#   d/dx(x²) = 2x  (regra da potência: expoente desce e diminui 1)
#   Neste caso simples, o resultado é uma reta: para cada x, a inclinação
#   de f(x) = x² é 2x. Em x = 0 → inclinação 0 (fundo da parábola).
#   Em x = 3 → inclinação 6 (curva subindo rápido).
#
# print("Derivative: ", derivative)


# ══════════════════════════════════════════════════════════════════════════════
# EXEMPLO 2 (ativo): Derivadas parciais de função de duas variáveis
# ══════════════════════════════════════════════════════════════════════════════

# ─── VARIÁVEIS SIMBÓLICAS ─────────────────────────────────────────────────────
x, y = sp.symbols('x y')

# ─── FUNÇÃO ────────────────────────────────────────────────────────────────────
# f(x, y) = x² + y²
#
# MATEMÁTICA: é um PARABOLOIDE de revolução (tigela simétrica).
# As curvas de nível são círculos perfeitos ao redor do mínimo em (0, 0).
# É convexa → tem UM único mínimo global, sem mínimos locais.
f = x**2 + y**2

# ─── DERIVADAS PARCIAIS ───────────────────────────────────────────────────────
# ∂f/∂x: trata y como CONSTANTE e deriva em relação a x.
#   ∂f/∂x = 2x  (o termo y² é constante → some 0)
grad_x = sp.diff(f, x)

# ∂f/∂y: trata x como CONSTANTE e deriva em relação a y.
#   ∂f/∂y = 2y  (o termo x² é constante → some 0)
grad_y = sp.diff(f, y)

# ─── SAÍDA ─────────────────────────────────────────────────────────────────────
# Imprime as duas derivadas parciais: 2x e 2y
# O gradiente ∇f = (2x, 2y) aponta da origem para fora (direção de maior subida).
# Para minimizar, andamos em sentido oposto: (x, y) ← (x, y) − lr·(2x, 2y)
print("Partial Derivatives:", grad_x, grad_y)
