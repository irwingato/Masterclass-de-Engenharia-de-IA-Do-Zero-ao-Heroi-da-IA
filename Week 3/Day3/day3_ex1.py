"""
PROGRAMA: Derivada Simples de Função Polinomial

OBJETIVO: Calcular a derivada de primeira ordem de uma função univariável
          f(x) = x³ − 5x + 7, usando a biblioteca SymPy (computação simbólica).
"""

# ─── BIBLIOTECA ────────────────────────────────────────────────────────────────
# SymPy permite cálculos matemáticos exatos (não aproximados numéricamente).
import sympy as sp


# ─── PASSO 1: DEFINIÇÃO DA VARIÁVEL SIMBÓLICA ──────────────────────────────────
# sp.Symbol('x') cria um símbolo x que SymPy entende como variável matemática.
# Ao contrário de uma variável Python comum, aqui x não tem valor fixo — ele
# representa "x" na expressão algébrica, permitindo derivar, integrar, etc.
x = sp.Symbol('x')

# ─── PASSO 2: DEFINIÇÃO DA FUNÇÃO ──────────────────────────────────────────────
# f(x) = x³ − 5x + 7
#
# MATEMÁTICA: é um polinômio de grau 3 (cúbico).
#   - x³ → cresce/decresce rapida;
#   - −5x → termo linear que inclina a curva;
#   - +7 → constante, desloca a curva para cima (não afeta a derivada).
#
# Comportamento da função:
#   - f'(x) = 3x² − 5
#   - Tem mínimo/máximo quando f'(x) = 0 → 3x² = 5 → x = ±√(5/3) ≈ ±1,29
f = x**3 - 5*x + 7


# ─── PASSO 3: CÁLCULO DA DERIVADA ─────────────────────────────────────────────
# sp.diff(f, x) calcula a derivada de primeira ordem de f em relação a x.
#
# MATEMÁTICA: cada termo é derivado separadamente (regra da soma):
#   d/dx(x³)   = 3x²   → expoente desce e diminui 1
#   d/dx(−5x)  = −5     → derivada de constante × x é a constante
#   d/dx(7)    = 0      → derivada de constante é sempre 0
#
# Resultado esperado: 3x² − 5
derivative = sp.diff(f, x)


# ─── SAÍDA ─────────────────────────────────────────────────────────────────────
# sp.diff retorna a expressão simbólica simplificada automaticamente (3x² − 5).
print("Function: ", f)
print("Derivative: ", derivative)