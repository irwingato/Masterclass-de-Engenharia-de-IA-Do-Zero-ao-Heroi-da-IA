import numpy as np

# =============================================================================
# OPERAÇÕES BÁSICAS COM MATRIZES
# =============================================================================
#
# MATEMÁTICA:
# As operações elementares com matrizes são a base de tudo em álgebra linear.
# Para duas matrizes A e B do MESMO tamanho, podemos somar e subtrair.
# Para multiplicação por escalar, multiplicamos CADA elemento por um número.
#
# Seja a matriz:
#     | a  b |
#     | c  d |
# =============================================================================

# Create matrices (Cria duas matrizes 2x2)
# np.array cria arrays NumPy bidimensionais (matrizes)
A = np.array([[1, 2], [3, 4]])  # Matriz A = [[1,2],[3,4]]
B = np.array([[9, 8], [7, 6]])  # Matriz B = [[9,8],[7,6]]

# --- ADIÇÃO DE MATRIZES ---
# MATEMÁTICA: Soma-se elemento por elemento (posição a posição)
#     | a1  a2 |   | b1  b2 |   | a1+b1  a2+b2 |
#     | a3  a4 | + | b3  b4 | = | a3+b3  a4+b4 |
#
# Neste caso:
#     | 1  2 |   | 9  8 |   | 1+9   2+8  |   | 10  10 |
#     | 3  4 | + | 7  6 | = | 3+7   4+6  | = | 10  10 |
#
# Regra: As matrizes devem ter as MESMAS dimensões para serem somadas
print("Addition: \n", A + B)

# --- SUBTRAÇÃO DE MATRIZES ---
# MATEMÁTICA: Mesmo princípio da adição, mas com subtração
#     | b1  b2 |   | a1  a2 |   | b1-a1  b2-a2 |
#     | b3  b4 | - | a3  a4 | = | b3-a3  b4-a4 |
#
# Neste caso (B - A):
#     | 9  8 |   | 1  2 |   | 9-1   8-2  |   | 8   6 |
#     | 7  6 | - | 3  4 | = | 7-3   6-4  | = | 4   2 |
print("Subtraction: \n", B - A)

# --- MULTIPLICAÇÃO POR ESCALAR ---
# MATEMÁTICA: Multiplica CADA elemento da matriz por um número (escalar)
#     | a  b |       | k*a  k*b |
# k × | c  d |   =   | k*c  k*d |
#
# Neste caso (3 × A):
#     | 1  2 |       | 3×1   3×2 |   | 3   6 |
# 3 × | 3  4 |   =   | 3×3   3×4 | = | 9  12 |
#
# O escalar pode ser qualquer número: inteiro, float, positivo ou negativo
print("Scalar Multiplication \n", 3 * A)