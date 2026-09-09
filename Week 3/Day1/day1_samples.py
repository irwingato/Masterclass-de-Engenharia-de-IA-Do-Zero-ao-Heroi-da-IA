import numpy as np

# =============================================================================
# AMOSTRAS DE OPERAÇÕES COM MATRIZES (CÓDIGO DE REFERÊNCIA)
# =============================================================================
# Este arquivo contém exemplos de operações matriciais.
# A maioria está comentada para servir como consulta rápida.
# Apenas a matriz diagonal é impressa por padrão.
# =============================================================================

# Cria duas matrizes 2×2 para exemplo
A = np.array([[1, 2], [3, 4]])  # A = [[1,2],[3,4]]
B = np.array([[5, 6], [7, 8]])  # B = [[5,6],[7,8]]

# --- ADIÇÃO: Soma elemento a elemento ---
# | 1+5  2+6 |   | 6   8 |
# | 3+7  4+8 | = | 10  12 |
# print("Addition: \n", A + B)

# --- SUBTRAÇÃO: Subtrai elemento a elemento ---
# | 5-1  6-2 |   | 4  4 |
# | 7-3  8-4 | = | 4  4 |
# print("Subtraction: \n", B - A)

# --- MULTIPLICAÇÃO POR ESCALAR ---
# C = 2 × A = [[2×1, 2×2],[2×3, 2×4]] = [[2, 4],[6, 8]]
C = 2 * A
# print("Scalar Multiplication \n", C)

# --- MULTIPLICAÇÃO MATRICIAL ---
# np.dot(A, B) ou A @ B ou np.matmul(A, B) - todos equivalentes
# Multiplicação matricial NÃO é elemento a elemento!
# result[0][0] = (1×5)+(2×7) = 5+14 = 19
# result[0][1] = (1×6)+(2×8) = 6+16 = 22
# result[1][0] = (3×5)+(4×7) = 15+28 = 43
# result[1][1] = (3×6)+(4×8) = 18+32 = 50
result = np.dot(A, B)
# print("Matrix Multiplication \n", result)

# --- MATRIZ IDENTIDADE 3×3 ---
# np.eye(3) = [[1,0,0],[0,1,0],[0,0,1]]
# Elemento neutro da multiplicação matricial
I = np.eye(3)
# print('Identity Matrix \n', I)

# --- MATRIZ ZERO 2×3 ---
# np.zeros((2,3)) = [[0,0,0],[0,0,0]]
# Matriz completamente preenchida com zeros
Z = np.zeros((2, 3))
# print('Zero Matrix \n', Z)

# --- MATRIZ DIAGONAL ---
# np.diag([1,2,3]) cria:
# | 1  0  0 |
# | 0  2  0 |
# | 0  0  3 |
# Valores na diagonal principal, zeros fora dela
D = np.diag([1, 2, 3])
print('Diagonal Matrix \n', D)
