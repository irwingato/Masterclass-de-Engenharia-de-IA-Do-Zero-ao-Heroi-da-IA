import numpy as np

# =============================================================================
# MULTIPLICAÇÃO MATRIZ-VETOR
# =============================================================================
#
# MATEMÁTICA:
# A multiplicação de uma matriz M (m×n) por um vetor v (n×1) resulta em
# um novo vetor (m×1). Cada elemento do resultado é o PRODUTO ESCALAR
# entre uma linha da matriz e o vetor.
#
# Para M (3×3) e v (3×1):
#     | m11  m12  m13 |   | v1 |   | m11×v1 + m12×v2 + m13×v3 |
#     | m21  m22  m23 | × | v2 | = | m21×v1 + m22×v2 + m23×v3 |
#     | m31  m32  m33 |   | v3 |   | m31×v1 + m32×v2 + m33×v3 |
#
# IMPORTANTE: O número de COLUNAS da matriz deve ser igual ao tamanho do vetor
# =============================================================================

# Create matrix and vector
# M é uma matriz 3×3 e v é um vetor de 3 elementos
M = np.array([[1, 2, 3],   # Linha 1: [1, 2, 3]
              [4, 5, 6],   # Linha 2: [4, 5, 6]
              [7, 8, 9]])  # Linha 3: [7, 8, 9]

v = np.array([1, 0, -1])  # Vetor: [1, 0, -1]

# Matrix-vector multiplication (Multiplicação Matriz-Vetor)
# np.dot(M, v) calcula o produto escalar de cada linha de M com o vetor v
#
# Cálculo passo a passo:
# resultado[0] = (1×1) + (2×0) + (3×-1) = 1 + 0 - 3 = -2
# resultado[1] = (4×1) + (5×0) + (6×-1) = 4 + 0 - 6 = -2
# resultado[2] = (7×1) + (8×0) + (9×-1) = 7 + 0 - 9 = -2
#
# Resultado final: [-2, -2, -2]
#
# Alternativas equivalentes: M @ v ou np.matmul(M, v)
result = np.dot(M, v)
print("Matrix-vector Multiplication: \n", result)