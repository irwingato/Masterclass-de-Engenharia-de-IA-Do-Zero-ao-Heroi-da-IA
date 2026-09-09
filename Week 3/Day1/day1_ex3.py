import numpy as np

# =============================================================================
# MATRIZES ESPECIAIS: IDENTIDADE, DIAGONAL E ZERO
# =============================================================================
#
# MATEMÁTICA:
#
# 1. MATRIZ IDENTIDADE (I):
#    - É a matriz "neutra" da multiplicação (assim como 1 é para números)
#    - Tem 1s na diagonal principal e 0s em todo o resto
#    - Para qualquer matriz A: A × I = I × A = A
#    - np.eye(n) cria uma matriz identidade n×n
#
# 2. MATRIZ DIAGONAL:
#    - Tem valores APENAS na diagonal principal (todos os outros são 0)
#    - np.diag([d1, d2, ..., dn]) cria uma matriz n×n com esses valores
#      na diagonal principal
#
# 3. MATRIZ ZERO:
#    - Todos os elementos são 0
#    - É o elemento neutro da ADIÇÃO (assim como 0 é para números)
#    - Para qualquer matriz A: A + Z = A
# =============================================================================

# --- MATRIZ IDENTIDADE ---
# np.eye(3) cria uma matriz identidade 3×3:
#     | 1  0  0 |
#     | 0  1  0 |
#     | 0  0  1 |
# O "eye" vem de "I" (identidade) em inglês
I = np.eye(3)

A = np.array([[1, 2, 3],   # Matriz A 3×3
              [4, 5, 6],
              [7, 8, 9]])

# A × I = A (a identidade não altera a matriz original)
# Cada linha de A é multiplicada pela identidade:
# Linha 1: (1×1)+(2×0)+(3×0)=1, (1×0)+(2×1)+(3×0)=2, (1×0)+(2×0)+(3×1)=3
# Resultado: mesma matriz A
print("A X I:\n", np.dot(A, I))

# --- MATRIZ DIAGONAL ---
# np.diag([1, 7, 9]) cria:
#     | 1  0  0 |
#     | 0  7  0 |
#     | 0  0  9 |
# Apenas os valores [1, 7, 9] aparecem na diagonal principal
# Todos os outros elementos são zero
D = np.diag([1, 7, 9])

# --- MATRIZ ZERO ---
# np.zeros((3, 3)) cria uma matriz 3×3 totalmente preenchida com zeros:
#     | 0  0  0 |
#     | 0  0  0 |
#     | 0  0  0 |
# É como o "zero" da álgebra matricial
Z = np.zeros((3, 3))

print("Diagonal Matrix\n", D)
print("Zero Matrix\n", Z)