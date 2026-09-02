import numpy as np

matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
print("Matrix Original 4x4:\n", matrix)

# 2. Soma das COLUNAS (Vertical -> de cima para baixo)
soma_colunas = np.sum(matrix, axis=0)
print("\nSoma de cada Coluna (axis=0):", soma_colunas) 
# Resultado: [1+5+9+13, 2+6+10+14, 3+7+11+15, 4+8+12+16] -> [28, 32, 36, 40]

# 3. Soma das LINHAS (Horizontal -> da esquerda para a direita)
soma_linhas = np.sum(matrix, axis=1)
print("Soma de cada Linha (axis=1):", soma_linhas)
# Resultado: [1+2+3+4, 5+6+7+8, 9+10+11+12, 13+14+15+16] -> [10, 26, 42, 58]

# 4. Criação da Segunda Matriz 4x4 (com a sintaxe corrigida)
another_matrix = np.array([
    [16, 15, 14, 13],
    [12, 11, 10, 9],
    [8, 7, 6, 5],
    [4, 3, 2, 1]
])

print("\nSoma das duas Matrizes 4x4:\n", matrix + another_matrix)
print("\nMultiplicação elemento por elemento 4x4:\n", matrix * another_matrix)