import numpy as np

# Criando uma matriz 4x4 para o cálculo de autovalores e autovetores
# Esta é uma matriz maior que as anteriores (2x2 e 3x3) para demonstrar
# que o cálculo de eigenvalues funciona para qualquer tamanho de matriz
A_maior = np.array([
  [1, 0, 2, 3], [6, 8, 0, 2], [8, 0, 2, 3], [2, 0, 1, 7]
])

# Calculando os Autovalores (eigenvalues) e Autovetores (eigenvectors)
# eigenvalues: array com os 4 valores próprios da matriz 4x4
# eigenvectors: matriz 4x4 onde cada coluna é um autovetor correspondente
eigenvalues, eigenvectors = np.linalg.eig(A_maior)

print("--- Matriz Original 4x4 ---")
print(A_maior)

print("\n--- Valores Próprios (Eigenvalues) ---")
# Eles indicam a "força" ou a variância de cada componente da matriz
# .real é usado pois a versão atual do numpy retorna números complexos
# mesmo para matrizes reais, e precisamos extrair a parte real
print(eigenvalues.real)

print("\n--- Vetores Próprios (Eigenvectors) ---")
# Eles indicam a direção ou o sentido de cada componente da matriz
# Cada coluna da matriz eigenvectors é um autovetor
print(eigenvectors.real)