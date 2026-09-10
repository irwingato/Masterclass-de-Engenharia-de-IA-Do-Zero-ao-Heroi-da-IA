import numpy as np

# Define uma matriz 3x3 chamada A
A = np.array([[2, 3, 4], [4, 5, 6], [7, 8, 9]])

# Calcula o determinante da matriz A
# NOTA: Na versão mais recente do numpy que estou usando, o resultado é 0.0
# No curso do professor (versão mais antiga), o resultado seria diferente.
# Um determinante igual a 0.0 indica que a matriz é singular (não invertível),
# o que faz sentido pois esta matriz tem linhas/colunas linearmente dependentes.
determinant = np.linalg.det(A)

# Calcula a pseudoinversa (inversa generalizada) da matriz A usando Moore-Penrose
# NOTA: Usei np.linalg.pinv ao invés de np.linalg.in que o professor usa,
# pois np.linalg.in dava erro na versão do numpy que estou usando.
# A pseudoinversa funciona mesmo para matrizes singulares (det = 0),
# enquanto a inversa normal não existiria neste caso.
inverse = np.linalg.pinv(A)

print("Determinant: ", determinant)
print("Inverse: ", inverse)