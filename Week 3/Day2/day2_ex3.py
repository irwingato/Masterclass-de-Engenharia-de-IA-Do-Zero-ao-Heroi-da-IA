import numpy as np

# Define uma matriz 3x3 chamada A
A = np.array([[3, 1, 1], [-1, 3, 1], [1, 1, 3]])

# Decomposição em Valores Singulares (SVD): A = U @ diag(S) @ Vt
# U: matriz ortogonal 3x3 de vetores singulares à esquerda
# S: array com os 3 valores singulares (não é a matriz diagonal, apenas os valores)
# Vt: transposta da matriz ortogonal 3x3 de vetores singulares à direita
U, S, Vt = np.linalg.svd(A)

print("U: \n", U)
print("Singular Values: \n", S)
print("V Transpose: \n", Vt)

# Reconstrução da matriz original a partir da decomposição SVD
# Cria uma matriz diagonal 3x3 zerada
Sigma = np.zeros((3, 3))

# Preenche a diagonal da matriz Sigma com os valores singulares S
np.fill_diagonal(Sigma, S)

# Reconstrói a matriz original multiplicando: U @ Sigma @ Vt
# O resultado deve ser igual (ou muito próximo) da matriz A original
reconstructed = U @ Sigma @ Vt
print("Reconstructed Matrix \n", reconstructed)