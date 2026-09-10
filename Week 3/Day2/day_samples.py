import numpy as np

# Define uma matriz 2x2 chamada A
A = np.array([[2,3], [1, 4]])

# Calcula o determinante da matriz A
determinant = np.linalg.det(A)
# print("Determinant:", determinant)

# Decomposição em Valores Singulares (SVD): A = U @ diag(S) @ Vt
# U: matriz ortogonal de vetores singulares à esquerda
# S: array com os valores singulares (não a matriz diagonal, apenas os valores)
# Vt: transposta da matriz ortogonal de vetores singulares à direita
U, S, Vt = np.linalg.svd(A)
print("U: \n", U)
print("Singular Values: \n", S)
print("V Transpose: \n", Vt)



# Calcula a inversa da matriz A (A^-1, tal que A @ A^-1 = I)
inverse = np.linalg.inv(A)
# print("Inverse of A: \n", inverse)

# Calcula autovalores e autovetores da matriz A
# Na versão do numpy que estou usando, os resultados podem vir como números complexos
# mesmo para matrizes reais, por isso usamos .real para extrair apenas a parte real
# e mostrar o mesmo resultado que o professor apresenta no curso da Udemy
eigenValues, eigneVectors = np.linalg.eig(A)
# print("EigenVal\n", eigenValues.real)
# print("EigenVectors\n", eigneVectors.real)

# Outra matriz 2x2 para teste
B = np.array([[4, 2], [1, 1]])
# eigval, eigvec = np.linalg.eig(B)
# print("EighVal: ", eigval.real)
# print("EigBVect: \n", eigvec.real)