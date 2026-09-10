import numpy as np

# Define uma matriz 2x2 chamada A
A = np.array([[4, -2], [1, 1]])

# Calcula os autovalores (eigenvalues) e autovetores (eigenvectors) da matriz A
# Os autovalores são os valores λ que satisfazem A @ v = λ * v
# Os autovetores são os vetores v que satisfazem essa equação
# NOTA: Na versão mais nova do numpy que estou usando, os resultados podem vir
# como números complexos (com parte imaginária) mesmo para matrizes reais,
# por isso usamos .real para extrair apenas a parte real e mostrar o mesmo
# resultado que o professor apresenta no curso da Udemy (versão mais antiga do numpy)
eigvals, eigvec = np.linalg.eig(A)

print("EigenVlaues:", eigvals.real)
print("EigenVectors: \n", eigvec.real)