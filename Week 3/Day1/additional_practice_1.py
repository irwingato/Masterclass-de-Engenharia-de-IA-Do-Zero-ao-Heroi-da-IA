import numpy as np

# =============================================================================
# DETERMINANTE E INVERSA DE UMA MATRIZ 2x2
# =============================================================================
#
# MATEMÁTICA POR TRÁS DO CÓDIGO:
#
# Para uma matriz genérica 2x2:
#     | a  b |
#     | c  d |
#
# 1. DETERMINANTE (det):
#    Fórmula: det = (a * d) - (b * c)
#    - O determinante indica se a matriz é invertível
#    - Se det ≠ 0: a matriz TEM inversa (é "não-singular")
#    - Se det = 0: a matriz NÃO tem inversa (é "singular")
#    - Geometricamente, |det| representa a área do paralelogramo
#      formado pelos vetores-linha (ou coluna) da matriz
#
# 2. INVERSA (inv):
#    Para uma matriz 2x2, a inversa é calculada como:
#         1      |  d  -b |
#    A⁻¹ = --- *  | -c   a |
#        det(A)
#
#    A matriz inversa A⁻¹ é tal que: A @ A⁻¹ = I (matriz identidade)
#    É como o "recíproco" da matriz: multiplicar por ela "desfaz" a operação
# =============================================================================

# 1. Defina a matrix 2x2
# np.array cria uma matriz (array bidimensional) no NumPy
# Aqui criamos a matriz A = [[1, 2], [3, 4]]
matriz = np.array([[1, 2], [3, 4]])

# 2. Calcule o determinante
# np.linalg.det calcula o determinante usando o método de decomposição LU
# Para esta matriz: det = (1*4) - (2*3) = 4 - 6 = -2
# Como det = -2 (≠ 0), a matriz TEM inversa
determinante = np.linalg.det(matriz)

# 3. Calcule a matriz inversa
# np.linalg.inv calcula a matriz inversa
# Para esta matriz 2x2:
#         1      |  4  -2 |
#    A⁻¹ = --- * | -3   1 |  =  [[-2,  1], [1.5, -0.5]]
#        -2
#
# Verificação: A @ A⁻¹ deve ser próximo da matriz identidade [[1,0],[0,1]]
matriz_inversa = np.linalg.inv(matriz)

# Exibindo os resultados
print("Matriz Original: \n", matriz)
print("Determinante:", determinante)
print("Matriz Inversa:\n", matriz_inversa)

# BÔNUS: Verificando que A @ A⁻¹ ≈ I (identidade)
print("\n--- Verificação: A @ A⁻¹ (deve ser ≈ Identidade) ---")
verificacao = matriz @ matriz_inversa
print("A @ A⁻¹:\n", verificacao)
