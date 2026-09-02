# Importa a biblioteca NumPy para trabalhar com arrays e operações numéricas.
import numpy as np

# Cria uma matriz 3x3 com os números 1 a 9 organizados em 3 linhas e 3 colunas.
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Cria um vetor unidimensional com 3 elementos: 1, 0 e -1.
vector = np.array([1, 0, -1])

# Soma o vetor à matriz usando broadcasting: como o vetor tem o mesmo tamanho
# de cada linha (3 elementos), ele é somado a cada linha da matriz.
result_add = matrix + vector
# Exibe o resultado da adição.
print("Add: ", result_add)

# Multiplica cada elemento da matriz por 2 (escalar). NumPy aplica a
# operação elemento a elemento em todo o array.
result_mul = matrix * 2
# Exibe o resultado da multiplicação.
print("Multiplication: \n", result_mul)