# Importa a biblioteca NumPy, que fornece funções e tipos para trabalhar
# com arrays numéricos e operações matemáticas de forma eficiente.
import numpy as np

# Define uma "semente" (seed) para o gerador de números aleatórios.
# Com a mesma seed, os números gerados serão sempre os mesmos, permitindo
# que os experimentos sejam reproduzíveis.
np.random.seed(42)

# Cria um array 3x3 com números aleatórios entre 0 e 1 (distribuição uniforme).
random_array = np.random.rand(3, 3)
# Exibe o array aleatório gerado. O "\n" insere uma quebra de linha na saída.
print("Random Array: \n", random_array)

# Cria um array 2x3 com números inteiros aleatórios entre 0 (inclusive) e
# 10 (exclusive), ou seja, valores de 0 a 9.
random_integers = np.random.randint(0, 10, size=(2,3))
# Exibe o array de inteiros aleatórios gerado.
print("Random Integers: \n", random_integers)

# # Cria um array unidimensional com os números 1 a 6.
# arr = np.array([1,2,3,4,5,6])

# # Filtra (seleciona) apenas os elementos pares do array usando indexação
# # booleana: arr % 2 == 0 retorna True/False para cada elemento, e a seleção
# # pega apenas onde o resultado é True (números pares).
# evens = arr[arr % 2 == 0]
# # Exibe os elementos pares encontrados.
# print("Evens: ", evens)

# # Atribui 0 a todos os elementos do array que são maiores que 3,
# # modificando o array original (indexação booleana para escrita).
# arr[arr > 3] = 0
# # Exibe o array após a modificação.
# print("Modified Array: ", arr)

# # Cria um array bidimensional (matriz) 2x3.
# arr = np.array([[1,2,3], [4,5,6]])

# # Soma de todos os elementos do array.
# print("Sum: ", np.sum(arr))
# # Média (média aritmética) de todos os elementos.
# print("Mean: ", np.mean(arr))
# # Valor máximo entre todos os elementos.
# print("Max: ", np.max(arr))
# # Valor mínimo entre todos os elementos.
# print("Min: ", np.min(arr))
# # Desvio padrão dos elementos (mede a dispersão dos dados).
# print("Standard Deviation: ", np.std(arr))
# # Soma ao longo das linhas (axis=1). Como há 2 linhas e 3 colunas,
# # são calculadas as somas de cada linha (resultado com 2 valores).
# print("Sum along rows: ", np.sum(arr, axis=1))
# # Soma ao longo das colunas (axis=0). São calculadas as somas de cada
# # coluna (resultado com 3 valores).
# print("Sum along columns: ", np.sum(arr, axis=0))

# # # Array e escalar broadcasting: operações entre vetor e escalar se
# # # estendem automaticamente a todos os elementos (numpy aplica a operação
# # # elemento a elemento).
# arr = np.array([1, 2, 3])
# # Soma 10 a cada elemento do array, resultando em [11, 12, 13].
# print(arr + 10)

# # NOTA: Esta linha contém um erro (falta um colchete). Deveria ser
# # np.array([[1, 2, 3], [4, 5, 6]]) — com dois colchetes [ ] para criar
# # uma matriz 2x3 corretamente.
# matrix = np.array([1, 2, 3], [4, 5, 6])
# vector = np.array([1, 0, 1])
# # Somaria o vetor a cada linha da matriz (broadcasting entre matriz e
# # vetor linha), mas não executa por causa do erro acima.
# print(matrix + vector)