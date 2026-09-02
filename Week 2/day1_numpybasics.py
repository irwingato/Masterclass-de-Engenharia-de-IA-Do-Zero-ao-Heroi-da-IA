import numpy as np

# Cria um array NumPy unidimensional com 6 elementos
arr = np.array([10, 20, 30, 40, 50, 60])

# Acessa o elemento no índice 2 (terceiro elemento) → imprime 30. Indexação começa em 0.
print(arr[2])

# Acessa o último elemento do array → imprime 60. Indexação negativa conta de trás para frente.
print(arr[-1])

# Faz slicing: extrai elementos dos índices 1 ao 3 (4 não incluso) → imprime [20 30 40]
print(arr[1:4])

# Faz slicing a partir do índice 3 até o fim → imprime [40 50 60]
print(arr[3:])

# Redimensiona o array para uma matriz 2×3 (2 linhas, 3 colunas):
# [[10 20 30]
#  [40 50 60]]
reshaped = arr.reshape(2,3)
print(reshaped)

# --- Operações element-wise entre arrays ---
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(a + b)   # Soma correspondente: [5 7 9]
# print(a * b)   # Multiplicação correspondente: [4 10 18]
# print(a / b)   # Divisão correspondente: [0.25 0.4 0.5]

# --- Funções matemáticas ---
# arr = np.array([4, 16, 25])
# print(np.sqrt(arr))  # Raiz quadrada de cada elemento: [2. 4. 5.]
# print(np.sum(arr))   # Soma todos os elementos → 45
# print(np.max(arr))   # Retorna o maior valor → 25
# print(np.max(arr))   # Duplicado na linha 23

# --- Reshape para matriz 3×3 ---
# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# reshaped = arr.reshape((3,3))  # Converte array 1D para matriz 3×3
# print(reshaped)

# --- Expansão de dimensão ---
# arr = np.array([1, 2, 3])
# expanded = arr[:, np.newaxis]  # Converte shape (3,) para (3,1) — transforma em coluna
# print(expanded)

# --- Criação de arrays especiais ---
# arr = np.array([1, 2, 3, 4])
# #print(arr)

# zeroes = np.zeros((3,3))       # Matriz 3×3全是zeros
# #print(zeroes)

# ones = np.ones((2,4))           # Matriz 2×4全是uns
# #print(ones)

# range_array = np.arange(1, 100, 3)  # Sequência de 1 a 99 com passo 3
# #print(range_array)

# linspace_array = np.linspace(0, 1, 3)  # 3 valores equally spaced entre 0 e 1: [0, 0.5, 1]
# print(linspace_array)
