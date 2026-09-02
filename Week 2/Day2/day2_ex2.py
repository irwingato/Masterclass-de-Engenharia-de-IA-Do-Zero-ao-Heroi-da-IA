# Importa a biblioteca NumPy para trabalhar com arrays e operações numéricas.
import numpy as np

# Cria um conjunto de dados (dataset): um array 5x5 de números inteiros
# aleatórios entre 1 (inclusive) e 51 (exclusive), ou seja, de 1 a 50.
dataset = np.random.randint(1, 51, size=(5, 5))
# Exibe o dataset original antes de qualquer modificação.
print("Original: \n" , dataset)


# Filter values > 25 replace with 0
# Índice booleano: todos os elementos maiores que 25 são substituídos por 0.
# Modifica o próprio array original.
dataset[dataset > 25] = 0
# Exibe o dataset após a substituição dos valores acima de 25 por 0.
print("Modified Dataset: \n", dataset)

# calculate summary sets
# Calcula e exibe estatísticas resumidas (agregadas) do dataset:
# Soma de todos os elementos.
print("Sum: ", np.sum(dataset))
# Média aritmética de todos os elementos.
print("Mean: ", np.mean(dataset))
# Desvio padrão dos elementos (mede a dispersão dos dados).
print("Standard Deviation: ", np.std(dataset))