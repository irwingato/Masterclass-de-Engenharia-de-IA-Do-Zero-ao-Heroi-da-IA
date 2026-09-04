# Importa a biblioteca NumPy para trabalhar com arrays e operações numéricas.
import numpy as np

# Criar um array 3D aleatório (formato: 2 matrizes, 3 linhas, 4 colunas)
# Valores inteiros aleatórios entre 1 e 10
np.random.seed(42)  # Garante resultados idênticos a cada execução
# Cria um array tridimensional (3D) de formato (2, 3, 4) com inteiros
# aleatórios de 1 a 10. A estrutura equivale a 2 matrizes, cada uma com
# 3 linhas e 4 colunas.
array_3d = np.random.randint(1, 11, size=(2, 3, 4))

# Exibe o cabeçalho que indica o conteúdo a seguir: o array 3D original.
print("--- Array 3D Original ---")
# Exibe o próprio array 3D gerado.
print(array_3d)
# Exibe o formato (shape/dimensões) do array usando f-string.
print(f"Formato (Shape): {array_3d.shape}\n")

# 2. Calcular estatísticas ao longo de eixoes específicos
print("--- Estatísticas (Média) ---")

# Reduz o eixo 0 (entre as matrizes) -> Resultado será (3, 4)
# Calcula a média colapsando o primeiro eixo (as 2 matrizes). NumPy soma e
# divide os elementos na mesma posição entre as matrizes, gerando um array
# de formato (3, 4).cls

print("Média ao longo do Eixo 0 (colapsa profundidade):")
print(np.mean(array_3d, axis=0))

# Reduz o eixo 1 (entre as linhas) -> Resultado será (2, 4)
# Calcula a média colapsando o segundo eixo (as 3 linhas) de cada matriz.
# O resultado é um array de formato (2, 4).
print("\nMédia ao longo do Eixo 1 (colapsa linhas/vertical):")
print(np.mean(array_3d, axis=1))

# Reduz o eixo 2 (entre as colunas) -> Resultado sera (2, 3)
# Calcula a média colapsando o terceiro eixo (as 4 colunas) de cada linha.
# O resultado é um array de formato (2, 3).
print("\nMédia ao longo do Eixo 2 (colapsa colunas/horizontal):")
print(np.mean(array_3d, axis=2))