import numpy as np

# 1. Criando um conjunto de dados fictício (10 instâncias/linhas e 5 colunas)
# Cada linha representa uma observação e cada coluna uma característica
np.random.seed(42) # Mantém os números iguais a cada execução (reprodutibilidade)
dados_originais = np.random.rand(10, 5)

print("--- Dados Originais (10 linhas x 5 colunas) ---")
print(dados_originais.shape)
print(dados_originais)

# 2. Aplicando a Decomposição SVD (Singular Value Decomposition)
# U: Matriz ortogonal de vetores singulares à esquerda (10x5 com full_matrices=False)
# S: Array com os valores singulares (5 valores), indicam a importância de cada dimensão
# Vt: Transposta da matriz ortogonal de vetores singulares à direita (5x5)
U, S, Vt = np.linalg.svd(dados_originais, full_matrices=False)

# 3. Definindo o número de dimensões que queremos manter (Redução de 5D para 2D)
n_componentes = 2

# 4. Reduzindo a dimensionalidade dos dados
# Multiplicamos as primeiras 'n' colunas de U pelos primeiros 'n' valores de S
# Isso projeta os dados originais de 5 dimensões para apenas 2 dimensões
# preservando a maior parte da informação (variância) dos dados
dados_reduzidos = U[:, :n_componentes] * S[:n_componentes]

print("\n--- Dados Reduzidos (10 linhas x 2 colunas) ---")
print(dados_reduzidos.shape)
print(dados_reduzidos)

# 5. Calculando a porcentagem de informação (variância) mantida após a redução
# Comparamos a soma dos quadrados dos primeiros valores singulares com a soma total
# Usei * 100 (multiplicação) ao invés de ** 100 (potência) que estava causando erro
variancia_mantida = (np.sum(S[:n_componentes]**2) / np.sum(S**2)) * 100
print(f"\n--- Porcentagem de preservada após a redução: {variancia_mantida:.2f}%")