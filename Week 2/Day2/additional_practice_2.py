# Importa a biblioteca NumPy para trabalhar com arrays e operações numéricas.
import numpy as np

# 1. Gerar conjunto de dados aleatórios (floats entre 10.0 e 50.0)
# Define a semente (seed) 42 para que os números aleatórios gerados sejam
# sempre os mesmos a cada execução (resultados reproduzíveis).
np.random.seed(42)
# Cria um array 3x4 de números float (ponto flutuante) uniformemente
# distribuídos entre 10.0 e 50.0.
dados = np.random.uniform(10.0, 50.0, size=(3,4))

# Exibe o cabeçalho indicando que serão mostrados os dados originais.
print(("--- Dados Originais (Floats) ---"))
# Exibe o array de dados originais gerado.
print(dados)

# 2. Encontrar os valors mínimo o máximo global
# Obtém o menor valor de todo o array (mínimo global).
val_min = np.min(dados)
# Obtém o maior valor de todo o array (máximo global).
val_max = np.max(dados)

# 3. Aplicar a fórmula de normalização Min-Max
# Normaliza os dados para o intervalo [0, 1] subtraindo o mínimo e dividindo
# pela diferença entre máximo e mínimo. O menor valor vira 0 e o maior vira 1.
dados_normalizados = (dados - val_min) / (val_max - val_min)

# Exibe o cabeçalho indicando os dados normalizados.
print("\n--- Dados Normalizados (Entre 0 e 1) ---")
# Exibe o array de dados já normalizados.
print(dados_normalizados)