import numpy as np

# 1. Gera um array com 8 números inteiros aleatórios entre 1 e 500
array_aleatorio = np.random.randint(1, 500, size=8)
print("🎲 Array Aleatório:", array_aleatorio)

# 2. Encontra o valor mínimo e o máximo
valor_minimo = np.min(array_aleatorio)
valor_maximo = np.max(array_aleatorio)

# 3. Exibe os resultados
print(f"📉 Valor Mínimo: {valor_minimo}")
print(f"📈 Valor Máximo: {valor_maximo}")