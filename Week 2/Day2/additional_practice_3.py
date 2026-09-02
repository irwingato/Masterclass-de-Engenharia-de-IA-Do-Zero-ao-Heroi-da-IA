import numpy as np

# 1. Gerar dados de exemplo (normalizados entre 0 e 1)
# np.random.seed fixa a semente para que os números aleatórios sejam reproduzíveis
np.random.seed(42)
# np.random.rand(3, 4) gera uma matriz 3x4 com valores uniformes no intervalo [0, 1)
dados = np.random.rand(3, 4)

print("--- Dados Originais ---")
print(dados)

# Definir o limite (threshold) que será usado nas comparações
limite = 0.5

# --- Abordagem 1: Máscara Booleana Direta (Mais rápida) ---
# Compara cada elemento com o limite; resultado é uma matriz de True (acima) e False (abaixo/igual)
mascara_bool = dados > limite

print(f"\n--- Máscara Booleana (Acima de {limite}) ---")
print(mascara_bool)

# --- Abordagem 2: Máscara Binária de Inteiros (0 e 1) ---
# O método .astype(int) converte True para 1 e False para 0
mascara_binaria = (dados > limite).astype(int)

print(f"\n--- Máscara Binária Inteira (0 e 1) ---")
print(mascara_binaria)

# --- Bonus: substituição Condicional com np.where ---
# np.where(condição, valor_se_true, valor_se_false):
# Se for maior que o limite vira 1, senão vira 0 (útil para outros valores)
mascara_customizada = np.where(dados > limite, 1, 0)

# Exibe a máscara customizada gerada pelo np.where
print(f"\n--- Máscara Customizada (np.where) ---")
print(mascara_customizada)