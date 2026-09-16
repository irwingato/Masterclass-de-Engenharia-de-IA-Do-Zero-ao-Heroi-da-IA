# ============================================================================
# day1_sample.py
# OBJETIVO: Calcular de forma EXATA (teórica, sem simulação) a EXPECTATIVA,
# a VARIÂNCIA e o DESVIO-PADRÃO de um DADO JUSTO.
# RESULTADOS ESPERADOS: E(X) = 3,5 | Var(X) = 35/12 ≈ 2,9167 | DP ≈ 1,7078.
# O bloco comentado no topo mostra a regra de Laplace P = favoráveis/possíveis.
# ============================================================================

# from itertools import product

# # Sample space of a dice roll
# # Espaço amostral: todos os resultados possíveis do dado (6 elementos).
# sample_space = list(range(1,7))

# # Probability of rolling an even number
# # Regra de Laplace (casos igualmente prováveis): P = favoráveis / possíveis.
# # Pares = {2, 4, 6} → 3 favoráveis em 6 possíveis → P(par) = 3/6 = 0,5.
# even_numbers = [2, 4, 6]
# P_even = len(even_numbers) / len(sample_space)
# print("P(Even):", P_even)

import numpy as np

# --- 1. DEFINIÇÃO DA VARIÁVEL ALEATÓRIA (dado justo) ---
# 'outcomes' = valores possíveis de X (faces 1 a 6).
# 'probabilities' = PMF uniforme: P(X = x) = 1/6 para toda face (dado justo).
# Random variable: dice roll
outcomes = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6] * 6)

# --- 2. EXPECTATIVA: E(X) = Σ x·P(x) ---
# Média ponderada pelas probabilidades. Para o dado justo:
# E(X) = (1+2+3+4+5+6)/6 = 21/6 = 3,5 (o "centro" da distribuição).
# 'outcomes * probabilities' multiplica termo a termo; np.sum soma tudo.
# Expectation
expectation = np.sum(outcomes * probabilities)
print("Expectation (Mean):", expectation)

# --- 3. VARIÂNCIA E DESVIO-PADRÃO ---
# Fórmula usada aqui: Var(X) = Σ (x − E(X))²·P(x) — média dos desvios
# quadráticos em relação à média (equivale a E(X²) − [E(X)]²).
# Para o dado justo: Var = [(1−3,5)² + ... + (6−3,5)²]/6 = 17,5/6 = 35/12
# ≈ 2,9167. O desvio-padrão DP = √Var ≈ 1,7078 volta à unidade "faces".
# Variance and Standard Deviation
variance = np.sum((outcomes - expectation) **2 * probabilities)
std_dev = np.sqrt(variance)
print("Variance: ", variance)
print("Standard Deviation: ", std_dev)