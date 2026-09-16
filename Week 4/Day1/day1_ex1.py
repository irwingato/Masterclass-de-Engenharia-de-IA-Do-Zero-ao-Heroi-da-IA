# ============================================================================
# day1_ex1.py
# OBJETIVO: Estimar por SIMULAÇÃO (Monte Carlo) duas probabilidades de um
# dado justo lançado 10.000 vezes: P(par) e P(maior que 4).
# IDEIA CENTRAL: a frequência observada converge para a probabilidade teórica
# (Lei dos Grandes Números). Valores teóricos: P(par) = 3/6 = 0,50 e
# P(>4) = 2/6 ≈ 0,333.
# ============================================================================

import numpy as np

# --- 1. SIMULAÇÃO DOS LANÇAMENTOS ---
# np.random.randint(1, 7, size=10000): sorteia 10.000 inteiros no intervalo
# [1, 7), ou seja, de 1 a 6 — cada face com chance 1/6 (dado justo).
# O resultado é um array numpy com os 10 mil lançamentos simulados.
rolls = np.random.randint(1, 7, size=10000)

# --- 2. CÁLCULO DAS PROBABILIDADES EXPERIMENTAIS (frequentista) ---
# P(evento) = (nº de vezes que o evento ocorreu) / (nº total de lançamentos).
# 'rolls % 2 == 0' cria uma máscara booleana (True onde a face é par:
# 2, 4 ou 6); np.sum conta os True (True vale 1). Dividir por len(rolls)
# dá a frequência relativa, que estima a probabilidade teórica.
P_even = np.sum(rolls % 2 == 0) / len(rolls)

# Mesmo raciocínio: 'rolls > 4' marca True só para as faces 5 e 6.
# Teórico: 2 faces favoráveis em 6 possíveis = 2/6 ≈ 0,333.
P_greater_than_4 = np.sum(rolls > 4) / len(rolls)

# --- 3. EXIBIÇÃO ---
# Com 10 mil lançamentos, esperamos P_even ≈ 0,50 e P_greater_than_4 ≈ 0,33.
print("P(Even): " , P_even)
print("P(Greater than 4): ", P_greater_than_4)