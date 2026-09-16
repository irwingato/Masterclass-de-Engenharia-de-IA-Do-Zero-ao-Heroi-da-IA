# ============================================================================
# day1_ex2.py
# OBJETIVO: Contrastar variável aleatória DISCRETA (dado: PMF em barras) com
# variável aleatória CONTÍNUA (uniforme: PDF em linha).
# CONCEITOS:
#   - PMF (função massa): P(X = x) para cada valor discreto; a soma dá 1.
#   - PDF (função densidade): f(x) para variável contínua; a ÁREA sob a curva
#     dá 1, e P(a <= X <= b) é a área entre a e b (a altura sozinha NÃO é
#     probabilidade).
# ============================================================================

import matplotlib.pyplot as plt
from scipy.stats import uniform
import numpy as np

# --- 1. VARIÁVEL DISCRETA: dado justo (bloco comentado, descomente p/ ver) ---
# 'outcomes' = valores possíveis de X; 'probabilities' = PMF: cada face tem
# P(X = x) = 1/6 ≈ 0,1667 (dado justo). Soma = 6 × 1/6 = 1 (regra da PMF).
# O gráfico de BARRAS é o adequado aqui: um ponto de massa por valor discreto.
# Discrete random variable: Dice roll
# outcomes = [1, 2, 3, 4, 5, 6]
# probabilities = [1/6] * 6
# plt.bar(outcomes, probabilities, color="blue", alpha=0.7)
# plt.title("PMF of a Dice Roll")
# plt.xlabel("Outcome")
# plt.ylabel("Probability")
# plt.show()

# --- 2. VARIÁVEL CONTÍNUA: distribuição Uniforme U(0, 100) ---
# ATENÇÃO: o título original dizia "Uniform(0,1)", mas os parâmetros abaixo
# (loc=0, scale=100) definem a Uniforme de 0 a 100 — título corrigido.
# Na parametrização do scipy: loc = início (a), scale = largura (b - a), ou
# seja, U(a=0, b=100). PDF: f(x) = 1/(b-a) = 1/100 = 0,01 dentro de [0, 100].
# Continuous random variable Uniform distribution
# 'x = np.linspace(0, 100)': grade de 50 pontos igualmente espaçados no
# intervalo [0, 100], onde a PDF será avaliada para desenhar a linha.
x = np.linspace(0, 100)
# 'uniform.pdf(...)': calcula f(x) = 0,01 em cada ponto da grade (constante,
# por isso o gráfico é uma reta horizontal — toda faixa tem igual densidade).
pdf = uniform.pdf(x, loc=0, scale=100)
# Gráfico de LINHA (não barras): variável contínua tem infinitos valores, a
# densidade varia de forma contínua. A área do retângulo 100 × 0,01 = 1.
plt.plot(x, pdf, color="red")
plt.title("PDF of Uniform(0,100)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()