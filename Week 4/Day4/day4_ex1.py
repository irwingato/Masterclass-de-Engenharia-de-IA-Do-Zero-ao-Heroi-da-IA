"""
day4_ex1.py — Teste-t para UMA AMOSTRA (one-sample t-test)
============================================================
Objetivo: testar se a média populacional de onde veio 'data' é igual a 15.

Hipóteses (bicaudal, padrão do scipy):
    H0: mu = 15
    H1: mu != 15

Matemática (Teste-t de Student, 1 amostra):
    1) Estatísticas amostrais: n = len(data),
       x_bar = média amostral, s = desvio padrão amostral (ddof=1).
    2) Erro Padrão da média: SE = s / sqrt(n).
       Mede a incerteza de x_bar como estimador de mu.
    3) Estatística de teste (padronização usando s no lugar de sigma):
           t = (x_bar - mu0) / SE  ~  t_{n-1} sob H0,
       onde t_{n-1} é a distribuição t de Student com n-1 graus de
       liberdade. Ela tem caudas mais pesadas que a Normal, compensando
       a incerteza extra de estimar sigma por s (crucial para n pequeno;
       aqui n=7!). Quando n -> infinito, t_{n-1} -> N(0,1).
    4) Valor-p bicaudal:
           p = 2 * P(T >= |t|), com T ~ t_{n-1}.
       É o que scipy.stats.ttest_1samp retorna junto com t.
    5) Regra: rejeita H0 se p <= alfa (alfa=0.05).

Pressupostos: observações i.i.d. e aproximadamente normais
(importante com n=7, onde o TCL ainda não atua).
"""
import numpy as np
from scipy.stats import ttest_1samp

# Sample data (amostra observada; n=7, amostra pequena -> teste-t, não Z)
data = [12, 14, 15, 16, 17, 18, 19]

# Null Hypothesis: mean = 15
# mu0 = valor da média populacional sob H0. A pergunta é: x_bar está
# longe de 15 demais para ser explicado pelo acaso amostral?
population_mean = 15

# Perform t-test (execução única; o código original repetia esta linha
# duas vezes sem necessidade — mantida apenas uma chamada)
# scipy calcula internamente: t = (mean(data)-mu0) / (std(data,ddof=1)/sqrt(n))
# e p bicaudal a partir da CDF da t_{n-1}.
t_stat, p_value = ttest_1samp(data, population_mean)
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Interpret Results
# Lógica: p pequeno = dados incompatíveis com H0.
# Ex.: t ~ 1.39, p ~ 0.21 aqui -> diferença (x_bar=15.86 vs 15) cabe no
# erro padrão, então não rejeitamos H0 a 5%.
alpha = 0.05
if p_value <= alpha:
    print("Reject the null hypothesis: significant difference")
else:
    print("Fail to reject the null hypothese: no significant difference")
