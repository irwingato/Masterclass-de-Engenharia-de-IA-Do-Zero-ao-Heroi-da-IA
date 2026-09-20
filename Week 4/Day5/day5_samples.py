"""day5_samples.py — Exemplos de ANOVA (bloco qui-quadrado comentado + one-way ativa).

OBJETIVO: arquivo de rascunho/aula mostrando dois testes; apenas a ANOVA
de uma via está ativa. O bloco qui-quadrado ficou comentado como referência.

MATEMÁTICA:
- Qui-quadrado (comentado): H0 de independência entre 2 categóricas 2x2,
    qui2 = soma (O-E)^2/E, gl = (2-1)*(2-1) = 1.
- ANOVA one-way (ativa): H0: mu1 == mu2 == mu3.
    F = MSB/MSW = [SSB/(k-1)] / [SSW/(N-k)], gl1 = k-1, gl2 = N-k.
    Aqui k = 3 grupos, N = 15 observações -> gl = (2, 12).
"""
# ANOVA de uma via para 3 amostras independentes (mesma teoria do day5_ex3.py).
from scipy.stats import f_oneway

# ---------------------------------------------------------------------------
# BLOCO COMENTADO — referência de qui-quadrado (não executa).
# Tabela 2x2: data = [[50, 30], [20, 40]]; testaria independência linha x coluna
# com gl = 1. Mantido comentado para focar a aula na ANOVA abaixo.
# ---------------------------------------------------------------------------
# from scipy.stats import chi2_contingency

# # Contingency Table
# data = [[50, 30], [20, 40]]

# # Perform Chi-Square Test
# chi2, p, dof, expected = chi2_contingency(data)
# print("Chi-Square Statistic:", chi2)
# print("P-Value:", p)
# print("Expected Frequencies: \n", expected)

# Dados para três grupos (amostras independentes, n = 5 cada).
group1 = [12, 14, 15, 16, 17]  # média 14.8
group2 = [11, 13, 14, 15, 16]  # média 13.8
group3 = [10, 12, 13, 14, 15]  # média 12.8

# ANOVA one-way: F = MSB/MSW; p = P(F_2,12 >= F_obs | H0).
# Médias próximas + N pequeno => p alto (não rejeita igualdade das médias).
f_stat, p_value = f_oneway(group1, group2, group3)
print("F-Statistic:", f_stat)
print("P-Value:", p_value)
