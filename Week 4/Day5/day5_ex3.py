"""day5_ex3.py — ANOVA de UMA via (one-way) com 3 grupos.

OBJETIVO: testar se as MÉDIAS de 3+ grupos independentes são todas iguais,
generalizando o teste t para mais de 2 grupos.

MATEMÁTICA:
- H0: mu1 == mu2 == mu3 (todas as médias populacionais iguais).
- Decomposição da variação total:
    SST = SSB + SSW,
    SST = soma (y - y_barra_geral)^2          (variação total),
    SSB = soma n_k*(y_barra_k - y_barra_geral)^2 (variação ENTRE grupos),
    SSW = soma soma (y_ki - y_barra_k)^2        (variação DENTRO dos grupos).
- Quadrados médios: MSB = SSB/(k-1), MSW = SSW/(N-k), com k = nº grupos.
- Estatística: F = MSB / MSW, com gl1 = k-1, gl2 = N-k.
    F >> 1 sugere que a variação entre grupos domina o ruído interno.
- p = P(F_gl1,gl2 >= F_obs | H0). p < 0.05 => ao menos um grupo difere.
- Pressupostos: independência, normalidade dentro dos grupos e
  homocedasticidade (variâncias iguais entre grupos).
"""
# Importa a ANOVA de uma via (teste F para 3+ amostras independentes).
from scipy.stats import f_oneway

# Dados dos grupos (cada lista = uma amostra independente).
group1 = [10, 12, 14, 16, 18]  # grupo 1: média 14
group2 = [9, 11, 13, 15, 17]   # grupo 2: média 13
group3 = [8, 10, 12, 14, 16]   # grupo 3: média 12

# Executa a ANOVA: calcula SSB, SSW, F = MSB/MSW e o valor-p.
# Retorna (F_obs, p_value). Aqui as médias são próximas e o N é pequeno,
# então espera-se F modesto e p > 0.05 (não rejeita H0).
f_stat, p_value = f_oneway(group1, group2, group3)
print("F-Statistic:", f_stat)  # razão MSB/MSW
print("P-Value:", p_value)     # P(F_2,12 >= F_obs | H0)
