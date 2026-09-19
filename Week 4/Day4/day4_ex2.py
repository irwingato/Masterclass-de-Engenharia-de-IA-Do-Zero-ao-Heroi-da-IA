"""
day4_ex2.py — Teste-t para DUAS AMOSTRAS INDEPENDENTES (Student)
================================================================
Objetivo: testar se as médias populacionais dos grupos 1 e 2 são iguais.

Hipóteses (bicaudal, padrão do scipy):
    H0: mu1 = mu2
    H1: mu1 != mu2

Matemática (Teste-t de Student, 2 amostras, equal_var=True por padrão):
    1) Estatísticas: n1, n2, x1_bar, x2_bar, s1^2, s2^2 (variâncias
       amostrais com ddof=1).
    2) Como equal_var=True, assume-se sigma1 = sigma2 e estima-se a
       variância comum pela variância POOLED (ponderada pelos gl):
           s_p^2 = ((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2)
    3) Erro Padrão da diferença das médias:
           SE = s_p * sqrt(1/n1 + 1/n2)
    4) Estatística de teste:
           t = (x1_bar - x2_bar) / SE  ~  t_{n1+n2-2} sob H0,
       com n1+n2-2 graus de liberdade (aqui 7+7-2=12).
    5) Valor-p bicaudal: p = 2*P(T >= |t|), T ~ t_{12}.
       É o que scipy.stats.ttest_ind retorna junto com t.
    6) Regra: rejeita H0 se p <= alfa (alfa=0.05).

Variante: se as variâncias forem diferentes, usar Welch
(ttest_ind(..., equal_var=False)), com SE = sqrt(s1^2/n1+s2^2/n2)
e gl aproximados de Welch-Satterthwaite.

Pressupostos: independência entre grupos, normalidade aproximada
em cada grupo (relevante pois n1=n2=7, pequenos) e (nesta versão)
igualdade de variâncias.
"""
from scipy.stats import ttest_ind

# Data from two groups (n1=n2=7; diferem por 1 unidade em quase todo ponto,
# então espera-se diferença pequena e p-valor alto)
group1 = [12, 14, 15, 16, 17, 18, 19]
group2 = [11, 13, 14, 15, 16, 17, 18]

# Perform t-test (execução única; o original repetia a chamada duas vezes)
# scipy calcula: t = (mean1-mean2)/SE_pooled e p bicaudal via t_{12}.
t_stat, p_value = ttest_ind(group1, group2)
print("T-Statistic:", t_stat)
print("P-Value", p_value)

# Interpretation
# Lógica: aqui t ~ 0.6, p ~ 0.55 -> diferença das médias (1.0) é menor
# que o ruído amostral (SE), logo não rejeitamos H0 a 5%.
alpha = 0.05
if p_value <= alpha:
    print("Reject the null hypothesis: significant difference")
else:
    print("Fail to reject the null hypothesis: no significant difference")
