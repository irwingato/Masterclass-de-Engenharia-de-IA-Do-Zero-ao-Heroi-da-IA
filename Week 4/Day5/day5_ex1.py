"""day5_ex1.py — Testes t (uma amostra, duas amostras, pareado).

OBJETIVO: testar hipóteses sobre médias quando o desvio-padrão populacional
é desconhecido (usa a distribuição t de Student).

MATEMÁTICA RESUMIDA:
- Teste t de uma amostra:
    H0: media_pop == mu0
    t = (x_barra - mu0) / (s / sqrt(n)), com gl = n - 1.
    s = desvio-padrão amostral. Se |t| grande -> p pequeno -> rejeita H0.
- Teste t de duas amostras independentes (Student/Welch no scipy):
    H0: media1 == media2
    t = (x1_barra - x2_barra) / erro_padrao_da_diferenca.
- Teste t pareado: aplica o teste de uma amostra às diferenças d = pos - pre:
    H0: media(d) == 0,  t = d_barra / (s_d / sqrt(n)), gl = n - 1.
- Em todos: p_value < 0.05 (alfa usual) => evidência contra H0.
"""
# Importa as três variantes do teste t do SciPy.
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel

# ---------------------------------------------------------------------------
# 1) Teste t de UMA amostra: "esta amostra veio de uma população com média 15?"
# ---------------------------------------------------------------------------
data = [12, 14, 15, 16, 17]   # amostra observada (n = 5)
population_mean = 15          # mu0 da hipótese nula H0
# Retorna (estatística t, valor-p bicaudal).
t_stat, p_value = ttest_1samp(data, population_mean)
print("One-Sample T-Test:", t_stat, p_value)

# ---------------------------------------------------------------------------
# 2) Teste t de DUAS amostras INDEPENDENTES: "as médias dos grupos são iguais?"
# ---------------------------------------------------------------------------
group1 = [12, 14, 15, 16, 17]  # grupo A (n1 = 5)
group2 = [11, 13, 14, 15, 16]  # grupo B (n2 = 5), observações não pareadas
# H0: media(group1) == media(group2). Por padrão, bicaudal.
t_stat, p_value = ttest_ind(group1, group2)
print("Two-Sample T-Test:", t_stat, p_value)

# ---------------------------------------------------------------------------
# 3) Teste t PAREADO: mesmos indivíduos antes/depois ("pre" vs "post").
# ---------------------------------------------------------------------------
pre_test = [12, 14, 15, 16, 17]    # medida antes
post_test = [13, 14, 16, 17, 18]   # medida depois (par i-a-i com pre_test)
# Internamente testa se a média das diferenças (post - pre) é zero.
t_stat, p_value = ttest_rel(pre_test, post_test)
print("Paired T-Test:", t_stat, p_value)
