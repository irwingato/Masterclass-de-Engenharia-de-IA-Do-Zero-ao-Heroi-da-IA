"""day5_ex2.py — Teste Qui-Quadrado de independência/associação.

OBJETIVO: testar se duas variáveis categóricas são independentes a partir
de uma tabela de contingência (contagens observadas).

MATEMÁTICA:
- H0: as variáveis (linha x coluna) são INDEPENDENTES.
- Frequência esperada sob H0 para a célula (i,j):
    E_ij = (total_linha_i * total_coluna_j) / total_geral.
- Estatística de teste:
    qui2 = soma_ij (O_ij - E_ij)^2 / E_ij,
    onde O_ij = contagem observada.
- Graus de liberdade: gl = (n_linhas - 1) * (n_colunas - 1).
    Aqui: tabela 2x3 -> gl = (2-1)*(3-1) = 2.
- p pequeno (< 0.05) => rejeita independência (há associação).
- Pressuposto: E_ij >= 5 na maioria das células (aproximação qui-quadrado).
"""
# Importa a função do teste qui-quadrado para tabelas de contingência.
from scipy.stats import chi2_contingency

# Tabela de contingência 2 linhas x 3 colunas (contagens observadas O_ij).
# Ex.: linha 0 = [50, 30, 20], linha 1 = [30, 40, 30].
data = [[50, 30, 20], [30, 40, 30]]

# Executa o teste. Retorna:
#   chi2     = estatística soma (O-E)^2/E,
#   p        = valor-p (cauda direita da distribuição qui2 com gl graus),
#   dof      = graus de liberdade (2 neste exemplo),
#   expected = matriz de frequências esperadas E_ij sob H0.
chi2, p, dof, expected = chi2_contingency(data)
print("Chi-Square Statistic:", chi2)  # quanto maior, mais longe de H0
print("P-Value:", p)                  # P(Qui2_gl >= chi2 | H0)
print("Expected Frequencies: \n", expected)  # E_ij para comparar com O_ij
