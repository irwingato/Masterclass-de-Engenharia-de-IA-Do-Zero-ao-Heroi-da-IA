"""
additional_practice_2.py — Teste Z para DUAS MÉDIAS com dados ÍRIS
===================================================================
Objetivo: testar se o comprimento médio da sépala (sepal_length) difere
entre duas espécies: setosa vs versicolor.

Hipóteses:
    H0: mu_setosa = mu_versicolor  (diferença das médias = 0)
    H1: mu_setosa != mu_versicolor (teste bicaudal)

Matemática (Teste Z para 2 amostras independentes):
    1) Médias amostrais: m1 = mean(setosa), m2 = mean(versicolor).
       Desvios-padrão amostrais: s1, s2 (com ddof=1, i.e. divididos por n-1,
       estimadores não-viesados da variância populacional).
    2) Pelo Teorema Central do Limite, a diferença das médias é
       aproximadamente Normal:
           (m1 - m2) ~ Normal(mu1 - mu2, SE^2),
       onde o Erro Padrão (caso não-pareado, variâncias diferentes —
       análogo a Welch) é:
           SE = sqrt( s1^2/n1 + s2^2/n2 )
       Esta fórmula vem de: Var(m1)=sigma1^2/n1, Var(m2)=sigma2^2/n2,
       e Var(m1-m2)=Var(m1)+Var(m2) por independência.
    3) Estatística de teste (padronização sob H0, onde mu1-mu2=0):
           Z = (m1 - m2) / SE  ~  N(0,1) sob H0
    4) Valor-p bicaudal:
           p = 2 * (1 - Phi(|Z|)), com Phi = FDA da Normal padrão.
       Rejeita-se H0 se p < alfa (aqui alfa = 0.05).

Nota metodológica importante:
    Cada espécie tem n=50 no dataset Íris. Com n pequeno e sigma
    DESCONHECIDO, o rigoroso seria o teste-t de Welch
    (scipy.stats.ttest_ind com equal_var=False), não o Teste Z.
    O Teste Z aqui é uma aproximação didática que trata s1,s2 como
    se fossem os sigmas populacionais. Para n grande (n>30 por grupo
    como regra prática) a aproximação é razoável pelo TCL.
"""
import pandas as pd
import numpy as np
from scipy import stats

# 1. Carregar o conjunto de dados a partir da URL fornecida
#    O CSV contém 150 linhas (50 por espécie) e colunas como
#    'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'.
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# 2. Separar o comprimento da sépala (sepal_length) para duas espécies
#    Filtragem booleana: df[condição][coluna] -> Series com os valores.
#    H0 compara as médias populacionais dessas duas subpopulações.
setosa = df[df['species'] == 'setosa']['sepal_length']
versicolor = df[df['species'] == 'versicolor']['sepal_length']

# 3. Coletar os parâmetros necessários (n, média e desvio padrão)
#    - n: tamanho de cada amostra.
#    - m: média amostral, estimador de mu.
#    - s: desvio padrão amostral com ddof=1 (divisor n-1), estimador
#      não-viesado de sigma. CORREÇÃO: era 'ddpf=1' (erro de digitação
#      que gerava TypeError); o parâmetro correto do pandas é 'ddof'.
n1, n2 = len(setosa), len(versicolor)
m1, m2 = setosa.mean(), versicolor.mean()
s1, s2 = setosa.std(ddof=1), versicolor.std(ddof=1)

# 4. Calcular o Erro Padrão da diferença entre as duas médias
#    Fórmula: SE = sqrt(s1^2/n1 + s2^2/n2).
#    Intuição: quanto maior a dispersão (s) ou menor o n, maior a
#    incerteza sobre a diferença das médias.
se = np.sqrt((s1**2 / n1) + (s2**2 / n2))

# 5. Calcular a estatística Z (quantos SEs a diferença está de zero)
#    Fórmula: Z = (m1 - m2) / SE. Sob H0, Z ~ N(0,1).
z_stat = (m1 - m2) / se

# 6. Calcular o Valor-p (Bicaudal)
#    Fórmula: p = 2 * (1 - Phi(|Z|)).
#    Multiplica-se por 2 porque desvios extremos em AMBAS as direções
#    contam como evidência contra H0 bicaudal.
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Exibir os resultados obtidos
print(f"--- Resultados do Teste Z (Iris) ---")
print(f"Média Setosa: {m1:.3f} cm (n={n1})")
print(f"Média Versicolor: {m2:.3f} cm (n={n2})")
print(f"Estatística Z: {z_stat:.4f}")
print(f"Valor-p: {p_value:.4e}")

# Conclusão Estatística (Alfa = 5%)
# Regra: p < alfa -> rejeita H0 (diferença significativa).
# Aqui espera-se p << 0.05, pois setosa tem sépalas bem menores.
alpha = 0.05
if p_value < alpha:
    print("\nResultado: Rejeita-se H0. A diferença entre as médias é estatisticamente significativa.")
else:
    print("\nResultado: Não se rejeita H0. Não há diferença estatisticamente significativa.")
