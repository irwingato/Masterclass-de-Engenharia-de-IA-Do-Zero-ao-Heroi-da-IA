"""
day7_ex1.py — Exercício guiado do Dia 7: Estatística + Regressão no dataset Tips.
================================================================================
DATASET: tips.csv (Seaborn) — 244 gorjetas em restaurante.
Colunas: total_bill (conta $), tip (gorjeta $), sex, smoker, day, time, size.

OBJETIVO DIDÁTICO: mostrar o pipeline clássico de análise exploratória:
  1) Carregar e inspecionar dados (pandas)
  2) Testar independência entre categóricas (Qui-Quadrado)
  3) Ver distribuição e correlação (histograma + heatmap)
  4) Testar diferença de médias (Teste-T)
  5) Modelar relação contínua (Regressão Linear Simples)

MATEMÁTICA GERAL: todo teste de hipótese segue o mesmo ritual:
  H0 (nula) vs H1 (alternativa), estatística de teste, p-valor,
  nível alpha=0.05. Se p <= alpha, rejeita H0.
"""

# ---------------------------------------------------------------------------
# IMPORTS — o que cada biblioteca faz (computacional)
# ---------------------------------------------------------------------------
import pandas as pd  # DataFrame: tabela 2D com index + colunas tipadas
import seaborn as sns  # gráficos estatísticos de alto nível (usa matplotlib por baixo)
import matplotlib.pyplot as plt  # figura/eixos, renderização da janela do gráfico
from scipy.stats import chi2_contingency  # teste Qui-Quadrado de independência
from sklearn.linear_model import LinearRegression  # MQO (Mínimos Quadrados Ordinários)

# ---------------------------------------------------------------------------
# 1. CARGA DOS DADOS
# Computacional: pd.read_csv(url) baixa o CSV via HTTP e infere tipos.
#   - numéricas -> float64/int64 | categóricas -> object (string).
# ---------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# ---------------------------------------------------------------------------
# 2. TESTE QUI-QUADRADO (comentado, mas documentado)
# Pergunta: "ser fumante (smoker) depende do turno (time: Lunch/Dinner)?"
# Matemática: monta tabela de contingência O[i,j] (frequências observadas).
#   H0: variáveis independentes. Frequência esperada sob H0:
#     E[i,j] = (total_linha_i * total_coluna_j) / total_geral
#   Estatística: chi2 = soma((O - E)^2 / E), com gl = (linhas-1)*(colunas-1).
#   p = 1 - CDF_chi2(chi2). Se p <= 0.05, há dependência.
# Computacional: pd.crosstab conta co-ocorrências; chi2_contingency retorna
#   (chi2, p, graus_liberdade, matriz_esperada).
# ---------------------------------------------------------------------------
# contingency_table = pd.crosstab(df['smoker'], df['time'])

# Perform Ch-Square Test
# chi2, p, dof, expected = chi2_contingency(contingency_table)
# print("Chi-Square Statistic:", chi2)
# print("P-Value:", p)

# Interpret Result
# alpha = 0.05
# if p <= alpha:
#     print("Reject the null hypothesis: Variables are dependent")
# else:
#     print("Fail to reject the null hypothesis: Variables are independent.")

# ---------------------------------------------------------------------------
# 3. INSPEÇÃO (comentado)
# df.info() = nº linhas, tipos, memória, nulos. df.describe() = count, mean,
# std, min, 25%/50%/75%, max. Essencial antes de qualquer conta: detecta
# nulos e outliers que quebrariam correlação/regressão.
# ---------------------------------------------------------------------------
# print(df.info())
# print(df.describe())

# del df["sex"] / ["smoker"] / ["day"] / ["time"]
# Computacional: remove colunas categóricas para sobrar só numéricas.
# Por que? df.corr() antigo falhava com strings. Hoje o correto é
# df.select_dtypes(include='number'), como feito no additional_practice1.py.

# ---------------------------------------------------------------------------
# 4. HISTOGRAMA + KDE (comentado)
# Matemática: histograma estima a densidade empírica por bins;
# KDE (Kernel Density Estimation) suaviza: f_hat(x) = (1/nh)*soma K((x-xi)/h),
# com kernel gaussiano K e largura h. Mostra assimetria da conta (cauda à direita).
# ---------------------------------------------------------------------------
# sns.histplot(df["total_bill"], kde=True)
# plt.title("Distribution of Total Bill")
# plt.show()

# ---------------------------------------------------------------------------
# 5. HEATMAP DE CORRELAÇÃO (comentado)
# Matemática: correlação de Pearson r = cov(X,Y)/(stdX*stdY), em [-1,1].
#   r~1 = sobem juntas; r~-1 = opostas; r~0 = sem relação linear.
# Computacional: df.corr() calcula matriz r par-a-par; sns.heatmap colore.
# Esperado aqui: total_bill x tip com r ~ 0.68 (correlação positiva moderada).
# ---------------------------------------------------------------------------
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()

# from scipy.stats import ttest_ind

# ---------------------------------------------------------------------------
# 6. TESTE-T DE DUAS AMOSTRAS (comentado)
# Pergunta: "homens dão gorjeta diferente de mulheres?"
# Matemática: H0: médias iguais. Estatística (Welch/Student):
#   t = (media1 - media2) / sqrt(s1²/n1 + s2²/n2).
#   Sob H0, t ~ t-Student(gl). p bicaudal = 2*P(T > |t|).
# ---------------------------------------------------------------------------
# male_tips = df[df['sex'] == 'Male']['tip']
# female_tips = df[df['sex'] == 'Female']['tip']
# t_stat, p_value = ttest_ind(male_tips, female_tips)
# print("T-Statistic:", t_stat)
# print("P-Value:", p_value)
# alpha = 0.05
# if p_value <= alpha:
#     print("Reject all null hypothesis: Significant difference.")
# else:
#     print("Fail to Reject the null hypothesis: NO significant difference.")

# ---------------------------------------------------------------------------
# 7. REGRESSÃO LINEAR SIMPLES — PARTE ATIVA DO SCRIPT
# Modelo: tip = b0 + b1*total_bill + erro.
# Matemática (MQO): escolhe b0,b1 que minimizam S = soma(yi - (b0+b1*xi))².
#   Solução fechada: b1 = cov(X,y)/var(X); b0 = media_y - b1*media_x.
#   Equivalente matricial: beta = (X^T X)^-1 X^T y.
# Computacional: sklearn exige X 2D (n,1) -> por isso reshape(-1,1).
#   model.coef_ = b1 (inclinação: $ extras de gorjeta por $1 de conta).
#   model.intercept_ = b0 (gorjeta quando conta=0, só extrapolação).
#   model.score = R² = 1 - SS_res/SS_tot (fração da variância explicada).
# ---------------------------------------------------------------------------
# Define variables
X = df['total_bill'].values.reshape(-1, 1)  # matriz (244,1): sklearn exige 2D
y = df['tip'].values  # vetor (244,): variável alvo

# # Fit linear regression
model = LinearRegression()  # cria estimador OLS (sem regularização)
model.fit(X, y)  # resolve a equação normal internamente via SVD

# Output coefficients
print("Slope: ", model.coef_[0])  # b1 ~ 0.10 -> cada $1 de conta ≈ +$0.10 de gorjeta
print("Intercept: ", model.intercept_)  # b0 ~ 0.92
print("R-Squared: ", model.score(X, y))  # R² ~ 0.45 -> conta explica ~45% da gorjeta

# ---------------------------------------------------------------------------
# 8. PLOT DA REGRESSÃO
# scatterplot = nuvem (X,Y). plt.plot(X, y_pred) = reta ajustada.
# Resíduos (distância ponto-reta) são o erro não explicado pelo modelo.
# ---------------------------------------------------------------------------
sns.scatterplot(x=df['total_bill'], y=df['tip'], color="blue")
plt.plot(df['total_bill'], model.predict(X), color="red", label="Regression Line")
plt.title("Total Bill vs Tip")
plt.legend()
plt.show()
