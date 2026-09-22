"""
additional_practice1.py — Expansão do projeto Tips: Dia da Semana + originais.
===========================================================================
OBJETIVO: estender day7_project.py com um relacionamento categórico NOVO
(day vs tip) sem perder as análises originais (correlação, teste-T, regressão).

DATASET: tips.csv — total_bill, tip (numéricas) + sex, smoker, day, time (categóricas).
NOVIDADE: ANOVA de 1 fator (4 dias) + boxplot/barplot por dia.
"""

import matplotlib.pyplot as plt  # backend de figuras: Figure, Axes, show()
import numpy as np  # arrays, dtypes numéricos, usado no select_dtypes
import pandas as pd  # read_csv, boolean mask, DataFrame
import seaborn as sns  # boxplot/barplot/heatmap/scatterplot estatísticos
from scipy.stats import f_oneway, ttest_ind  # ANOVA (k>2 grupos) e Teste-T (2 grupos)
from sklearn.linear_model import LinearRegression  # regressão OLS

# ---------------------------------------------------------------------------
# 1. CARGA DOS DADOS
# Computacional: mesmo CSV do projeto. Mantemos TODAS as colunas de propósito
# (o projeto original deletava sex/smoker/day/time — aqui NÃO deletamos).
# ---------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# [CORREÇÃO]: Mantemos as colunas categóricas (sex, smoker, day, time) para a expansão do projeto
# Por quê? Sem elas não há como testar day vs tip nem sex vs tip.
# Lição computacional: filtrar só na hora da conta numérica (select_dtypes),
# em vez de destruir a coluna com `del`.

# ==========================================
# EXPANSAO: Dia da Semana vs. Gorjeta (tip)
# ==========================================
# Pergunta de negócio: "gorjeta muda conforme o dia?".
# Variável independente: categórica nominal com 4 níveis (Thur/Fri/Sat/Sun).
# Variável dependente: contínua (tip $).

print("--- Análise: Dia da Semana vs Gorjeta ---")

# ---------------------------------------------------------------------------
# Visualização 1: Boxplot por dia
# Matemática do boxplot (Tukey): para cada dia calcula Q1 (25%), mediana (50%),
# Q3 (75%), IQR = Q3-Q1, whiskers = [Q1-1.5*IQR, Q3+1.5*IQR], pontos fora = outliers.
# Computacional: sns.boxplot agrupa por x e desenha as 5 estatísticas + outliers.
# Leitura: se as caixas/medians estão em alturas bem diferentes, suspeita de efeito do dia.
# `order=[...]` fixa ordem cronológica (senão seria alfabética).
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="day", y="tip", palette="Set2", order=["Thur", "Fri", "Sat", "Sun"])
plt.title("Distribuição de Gorjetas por Dia da Semana")
plt.xlabel("Dia da Semana")
plt.ylabel("Gorjeta ($)")
plt.show()


# ---------------------------------------------------------------------------
# Visualização 2: Barplot da MÉDIA por dia
# Matemática: barra = média amostral mean = soma/n por grupo.
# `errorbar=None` desliga intervalo de confiança (por padrão seaborn mostra IC 95%
# via bootstrap). Aqui queremos só a média pontual para comparar dias.
# Diferença boxplot vs barplot: boxplot mostra distribuição inteira (robusto a
# outliers); barplot resume num número (sensível a outliers).
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="day", y="tip", palette="Set2", errorbar=None, order=["Thur", "Fri", "Sat", "Sun"])
plt.title("Média de Gorjeta por Dia da Semana")
plt.xlabel("Dia da Semana")
plt.ylabel("Média da Gorjeta ($)")
plt.show()

# ---------------------------------------------------------------------------
# Teste Estatístico ANOVA (Analysis of Variance) de 1 fator
# H0: mu_Thur = mu_Fri = mu_Sat = mu_Sun (dia não influencia).
# H1: pelo menos uma média difere.
# Matemática: decompõe variação total SST = SSB (entre grupos) + SSW (dentro).
#   MSB = SSB/(k-1), MSW = SSW/(N-k), onde k=4 grupos, N=244.
#   F = MSB/MSW. Sob H0, F ~ F(k-1, N-k). F grande = variação ENTRE dias
#   domina o ruído DENTRO dos dias = evidência de efeito real.
#   p = P(F >= f_obs | H0). Suposições: normalidade por grupo, homocedasticidade
#   (variâncias iguais) e independência. Robusta para N moderado.
# Computacional: f_oneway(*grupos) recebe 4 Series e retorna (F, p).
# Por que não 6 testes-T? Múltiplos testes inflam o Erro Tipo I; ANOVA controla
# isso num único teste global (depois far-se-ia post-hoc como Tukey).
# ---------------------------------------------------------------------------
thur_tips = df[df["day"] == "Thur"]["tip"]  # boolean mask: filtra só quinta
fri_tips = df[df["day"] == "Fri"]["tip"]
sat_tips = df[df["day"] == "Sat"]["tip"]
sun_tips = df[df["day"] == "Sun"]["tip"]

f_stat, p_value_anova = f_oneway(thur_tips, fri_tips, sat_tips, sun_tips)
print(f"ANOVA F-Statistic: {f_stat:.4f}")
print(f"ANOVA P-Value: {p_value_anova:.4f}")

alpha = 0.05  # nível de significância: 5% de chance de falso-positivo aceita
if p_value_anova <= alpha:
    print("Resultado: Rejeita a hipótese nula. O dia da semana INFLUENCIA significativamente a gorjeta.")
else:
    print("Resultado: Falha em rejeitar a hipótese nula. NÃO há diferença significativa entre os dias.")

# ==========================================
# ANÁLISES ORIGINAIS AJUSTADAS
# ==========================================

# ---------------------------------------------------------------------------
# Heatmap de Correlação (só numéricas)
# Matemática: Pearson r = cov(X,Y)/(sx*sy). Aqui matriz 3x3:
# total_bill, tip, size. Diagonal = 1. Esperado: total_bill-tip ~0.68.
# Computacional: select_dtypes(include=[np.number]) evita o erro que o projeto
# original contornava com `del` — solução elegante e não-destrutiva.
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
df_numeric = df.select_dtypes(include=[np.number])  # Garante selecionar apenas números
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap de Correlação (Variáveis Numéricas)")
plt.show()

# ---------------------------------------------------------------------------
# Teste-T por Gênero (2 grupos: Male vs Female)
# H0: mu_male = mu_female. Estatística t = (m1-m2)/sqrt(s1²/n1+s2²/n2).
# É o caso especial da ANOVA com k=2 (F = t²). Mantido do projeto original.
# ---------------------------------------------------------------------------
print("--- Análise: Gênero vs Gorjeta ---")
male_tips = df[df["sex"] == "Male"]["tip"]
female_tips = df[df["sex"] == "Female"]["tip"]

t_stat, p_value_t = ttest_ind(male_tips, female_tips)
print(f"T-Statistic:", t_stat)
print("P-Value:", p_value_t)

if p_value_t <= alpha:
    print("Resultado: Diferença significativa entre gêneros.")
else:
    print("Resultado: NÃO há diferença significativa entre gêneros.")
print("\n")

# ---------------------------------------------------------------------------
# Regressão Linear Simples: tip = b0 + b1*total_bill
# Matemática MQO: min soma(yi-b0-b1*xi)² -> b1 = cov/var, b0 = my-b1*mx.
# R² = 1 - SSres/SStot: 0 = nada explica, 1 = perfeito. Aqui ~0.45.
# Computacional: reshape(-1,1) pois sklearn exige matriz (n,1).
# ---------------------------------------------------------------------------
print("--- Regressão Linear: Conta vs Gorjeta ---")
X = df["total_bill"].values.reshape(-1, 1)
y = df["tip"].values

model = LinearRegression()
model.fit(X, y)

print("Slope (Inclinação): ", model.coef_[0])  # ~0.10 $tip por $conta
print("Intercept (Intercepto): ", model.intercept_)  # extrapolação p/ conta=0
print("R-Squared ((R²): ", model.score(X, y))

# ---------------------------------------------------------------------------
# Plot: nuvem + reta E[tip|conta]. Resíduo = yi - y_pred_i (erro do modelo).
# alpha=0.6 deixa pontos semi-transparentes (resolve overplotting).
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["total_bill"], y=df["tip"], color="blue", alpha=0.6)
plt.plot(df["total_bill"], model.predict(X), color="red", label="Linha de Regressão")
plt.title("Total da Conta vs Gorjeta")
plt.xlabel("Total da Conta ($)")
plt.ylabel("Gorjeta ($)")
plt.legend()
plt.show()
