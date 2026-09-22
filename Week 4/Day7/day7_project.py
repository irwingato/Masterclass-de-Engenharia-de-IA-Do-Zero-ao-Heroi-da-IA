"""
day7_project.py — Projeto do Dia 7 (GABARITO COMENTADO): Tips completos.
========================================================================
STATUS: arquivo 100% comentado = serve como roteiro/template.
A versão executável e expandida está em additional_practice1.py.

DATASET: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv
PIPELINE PROPOSTO:
  1) Load + info()/describe() — inspeção
  2) del colunas categóricas (abordagem antiga) — limpeza
  3) histplot + heatmap — visualização
  4) ttest_ind por gênero — inferência (2 grupos)
  5) LinearRegression total_bill -> tip — modelagem

LEITURA MATEMÁTICA DO PROJETO: ele encadeia estatística descritiva
(média/desvio), visual (densidade/correlação), inferencial (teste de
hipótese) e preditiva (regressão). É o "ciclo completo" de um mini-projeto
de Data Science.
"""

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# ACIMA: stack padrão. pandas=tabela, numpy=vetores/CPU, seaborn+matplotlib=
# gráficos, sklearn=ML. Conceito computacional: cada lib opera num nível:
# numpy (array C rápido) < pandas (index + rótulos) < seaborn/sklearn (API de alto nível).

# # Load Dataset
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
# df = pd.read_csv(url)
# COMPUTACIONAL: pd.read_csv faz request HTTP GET, parseia vírgulas, infere
# dtype. Custo O(n*m). Falha se URL/raw errado (ver additional_practice3.py).

# # Inspect Data
# print(df.info())
# print(df.describe())
# MATEMÁTICA: describe() calcula momentos: média mu = soma(xi)/n,
# variância s² = soma(xi-mu)²/(n-1), desvio s = sqrt(s²), quartis Q1/Q2/Q3
# (inversão da CDF empírica). info() é computacional: ocupação de memória.

# del df["sex"]
# del df["smoker"]
# del df["day"]
# del df["time"]
# HISTÓRICO/COMPUTACIONAL: deletar strings era o truque para df.corr() não
# quebrar em pandas antigo (corr só aceita número). Hoje é melhor filtrar:
# df.select_dtypes(include=[np.number]). Deletar perde informação — por isso
# o practice1 CORRIGIU isso e manteve as categóricas para ANOVA/teste-T.

# # Visualize Distributions
# # sns.histplot(df["total_bill"], kde=True)
# # plt.title("Distribution of Total Bill")
# # plt.show()
# MATEMÁTICA: histograma = estimador de densidade por bins; KDE suaviza com
# kernel gaussiano. Responde: "a conta é simétrica ou com cauda de ricos?".

# # Correlation heatmap
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()
# MATEMÁTICA: r de Pearson = cov(X,Y)/(sx*sy). Matriz simétrica com 1 na
# diagonal. cmap coolwarm: azul=negativa, vermelho=positiva.

# from scipy.stats import ttest_ind

# # Separate data by gender
# male_tips = df[df['sex'] == 'Male']['tip']
# female_tips = df[df['sex'] == 'Female']['tip']
# COMPUTACIONAL: boolean masking — df[condição] filtra linhas em O(n).
# Gera duas Series independentes para comparar.

# # Perform t-test
# t_stat, p_value = ttest_ind(male_tips, female_tips)
# print("T-Statistic:", t_stat)
# print("P-Value:", p_value)
# MATEMÁTICA: t = (m1-m2)/EP, EP = sqrt(s1²/n1 + s2²/n2).
# Sob H0 (médias iguais), t ~ Student. p = P(|T|>=|t|).

# # Interpret results
# alpha = 0.05
# if p_value <= alpha:
#     print("Reject all null hypothesis: Significant difference.")
# else:
#     print("Fail to Reject the null hypothesis: NO significant difference.")
# LÓGICA: alpha = taxa tolerada de falso-positivo (Erro Tipo I).
# p pequeno = dados raros sob H0 = evidência contra H0.

# Define variables
# X = df['total_bill'].values.reshape(-1, 1)
# y = df['tip'].values
# COMPUTACIONAL: .values extrai ndarray; reshape(-1,1) vira matriz coluna
# porque sklearn segue convenção (n_samples, n_features).

# Fit linear regression
# model = LinearRegression()
# model.fit(X, y)
# MATEMÁTICA: minimiza soma dos quadrados S(b0,b1)=soma(yi-b0-b1*xi)².
# Condição de 1ª ordem -> equações normais -> b1=cov/var, b0=my-b1*mx.
# Complexidade O(n) para 1 feature; O(n*p²) no caso múltiplo.

# Output coefficients
# print("Slope: ", model.coef_[0])
# print("Intercept: ", model.intercept_)
# print("R-Squared: ", model.score(X, y))
# INTERPRETAÇÃO: slope ~0.10 ($/gorjeta por $/conta), intercept ~0.9,
# R² = 1-SSres/SStot ~0.45. Ou seja, modelo linear simples explica menos
# da metade da variância — gorjeta tem forte componente comportamental.

# Plot regression
# sns.scatterplot(x=df['total_bill'], y=df['tip'], color="blue")
# plt.plot(df['total_bill'], model.predict(X), color="red", label="Regression Line")
# plt.title("Total Bill vs Tip")
# plt.legend()
# plt.show()
# VISUAL: pontos = dados; reta = E[tip|conta]. Distância vertical = resíduo.
# Padrão em leque indicaria heterocedasticidade (variância cresce com a conta).
