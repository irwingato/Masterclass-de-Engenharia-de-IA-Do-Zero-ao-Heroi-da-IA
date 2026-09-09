# additional_practice_2.py
# ------------------------------------------------------------------
# Objetivo: Explorar VISUALIZAÇÕES AVANÇADAS com seaborn (boxplots,
# pairplot, histograma com KDE e heatmap de correlação) usando o
# dataset Iris.
# ------------------------------------------------------------------

# Importa as bibliotecas utilizadas:
# pandas  -> manipulação de dados
# matplotlib.pyplot -> criação de gráficos básicos
# seaborn -> biblioteca de visualização estatística (base usada aqui)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define o tema visual padrão do seaborn (fundo com grade branca)
sns.set_theme(style="whitegrid")

# Carrega o dataset Iris a partir de uma URL (CSV público no GitHub)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Boxplot 1: distribuição do comprimento da sépala por espécie
# O boxplot mostra mediana, quartis e outliers da variável.
# hue="species" colore cada espécie (e legend=False evita legenda duplicada)
sns.boxplot(data=df, x="species", y="sepal_length", hue="species", palette="Set2", legend=False)
plt.title("Sepal Length by Species")
plt.show()

# Boxplot 2: distribuição de TODAS as medidas por espécie
# melt() "derrete" o DataFrame do formato largo para longo (linha por medida),
# permitindo comparar todas as variáveis numéricas num único gráfico.
df_melted = df.melt(id_vars="species", var_name="measurement", value_name="value")
sns.boxplot(data=df_melted, x="measurement", y="value", hue="species", palette="Set2")
plt.title("Measurements by Species")
plt.show()

# Pairplot: matriz de dispersão entre todas as variáveis numéricas,
# colorida por espécie. Relevante para detectar correlações e separação
# das classes. diag_kind="kde" mostra a densidade (KDE) na diagonal.
sns.pairplot(df, hue="species", palette="Set2", diag_kind="kde")
plt.suptitle("Iris Pairplot", y=1.02)
plt.show()

# Histograma com curva de densidade (KDE): distribuição do comprimento
# da pétala, separada por espécie (hue), para comparar as distribuições.
sns.histplot(data=df, x="petal_length", hue="species", kde=True, palette="Set2", bins=15)
plt.title("Petal Length Distribution by Species")
plt.show()

# Heatmap de correlação: select_dtypes inclui apenas colunas numéricas,
# corr() calcula a matriz de correlação (Pearson) entre elas.
# annot=True exibe os valores dentro das células; fmt=".2f" formata com 2 casas.
corr = df.select_dtypes(include="number").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()