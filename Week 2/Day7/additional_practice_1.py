# additional_practice_1.py
# ------------------------------------------------------------------
# Objetivo: Prática extra de EDA. Reproduz a mesma estrutura do
# day7_ex.py (inspeção, limpeza, filtro e visualizações), porém
# aplicada a OUTRO conjunto de dados: o dataset Iris (flores de
# três espécies: setosa, versicolor e virginica).
# ------------------------------------------------------------------

# Importa as bibliotecas utilizadas:
# pandas  -> manipulação de dados
# matplotlib.pyplot -> criação de gráficos básicos
# seaborn -> visualização estatística
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega o dataset Iris a partir de uma URL (CSV público no GitHub)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Inspeção inicial dos dados:
# info()    -> tipos das colunas e quantidade de valores não-nulos
# describe()-> estatísticas descritivas das colunas numéricas
print(df.info())
print(df.describe())

# Tratamento de valores ausentes (missing values):
# "sepal_length" -> preenche com a MEDIANA
# "species"      -> preenche com a MODA (valor mais frequente)
df["sepal_length"] = df["sepal_length"].fillna(df["sepal_length"].median())
df["species"] = df["species"].fillna(df["species"].mode()[0])

# Remove linhas duplicadas do DataFrame
df = df.drop_duplicates()

# Filtragem: seleciona apenas as flores da espécie "setosa"
setosa = df[df["species"] == "setosa"]
print("Setosa samples: \n", setosa.head())

# Gráfico de barras: quantidade de amostras por espécie
# value_counts() conta quantas vezes cada espécie aparece no dataset
species_count = df["species"].value_counts()
species_count.plot(kind="bar", color="skyblue")
plt.title("Samples per Species")
plt.ylabel("Count")
plt.show()

# Histograma da distribuição do comprimento da sépala (fica comentado)
# sns.histplot(df["sepal_length"], kde=True, bins=20, color="purple")
# plt.title("Sepal Length Distribution")
# plt.xlabel("Sepal Length")
# plt.ylabel("Frequency")
# plt.show()

# Gráfico de dispersão: relação entre comprimento da sépala e da pétala
# alpha=0.5 dá transparência aos pontos para visualizar sobreposições
plt.scatter(df["sepal_length"], df["petal_length"], alpha=0.5, color="green")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()