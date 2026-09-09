# day7_ex.py
# ------------------------------------------------------------------
# Objetivo: Realizar uma Análise Exploratória de Dados (EDA) no
# dataset Titanic, cobrindo inspeção, limpeza, filtragem e algumas
# visualizações.
# ------------------------------------------------------------------

# Importa as bibliotecas utilizadas no script:
# pandas  -> manipulação de dados (DataFrames, leitura de CSV, estatísticas)
# matplotlib.pyplot -> criação de gráficos básicos
# seaborn -> biblioteca de visualização estatística (gráficos avançados)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega o dataset Titanic a partir de uma URL (CSV público no GitHub)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Inspeção inicial dos dados:
# info()    -> mostra tipos das colunas, quantidade de valores não-nulos e memória usada
# describe()-> mostra estatísticas descritivas (média, desvio, min/max, quartis)
print(df.info())
print(df.describe())

# Tratamento de valores ausentes (missing values):
# "Age"      -> preenche com a MEDIANA (robusto a outliers na idade)
# "Embarked" -> preenche com a MODA (valor mais frequente: porto de embarque)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove linhas duplicadas do DataFrame
df = df.drop_duplicates()

# Filtragem: seleciona apenas passageiros da 1ª classe (Pclass == 1)
first_class = df[df["Pclass"] == 1]
print("First Class Passengers: \n", first_class.head())

# Gráfico de barras: taxa de sobrevivência agrupada por classe
# groupby("Pclass")["Survived"].mean() calcula a proporção de sobreviventes
# em cada classe (0 a 1)
survival_by_class = df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind="bar", color="skyblue")
plt.title("Survival Rate by Class")
plt.ylabel("Survival Rate")
plt.show()

# Histograma da distribuição de idades (fica comentado, pode ser reativado)
# sns.histplot(df["Age"], kde=True, bins=20, color="purple")
# plt.title("Age Distribution")
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.show()

# Gráfico de dispersão (scatter): relação entre Idade e Tarifa paga
# alpha=0.5 dá transparência aos pontos para visualizar sobreposições
plt.scatter(df["Age"], df["Fare"], alpha=0.5, color="green")
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()