# URL de onde o dataset Iris será baixado (fonte: repositório seaborn-data no GitHub)

import pandas as pd  # importa a biblioteca pandas (usada para manipular dados tabulares)

# Load Dataset
# read_csv lê o arquivo CSV da URL e carrega na variável df (DataFrame)
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Explore structure (as linhas abaixo estão comentadas, mas serviam para explorar os dados)
# print("First 5 rows: \n", df.head())    # mostra as 5 primeiras linhas
# print("Last 5 rows: \n", df.tail())     # mostra as 5 últimas linhas
# print(df.info())                        # mostra informações gerais (tipos, valores não nulos)
# print(df.describe())                    # mostra resumo estatístico das colunas numéricas

# Seleciona apenas as colunas "species" e "sepal_length" e guarda na variável
selected_columns = df[["species", "sepal_length"]]
# print("Selected Columns: \n", selected_columns)   # (comentado) imprimiria as colunas selecionadas

# Filtra as linhas: mantém apenas as flores com sepal_length > 5.0 E espécie igual a "setosa"
filtered_rows = df[(df["sepal_length"] > 5.0) & (df["species"] == "setosa")]
# Imprime as linhas filtradas na tela
print("Filtered Rows: \n", filtered_rows)