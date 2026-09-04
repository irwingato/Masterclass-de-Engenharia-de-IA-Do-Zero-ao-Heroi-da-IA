import pandas as pd  # importa a biblioteca pandas

# Cria uma Series (estrutura 1D) com os valores 10, 20, 30 e índices a, b, c
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
#print(s)  # (comentado) imprimiria a Series

# data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}   # dicionário com dados
# df = pd.DataFrame(data)                              # transforma em DataFrame (tabela)
# print(df)                                            # imprime a tabela

# Viewing Data
print(df.head())          # mostra as 5 primeiras linhas do DataFrame
print(df.tail(3))         # mostra as 3 últimas linhas

print(df.info())          # mostra informações gerais (tipos de dados, valores não nulos)
print(df.describe())      # mostra resumo estatístico das colunas numéricas

print(df[["Name", "Age"]])   # seleciona e imprime apenas as colunas "Name" e "Age"

print(df[df["Age"] > 25])    # filtra e imprime as linhas onde a idade é maior que 25

print(df.iloc[0])        # iloc seleciona por posição: primeira LINHA (índice 0)
print(df.iloc[:, 0])     # primeiro parâmetro = linhas, segundo = colunas; aqui: todas as linhas, primeira COLUNA

print(df.loc[0])         # loc seleciona por rótulo: linha com rótulo 0
print(df.loc[:, "Name"]) # todas as linhas da coluna com rótulo "Name"

# df = pd.read_csv("data.csv")        # (comentado) leria um arquivo CSV
# df.to_csv("data.csv", index=False)  # (comentado) salvaria em CSV sem o índice
# df.to_excel("data.xlsx", index=False) # (comentado) salvaria em Excel sem o índice