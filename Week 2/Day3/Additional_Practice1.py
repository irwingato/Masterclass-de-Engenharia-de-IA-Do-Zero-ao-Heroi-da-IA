import pandas as pd  # importa a biblioteca pandas

# 1. Carregar o arquivo dos pinguins (ajuste o caminho se necessário)
# read_csv lê o arquivo CSV "penguins.csv" e armazena no DataFrame df
df = pd.read_csv("penguins.csv")

print("--- Primeiras 5 linhas ---")
print(df.head())   # mostra as 5 primeiras linhas

print("\n--- Últimas 3 linhas ---")
print(df.tail(3))  # mostra as 3 últimas linhas

print("\n--- Informações gerais do DataFrame ---")
print(df.info())   # mostra tipos de dados, colunas e valores não nulos

print("\n--- Resumo estatístico das colunas numéricas ---")
print(df.describe())   # mostra média, desvio padrão, min, max, etc.

# --- Selecionando Colunas ---
# Trocamos "Name" e "Age" por colunas reais dos pinguins (Espe´cie e Massa Corporal)
print("\n--- Apenas as colunas 'species' e 'body_mass_g' ---")
print(df[["species", "body_mass_g"]])   # seleciona e imprime essas duas colunas

# --- Filtrando Dados ---
# Trocamos o filtro de idade para selecionar pinguins com mais de 4500g (4.5kg)
print("\n --- Pinguins com massa corporal maior que 4500g ---")
print(df[df["body_mass_g"] > 4500])   # mantém somente as linhas com body_mass_g > 4500

# --- Seleção por Posição (iloc) ---
print("\n--- Primeira linha completa ---")
print(df.iloc[0])   # iloc seleciona por posição: imprime a primeira linha (índice 0)

print("\n--- Primeira coluna completa (todas as linhas da coluna 0) ---")
print(df.iloc[0])   # bônus: aqui também imprime a linha 0 (o comentário do original estava trocado)

# --- Exportando os Dados ---
# Caso queira salvar os filtros ou alterações no futuro:
# df.to_csv("novos_pinguins.csv", index=False)          # salva em CSV sem o índice
# df.to_excel("novos_pinguins.xlsx", index=False)       # salva em Excel sem o índice