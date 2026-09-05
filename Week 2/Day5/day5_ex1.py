import pandas as pd

# Dicionário com os dados crus (em formato de listas por coluna).
data = {
    "Class": ["A", "B", "A", "B", "C", "C"],
    "Score": [85, 90, 88, 72, 95, 80],
    "Age": [15, 16, 15, 17, 16, 15]
}

# Converte o dicionário em um DataFrame (tabela tabular) do pandas.
df = pd.DataFrame(data)

# Exibe o conjunto de dados original.
print("Original Dataset \n", df)

# groupby("Class") agrupa as linhas por turma e .mean() calcula a média
# das colunas numéricas (Score e Age) dentro de cada grupo.
grouped = df.groupby("Class").mean()
# print(grouped)

# .agg() permite aplicar várias agregações de uma vez: aqui calcula a média,
# o máximo e o mínimo tanto de Score quanto de Age, para cada classe.
stats = df.groupby("Class").agg(
    {"Score": ["mean", "max", "min"], "Age": ["mean", "max", "min"]}
)

# Exibe a tabela de estatísticas com o cabeçalho em múltiplos níveis (coluna + agregação).
print(stats)