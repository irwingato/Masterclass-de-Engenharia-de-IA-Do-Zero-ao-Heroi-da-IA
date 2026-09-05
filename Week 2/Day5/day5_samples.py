
# Arquivo de referência com trechos de exemplo sobre groupby() e pivot_table().
# ATENÇÃO: este arquivo NÃO é executável sozinho. Ele supõe que já exista um DataFrame
# chamado `df` (ex.: criado com pd.DataFrame(...)) e que pandas tenha sido importado com `import pandas as pd`.

# groupby("columns_name") divide o DataFrame em grupos conforme os valores de uma coluna.
# O objeto `grouped` é um grupo de pares (nome_do_grupo, DataFrame_do_subgrupo).
grouped = df.groupby("columns_name")

# Iterar pelos grupos: para cada um, exibe o nome do grupo e as linhas que pertencem a ele.
for name, group in grouped:
    print(name)
    print(group)

# Aplica uma agregação (média ou soma) a todas as colunas numéricas de cada grupo.
grouped.mean()
grouped.sum()

# Calcula a média (mean) da coluna numérica "numeric_column" dentro de cada grupo da coluna categoria.
df.groupby("category_column")["numeric_column"].mean()
# O mesmo resultado, mas usando .agg(): permite aplicar várias agregações ao mesmo tempo
# (aqui: média, máximo e mínimo sobre a coluna numérica).
df.groupby("category_column").agg({"numeric_column": ["mean", "max", "min"]})

# pivot_table cria uma "tabela dinâmica": linhas = categorias, valores = média de "numeric_column".
pivot = df.pivot_table(
    values="numeric_column",
    index="category_column",
    aggfunc="mean"
)

# Função personalizada de agregação: calcula a amplitude (máximo - mínimo) de um grupo.
def range_func(x):
    return x.max() - x.min()

# Aplica a função personalizada aos valores numéricos de cada grupo de categoria.
df.groupby("category_column")["numeric_column"].agg(range_func)

# As três linhas abaixo demonstram aplicar, separadamente, mean/max/min por grupo.
df.groupby("category_column")["numeric_column"].mean()
df.groupby("category_column")["numeric_column"].max()
df.groupby("category_column")["numeric_column"].min()

# Mesmo resultado do exemplo acima, mas numa única chamada: várias agregações por coluna.
df.groupby("category_column").agg(
    {"numeric_column": ["mean", "max", "min"]}
)