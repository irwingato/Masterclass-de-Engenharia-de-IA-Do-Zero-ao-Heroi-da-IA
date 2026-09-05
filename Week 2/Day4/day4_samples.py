

# Exemplos de referência de operações do pandas (não executáveis sozinhos,
# pois dependem de df1, df2 previamente definidos)

# Junta dois DataFrames empilhando as linhas (axis=0: um embaixo do outro)
combined = pd.concat([df1, df2], axis=0)
# Junta dois DataFrames lado a lado, colocando as colunas uma ao lado da outra (axis=1)
combined = pd.concat([df1, df2], axis=1)

# Faz o merge (junção por coluna) pela chave "common_column"
# Sem indicar "how", o padrão é "inner" (só linhas que existem nos dois)
merged= pd.merge(df1, df2, on="common_column")
# how="left": mantém todas as linhas do df1 (dataset da esquerda),
# preenchendo com NaN o que não existir no df2
merged = pd.merge(df1, df2, how="left", on="common_column")
# how="inner": mantém apenas as linhas cuja chave existe nos DOIS datasets
merged = pd.merge(df1, df2, how="inner", on="common_column")


# .join() é uma junção pelo ÍNDICE dos DataFrames (não por coluna),
# aqui utilizando junção "inner" (só índices presentes nos dois)
joined = df1.join(df2, how="inner")

# Renomeia uma coluna existente: de "old_name" para "new_name"
# df = df.rename(columns={"old_name": "new_name"})

# Converte o tipo dos dados de uma coluna, ex.: para float (número decimal)
# df["column_name"] = df["columns_name"].astype("float")
# Converte uma coluna de texto para o tipo data/hora (datetime)
# df["column_name"] = pd.to_datetime(df["column_name"])

# Cria uma nova coluna a partir de um cálculo com uma coluna existente (ex.: x 2)
# df["new_column"] = df["existing_column"] * 2 

# Remove linhas que tenham qualquer valor ausente (NaN)
# df = df.dropna()
# Remove colunas que tenham qualquer valor ausente (axis=1)
# df = df.dropna(axis=1)

# Preenche os valores ausentes (NaN) de uma coluna com um valor fixo (0)
# df["column_name"] = df["column_name"].fillna(0)

# Preenche valores ausentes propagando o último valor válido (forward fill)
# df.fillna(method="ffill")
# Preenche valores ausentes usando o próximo valor válido (backward fill)
# df.fillna(method="bfill")

# Seleção de colunas/linhas de um DataFrame (ex.: df["coluna"], df.loc[...], df.iloc[...])
# df[]

# Versões por coluna (axis=1) do preenchimento:
# fica com o valor mais à esquerda válido (ffill) e
# df.fillna(method="ffill", axis=1)
# fica com o valor mais à direita válido (bfill)
# df.fillna(method="bfill", axis=1)