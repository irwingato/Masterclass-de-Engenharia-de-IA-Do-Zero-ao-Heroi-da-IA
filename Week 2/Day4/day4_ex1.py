# Importa a biblioteca pandas, usada para trabalhar com dados tabulares (DataFrames)
import pandas as pd
# Importa a biblioteca numpy, usada para cálculos numéricos (aqui fornece np.nan = valor ausente)
import numpy as np

# Criação de um dataset de exemplo
data = {
    # Coluna "Name": nomes dos alunos, com um valor ausente (NaN) em 1 posição
    "Name": ["Alice", "Bob", np.nan, "David"],
    # Coluna "Age": idades, com um valor ausente (NaN) na 2ª posição
    "Age": [25, np.nan, 30, 35],
    # Coluna "Score": notas, com um valor ausente (NaN) na 3ª posição
    "Score": [85, 90, np.nan, 88]
}
# Converte o dicionário em um DataFrame (tabela estruturada do pandas)
df = pd.DataFrame(data)

# Exibe o dataset original, mostrando os valores ausentes (NaN)
print("Original Dataset: \n", df)

# Preenche os valores ausentes da coluna "Age" com a média das idades existentes (.mean())
df["Age"] = df["Age"].fillna(df["Age"].mean())
# Preenche os valores ausentes da coluna "Score" por interpolação linear
# (estima o valor com base nos vizinhos acima e abaixo do NaN)
df["Score"] = df["Score"].interpolate()

# Exibe o dataset depois da limpeza dos dados ausentes
print("Dataset: \n", df)

# Renomeia as colunas: "Name" vira "Student_Name" e "Score" vira "Exam:Score"
df = df.rename(columns={"Name":"Student_Name", "Score": "Exam:Score"})
# Exibe o dataset com os novos nomes de colunas
print("Dataset: \n", df)