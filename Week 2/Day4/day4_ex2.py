# Importa as bibliotecas pandas (tabelas/DataFrames) e numpy (cálculo numérico)
import pandas as pd
import numpy as np

# Primeiro DataFrame com dados de identificação e idade dos alunos
df1 = pd.DataFrame({
    "ID": [1,2,3],             # Coluna de chave identificadora (vai ser usada para o merge)
    "Name": ["Alice", "Bob", "Charlie"],  # Nomes dos alunos
    "Age": [25, 30, 35],       # Idades dos alunos
})

# Segundo DataFrame com as notas dos mesmos alunos
df2 = pd.DataFrame({
    "ID": [1,2,3],             # Mesma chave "ID" para poder combinar com df1
    "Score": [85, 90, 88]      # Notas de cada aluno
})

# Exibe os dois datasets separadamente
print("Dataset 1: \n", df1)
print("Dataset 2: \n", df2)

# Combina os dois DataFrames usando a coluna "ID" como chave
# how="inner" mantém apenas as linhas cujo ID existe nos dois datasets
merged = pd.merge(df1, df2, how="inner", on="ID")
# Exibe o resultado da junção
print("Merged Dataset: \n", merged)

# Cria uma nova coluna "Score_Percentage" transformando a nota
# em porcentagem: divide por 200 (nota máxima) e multiplica por 100
merged["Score_Percentage"] = (merged["Score"] / 200) * 100
# Exibe o dataset final com a coluna calculada
print("Transformed Dataset: \n", merged)