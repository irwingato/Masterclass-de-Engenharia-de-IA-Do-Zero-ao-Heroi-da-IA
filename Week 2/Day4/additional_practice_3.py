# Importa o pandas, biblioteca usada para trabalhar com dados tabulares (DataFrames)
import pandas as pd

# 1. Criando um DataFrame de exemplo com dados categóricos
df = pd.DataFrame({
    # Chave única de cada cliente (identificador numérico)
    'id_cliente': [1, 2, 3, 4],
    # Coluna categórica de texto: tipo de plano contratado por cada cliente
    'plano': ['Premium', 'Básico', 'Premium', 'Gratuito'],
    # Coluna categórica de texto: região de cada cliente
    'regiao': ['Norte', 'Sul', 'Sul', 'Centro']
})

# Exibe a tabela original com os dados categóricos em formato de texto
print("--- DataFrame Original ---")
print(df)

# 2. Aplicando a One-Hot Encoding
# O Pandas identifica automaticamente as colunas de texto e as transforma
# get_dummies() converte cada CATEGORIA de 'plano' e 'regiao' em uma coluna separada
# com valor 1 (cliente pertence à categoria) ou 0 (não pertence).
# dtype=int garante que os 0s e 1s sejam inteiros.
df_codificado = pd.get_dummies(df, columns=['plano', 'regiao'], dtype=int)
# Obs.: a coluna original 'plano'/'regiao' é substituída por N colunas binárias,
# e a coluna 'id_cliente' permanece intacta por ser numérica.

# Exibe o DataFrame após o One-Hot Encoding (variáveis binárias 0/1)
print("\n--- DataFrame com One-Hot Enconding ---")
print(df_codificado)