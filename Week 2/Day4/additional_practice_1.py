# Importa o pandas, biblioteca usada para trabalhar com dados tabulares (DataFrames)
import pandas as pd

# 1.  Cria uma tabela (DataFrame) de exemplo
dados = {
    # Coluna "Nome": nomes dos 4 clientes (todos preenchidos)
    'Nome': ['Ana', 'Bruno', 'Carlos', 'Diego'],
    # Coluna "Idade": idades, com 1 valor ausente (None) em 4 -> 25% de faltantes (será mantida)
    'Idade': [25, None, 30, 22],                # 25% de faltantes (será mantida)
    # Coluna "Cidade": apenas 1 preenchida, 3 ausentes -> 75% de faltantes (será removida)
    'Cidade': [None, None, 'São Paulo', None],  # 75% de faltantes (será removida)
    # Coluna "Salário": salários, com 1 valor ausente -> 25% de faltantes (será mantida)
    'Salário': [5000, 6000, None, 8000]         # 25% de faltantes (será mantida)
}

# Converte o dicionário em um DataFrame (tabela estruturada do pandas)
df = pd.DataFrame(dados)
# Exibe a tabela original com todos os valores ausentes (None)
print("--- Tabela Original ---")
print(df)

# 2. Identifica e remove colunas com mais de 50% de valores faltantes
# Calcula o "thresh": número mínimo de valores NÃO ausentes que cada coluna deve ter
# para ser mantida (aqui: metade do total de linhas = 4 * 0.5 = 2)
limite = len(df) * 0.5
# dropna com axis=1 remove COLUNAS; o parâmetro thresh garante que só sejam removidas
# aquelas com menos de "limite" valores válidos (no caso, a coluna "Cidade" com só 1 valor válido)
df_limpo = df.dropna(thresh=limite, axis=1)

# Exibe a tabela já limpa, mostrando que só a coluna "Cidade" foi removida
print("\n--- Tabela Após o Drop (Coluna 'Cidade' removida) ---")
print(df_limpo)