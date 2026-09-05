import pandas as pd
import numpy as np

# 1. Configurar semente para gerar sempre os mesmos resultados
np.random.seed(42)

# 2. Criar dados simulados de vendas
n_vendas = 200  # Quantidade de linhas/registros a serem gerados
regioes = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']  # Regiões possíveis
categorias = ['Alimentos', 'Eletrodomésticos', 'Eletrônicos', 'Móveis', 'Vestuário']  # Categorias possíveis

dados = {
    # pd.date_range gera uma sequência de datas; aqui 200 datas diárias a partir de 2026-01-01
    # .strftime('%Y-%m-%d') converte as datas para o formato de texto AAAA-MM-DD
    'Data': pd.date_range(start='2026-01-01', periods=n_vendas, freq='D').strftime('%Y-%m-%d'),
    # np.random.choice sorteia aleatoriamente uma Regiao para cada uma das n_vendas linhas
    'Regiao': np.random.choice(regioes, size=n_vendas),
    # np.random.choice sorteia aleatoriamente uma Categoria para cada linha
    'Categoria': np.random.choice(categorias, size=n_vendas),
    # np.random.randint gera valores inteiros aleatórios entre 100 e 4999 para o Valor_Total
    'Valor_Total': np.random.randint(100, 5000, n_vendas)
}

# Converte o dicionário em um DataFrame (tabela) do pandas
df = pd.DataFrame(dados)

# 3. Agrupar e somar o faturamento por Região
# groupby('Regiao') agrupa as linhas por região; ['Valor_Total'].sum() soma os valores de cada grupo
# reset_index() transforma o índice (Regiao) em uma coluna novamente, gerando uma tabela limpa
vendas_por_regiao = df.groupby('Regiao')['Valor_Total'].sum().reset_index()

# 4. Agrupar e somar o faturamento por Categoria (mesmo princípio do passo 3, agora por Categoria)
vendas_por_categoria = df.groupby('Categoria')['Valor_Total'].sum().reset_index()

# Exibir os resultados consolidados
print("--- Faturamento por Região ---")
# to_string(index=False) exibe a tabela sem a coluna de índice
print(vendas_por_regiao.to_string(index=False))

print("\n--- Faturamento por Categoria ---")
print(vendas_por_categoria.to_string(index=False))