import pandas as pd
import numpy as np

# 1. Configurar semente para gerar sempre os mesmos resultados
# np.random.seed fixa a semente do gerador de números aleatórios,
# garantindo que os mesmos valores sejam produzidos a cada execução.
np.random.seed(42)

# 2. Criar dados simulados de vendas
n_vendas = 200  # Quantidade de registros/linhas a serem gerados
regioes = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']  # Lista de regiões possíveis
categorias = ['Alimentos', 'Eletrodomésticos', 'Eletrônicos', 'Móveis', 'Vestuário']  # Lista de categorias possíveis

dados = {
    # pd.date_range gera uma sequência de datas diárias a partir de 01/01/2026, totalizando n_vendas datas
    # .strftime('%Y-%m-%d') converte essas datas para o formato de texto AAAA-MM-DD
    'Data': pd.date_range(start='2026-01-01', periods=n_vendas, freq='D').strftime('%Y-%m-%d'),
    # np.random.choice sorteia aleatoriamente uma Regiao para cada um dos n_vendas registros
    'Regiao': np.random.choice(regioes, n_vendas),
    # np.random.choice sorteia aleatoriamente uma Categoria para cada registro
    'Categoria': np.random.choice(categorias, n_vendas),
    # np.random.randint gera valores inteiros aleatórios entre 100 (inclusive) e 5000 (exclusive)
    'Valor_Total': np.random.randint(100, 5000, n_vendas)
}

# Converte o dicionário em um DataFrame (tabela tabular) do pandas
df = pd.DataFrame(dados)

# 3. Agrupar e somar o faturamento por Região
# groupby('Regiao') agrupa as linhas por região; ['Valor_Total'].sum() soma os valores de cada grupo;
# reset_index() transforma a Regiao (índice do grupo) de volta em coluna, formando uma tabela final limpa.
vendas_por_regiao = df.groupby('Regiao')['Valor_Total'].sum().reset_index()

# 4. Agrupar e somar o faturamento por Categoria
# Mesmo princípio do passo 3, mas agora agrupando pela coluna Categoria.
vendas_por_categoria = df.groupby('Categoria')['Valor_Total'].sum().reset_index()

# Exibir os resultados consolidados
print("--- Faturamento por Região ---")
# to_string(index=False) imprime a tabela sem exibir a coluna de índice
print(vendas_por_regiao.to_string(index=False))

print("\n--- Faturamento por Categoria ---")
print(vendas_por_categoria.to_string(index=False))