import pandas as pd
import numpy as np

# 1. Configurar semente para consistência nos resultados
# np.random.seed fixa a semente do gerador de números aleatórios,
# garantindo que os mesmos dados sejam gerados a cada execução.
np.random.seed(42)

# 2. Criar dados simulados de vendas contendo múltiplos anos
n_vendas = 300  # Quantidade de registros a serem gerados
regioes = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']  # Regiões possíveis

dados = {
    # pd.date_range gera 300 datas igualmente distribuídas entre 01/01/2024 e 31/12/2026
    'Data': pd.date_range(start='2024-01-01', end='2026-12-31', periods=n_vendas),
    # np.random.choice sorteia aleatoriamente uma Regiao para cada registro
    'Regiao': np.random.choice(regioes, n_vendas),
    # np.random.randint gera valores inteiros de 100 a 4999 como Valor_Total de cada venda
    'Valor_Total': np.random.randint(100, 5000, n_vendas)
}

# Converte o dicionário em um DataFrame (tabela tabular) do pandas
df = pd.DataFrame(dados)

# 3. Extrair o Ano da coluna de Data para usar no pivot_table
# df['Data'].dt.year acessa o componente "ano" de cada data da série;
# o resultado é armazenado em uma nova coluna chamada 'Ano'.
df['Ano'] = df['Data'].dt.year

# 4. Criar o pivot_table calculando a soma das vendas
# A tabela dinâmica usa: linhas = Regiao, colunas = Ano e células = soma de Valor_Total.
# aggfunc='sum' define a agregação a ser aplicada;
# fill_value=0 substitui por 0 as combinações região/ano sem nenhuma venda.
tabela_dinamica = df.pivot_table(
    values='Valor_Total',
    index='Regiao',
    columns='Ano',
    aggfunc='sum',
    fill_value=0 # Substitui anos sem vendas por 0, caso ocorra
)

# Exibir o resultado formatado
print("--- Faturamento Total por Região e Ano ---")
print(tabela_dinamica)