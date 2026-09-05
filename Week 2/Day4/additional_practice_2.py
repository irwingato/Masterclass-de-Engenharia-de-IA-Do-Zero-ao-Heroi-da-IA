# Importa o pandas, biblioteca usada para trabalhar com dados tabulares (DataFrames)
import pandas as pd


# 1. CRIANDO OS TRÊS CONJUNTOS DE DADOS DE EXEMPLO
# df_clientes: dados cadastrais de cada cliente, identificado pela coluna-chave 'id_cliente'
df_clientes = pd.DataFrame({
    'id_cliente': [1, 2, 3, 4, 5],          # Chave única de cada cliente (será usada no merge)
    'idade': [25, 40, 35, 50, 22],          # Idade de cada cliente
    'renda_mensal': [3000, 7500, 5000, 11000, 2500]  # Renda mensal de cada cliente
})

# df_vendas: valor total gasto por cada cliente (referenciado pelo mesmo 'id_cliente')
df_vendas = pd.DataFrame({
    'id_cliente': [1, 2, 3, 4, 5],          # Chave que conecta ao df_clientes
    'total_gasto': [150, 600, 350, 1200, 90] # Total gasto em vendas por cliente
})

# df_satisfacao: nota de satisfação de cada cliente (referenciado pelo mesmo 'id_cliente')
df_satisfacao = pd.DataFrame({
    'id_cliente': [1, 2, 3, 4, 5],          # Chave que conecta aos outros DataFrames
    'nota_satisfacao': [4.5, 3.8, 4.2, 4.9, 2.5]  # Nota de 0 a 5 dada pelo cliente
})

# 2. MESCLANDO OS TRÊS CONJUNTOS (Apenas com Pandas)
# .merge() combina tabelas pela coluna-chave 'id_cliente' (junção tipo "inner" por padrão).
# O resultado é encadeado: primeiro junta clientes+vendas e depois acrescenta satisfação.
df_final = df_clientes.merge(df_vendas, on='id_cliente').merge(df_satisfacao, on='id_cliente')

# 3. ANALISANDO OS RELACIONAMENTOS APENAS COM PANDAS

# Abordagem A: Matriz de Correlação Numérica
# (Mostra a força da relação entre todas as variáveis de uma vez)
print("--- Matriz de Correlação ---")
# Lista com as colunas numéricas que queremos analisar (exclui a chave 'id_cliente')
colunas_numericas = ['idade', 'renda_mensal', 'total_gasto', 'nota_satisfacao']
# .corr() calcula a correlação de Pearson entre cada par de colunas numéricas.
# Valores de 1 (correlação perfeita positiva) a -1 (negativa); 0 = sem relação.
print(df_final[colunas_numericas].corr())

# Abordagem B: Agrupamento por Faixas (Análise Direta)
# (Exemplo: Clientes com renda acima da média gastam mais ou estão mais satisfeitos?)
print("\n--- Análise de Renda Acima vs Abaixo da Média ---")
# Calcula a renda média de todos os clientes
renda_media = df_final['renda_mensal'].mean()
# Cria a coluna 'perfil_renda' aplicando uma função linha por linha:
# se a renda do cliente for maior que a média vira 'Alta', senão 'Baixa'
df_final['perfil_renda'] = df_final['renda_mensal'].apply(lambda x: 'Alta' if x > renda_media else 'Baixa')

# .groupby('perfil_renda') agrupa as linhas por 'Alta'/'Baixa' e, para cada grupo,
# .mean() calcula a média das colunas 'total_gasto' e 'nota_satisfacao'
analise_perfil = df_final.groupby('perfil_renda')[['total_gasto', 'nota_satisfacao']].mean()
# Exibe a comparação entre os dois grupos
print(analise_perfil)