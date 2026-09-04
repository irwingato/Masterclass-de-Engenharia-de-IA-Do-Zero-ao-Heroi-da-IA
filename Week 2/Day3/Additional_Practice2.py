import pandas as pd  # importa a biblioteca pandas

# 1. Criar o dicionário com os dados
# Um dicionário onde cada chave vira uma coluna e cada lista vira os valores dessa coluna
dados = {
    "Produto": ["Notebook", "Mouse", "Teclado", "Monitor"],   # coluna com nomes de produtos
    "Preco": [4500.00, 150.00, 250.00, 1200.00],             # coluna com preços
    "Quantidade_Vendida": [5, 20, 15, 8]                     # coluna com quantidades vendidas
}

# 2. Criar o DataFrame a partir do dicionário
# Converte o dicionário em um DataFrame (tabela) chamado df
df = pd.DataFrame(dados)

# 3. Adicionar a nova coluna calculada (Preco multiplicado por Quantidade_Vendida)
# Cria a coluna "Faturamento_Total" multiplicando os valores das colunas "Preco" e "Quantidade_Vendida"
df["Faturamento_Total"] = df["Preco"] * df["Quantidade_Vendida"]

# Visualizar o resultado
print(df)   # imprime a tabela completa com a nova coluna