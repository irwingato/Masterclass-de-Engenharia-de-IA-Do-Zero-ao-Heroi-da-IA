import pandas as pd  # importa a biblioteca pandas

# 1. Carregar os dados (usando o exemplo anterior)
# Cria um dicionário com produtos, preços e quantidades vendidas
dados = {
    "Produto": ["Notebook", "Mouse", "Teclado", "Monitor"],
    "Preco": [4500.00, 150.00, 250.00, 1200.00],
    "Quantidade_Vendida": [5, 20, 15, 8]
}
df = pd.DataFrame(dados)   # converte o dicionário em DataFrame (tabela)

# 2. Filtrar os dados (ex: apenas produtos com faturamento ou preço alto)
# Vamos filtrar produtos que custam mais de R$ 200,00
# Mantém somente as linhas onde o preço é maior que 200 e salva em df_filtrado
df_filtrado = df[df["Preco"] > 200.00]

# 3. Salvar o DataFrame filtrado em um novo arquivo CSV
# to_csv escreve o DataFrame filtrado em um arquivo chamado "produtos_caros.csv"
# index=False impede que o índice do DataFrame seja salvo como coluna extra
df_filtrado.to_csv("produtos_caros.csv", index=False)

print("Arquivo 'produtos_caros.csv' salvo com sucesso!")   # avisa que o arquivo foi criado