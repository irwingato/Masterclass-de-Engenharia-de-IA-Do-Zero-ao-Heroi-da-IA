# Define a função que escreve uma lista de itens em um arquivo, um por linha.
def write_iteam_to_file(filename, items):
    # Abre o arquivo em modo escrita ("w"), criando-o ou sobrescrevendo-o,
    # e o fecha automaticamente ao sair do bloco.
    with open(filename, "w") as file:
        # Percorre cada item da lista.
        for item in items:
            # Escreve o item seguido de uma quebra de linha "\n".
            file.write(item + "\n")

# Define a função que lê e exibe os itens armazenados em um arquivo.
def read_items_from_file(filename):
    # Bloco try/except para tratar o caso de o arquivo não existir.
    try:
        # Abre o arquivo em modo leitura ("r").
        with open(filename, "r") as file:
            # Lê todas as linhas do arquivo e as armazena em uma lista.
            items = file.readlines()
            # Exibe um cabeçalho.
            print("Items in the file:")
            # Percorre cada linha lida.
            for item in items:
                # Remove espaços/quebras de linha das pontas e imprime o item.
                print(item.strip())
    except FileNotFoundError:
        # Caso o arquivo não seja encontrado, exibe uma mensagem de erro.
        print(f"File {filename} not found!")

# Lista de frutas que será gravada no arquivo.
fruits = ["Apple", "Banana", "Cherry", "Dates"]
# Chama a função para escrever as frutas no arquivo "fruits.txt".
write_iteam_to_file("fruits.txt", fruits)
# Chama a função para ler e exibir o conteúdo de "fruits.txt".
read_items_from_file("fruits.txt")