# Abre o arquivo "sample.text" no modo leitura ("r") e cria um gerenciador de contexto.
# O "with" garante que o arquivo seja fechado automaticamente ao sair do bloco.
with open("sample.text", "r") as file:
    # Lê apenas a primeira linha do arquivo (comentado, não é executado).
    #content = file.readline()
    # Imprime o conteúdo lido da primeira linha (comentado, não é executado).
    #print(content)
    # Escreve o texto "Hello World!" no arquivo (observação: em modo "r" isso gera erro).
    file.write("Hello World!")
    # Escreve múltiplas strings de uma lista no arquivo (também falha em modo "r").
    file.writelines(["Alice", "Bob", "Cherry"])

# O arquivo é fechado automaticamente ao encerrar o bloco "with".

# Bloco try/except para tratar possíveis erros ao abrir o arquivo.
try:
    # Tenta abrir o arquivo "sample.text" em modo leitura.
    with open("sample.text", "r") as file:
        # Lê todo o conteúdo do arquivo e armazena na variável "content".
        content = file.read()
except FileNotFoundError:
    # Caso o arquivo não exista, exibe uma mensagem de erro.
    print("File Not Found!")