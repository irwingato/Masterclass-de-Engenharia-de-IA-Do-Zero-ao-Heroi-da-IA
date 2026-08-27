# Define a função que conta linhas e palavras de um arquivo de texto.
def count_words_adns_lines(filename):
    # Bloco try/except para tratar o caso de o arquivo não existir.
    try:
        # Abre o arquivo em modo leitura ("r") e o fecha automaticamente ao sair do bloco.
        with open(filename, "r") as file:
            # Lê todas as linhas do arquivo e as armazena em uma lista.
            lines = file.readlines()
            # Conta o número de linhas (tamanho da lista).
            line_count = len(lines)
            # Conta o número total de palavras: divide cada linha em palavras
            # e soma a quantidade de palavras de todas as linhas.
            word_count = sum(len(line.split()) for line in lines)

            # Exibe o número de linhas.
            print(f"Number of lines: {line_count}")
            # Exibe o número de palavras.
            print(f"Number of words: {word_count}")
    except FileNotFoundError:
        # Caso o arquivo não seja encontrado, exibe uma mensagem de erro.
        print(f"File '{filename}' not found!")

# Chama a função passando o arquivo "sample.txt" como argumento.
count_words_adns_lines("sample.txt")