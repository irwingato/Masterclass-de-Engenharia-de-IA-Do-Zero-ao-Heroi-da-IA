# Solicita ao usuário que digite uma frase.
sentence = input("Enter a sentence: ")

# Divide a frase em palavras.
# O método split() separa o texto sempre que encontra espaços.
words = sentence.split()

# Cria um dicionário vazio para armazenar
# cada palavra e a quantidade de vezes que ela aparece.
word_count = {}

# Percorre todas as palavras da frase.
for word in words:
    # Converte a palavra para letras minúsculas.
    # Assim, "Python" e "python" serão consideradas a mesma palavra.
    word = word.lower()

    # Verifica se a palavra já existe no dicionário.
    if word in word_count:
        # Se a palavra já estiver no dicionário,
        # aumenta sua quantidade em 1.
        word_count[word] += 1
    else:
        # Se a palavra ainda não estiver no dicionário,
        # adiciona a palavra com a quantidade inicial de 1.
        word_count[word] = 1

# Exibe o dicionário com cada palavra
# e a quantidade de vezes que ela apareceu.
print(word_count)