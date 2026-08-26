# Solicita ao usuário que digite uma frase
sentence = input("Digite uma frase: ")

# Divide a frase em palavras.
# O método split() separa o texto sempre que encontra espaços.
words = sentence.split()

# Cria um dicionário vazio para armazenar
# cada palavra e a quantidade de vezes que ela aparece.
word_count = {}

words = sentence.split()

# Exibe as palavras na ordem em que foram digitadas.
print("\nPalavras na ordem normal:")
print(" ".join(words))

# Inverte a ordem das palavras
# O operador [::1-1] percorre a lista do último elemento até o primeiro.
reversed_words = words[::-1]

# Exibe palavras na ordem inversa.
print("\nPalavras na ordem inversa:")
print(" ".join(reversed_words))

# Cria uma lista vazia para armazenar as palavras sem repetição.
unique_words = []
    
for word in words:
    # Verifica se a palavra ainda não foi adicionado à lista.
    if word not in unique_words:
        #Adiciona a palavra somente uma vez
        unique_words.append(word)

# Exibe as palavras removendo as repetidas.
# Nenhuma quantidade ou valor de repetição será mostrado.
print("\nPalavras sem repetição:")
print(" ".join(unique_words))