# Solicita uma frase ao usuário
sentence = input("Digite uma frase: ")

# Divide a frase em palavras e converte tudo para letras minúsculas
words = sentence.lower().split()

# Cria um dicionário com as palavras na ordem normal.
# Cada palavra será um chave e o valor será None.
normal_dictionary = dict.fromkeys(words)

# Cria um dicionário com as palavras na ordem inversa.
# O dicionário também remove as palavras repetidas.
inverse_dictionary = dict.fromkeys(reversed(words))

# Exibe as palavras na ordem normal.
# O método keys() acessa somente as chaves, sem mostrar valores.
print("\nPalavras na ordem normal:")
print(" ".join(normal_dictionary.keys()))

# Exibe as palavras na ordem inversa.
print("\nPalavras na ordem inversa:")
print(" ".join(inverse_dictionary.keys()))

# Exibe o dicionário normal, sem valores de repetição.
print("\nDicionário normal:")
# " ".join() está percorrendo apenas as chaves do dicionário, e não os valores por isso o none não aparece
print(" ".join(normal_dictionary))

# Exibe o dicionário invertido, sem valores de repetição.
print("\nDicionário inverso:")
# " ".join() está percorrendo apenas as chaves do dicionário, e não os valores por isso o none não aparece
print(" ".join(inverse_dictionary))