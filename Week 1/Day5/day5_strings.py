# === Concatenação de strings (juntar textos com o operador +) ===
# first = "Hello"
# second = "World"
# result = first + " " + second  # Junta "Hello" + " " + "World" = "Hello World"
# # print(result)

# === Slicing (fatias) — acessar partes de uma string pelo índice ===
# text = "Python Programming"
# # print(text[0:6])   # Pega do índice 0 ao 5 (exclui o 6) = "Python"
# # print(text[-11:])  # Pega os últimos 11 caracteres = "Programming"

# === f-strings — formatar texto com variáveis dentro de chaves {} ===
# name = "Alice"
# age = 25
# print(f"My name is {name} and I am {age} years old.")
# # Resultado: "My name is Alice and I am 25 years old."

# === split() — quebra uma string em uma lista de palavras ===
sentence = "Python is fun"
words = sentence.split()  # Divide por espaços = ["Python", "is", "fun"]
# print(words)

# === join() — junta os itens de uma lista em uma string, com um separador ===
new_sentence = "|".join(words)  # Junta com "|" = "Python|is|fun"
#print(new_sentence)

# === replace() — substitui um trecho de texto por outro ===
text = "I love Java"
update_text = text.replace("Java", "Python")  # Troca "Java" por "Python" = "I love Python"
#print(update_text)

# === strip() — remove espaços em branco (ou outros caracteres) das bordas ===
messy = "      Hello World    "  # String com espaços extras no início e fim
print(messy)                     # Mostra com os espaços: "      Hello World    "
cleaned_text = messy.strip()     # Remove os espaços das bordas
print(cleaned_text)              # Mostra limpo: "Hello World"
