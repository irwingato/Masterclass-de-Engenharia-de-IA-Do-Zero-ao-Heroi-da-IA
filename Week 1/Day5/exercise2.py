def is_palindrome(text):
    # Limpa o texto para verificar se é palíndromo:
    # 1. for char in text — percorre cada caractere da string
    # 2. char.isalnum() — filtra: mantém apenas letras e números (remove espaços, pontuação, acentos)
    # 3. char.lower() — converte cada caractere para minúsculo
    # 4. "".join(...) — junta tudo de volta numa string limpa
    # Ex: "A FootbaLl" → "afootball"
    text = "".join(char.lower() for char in text if char.isalnum())

    # Verifica se o texto limpo é igual ao seu inverso
    # text[::-1] — fatia a string do fim ao início, invertendo-a
    # "racecar"[::-1] → "racecar" → igual → True (é palíndromo)
    # "hello"[::-1] → "olleh" → diferente → False
    return text == text[::-1]

# Pede ao usuário que digite uma frase no terminal
input_text = input("Digite uma frase: ")

# Chama a função e verifica o resultado
if is_palindrome(input_text):
    # Se True — a frase lida da mesma forma de trás pra frente
    print(f'"{input_text}" é um palíndromo.')
else:
    # Se False — a frase NÃO é igual ao contrário
    print(f'"{input_text}" não é um palíndromo.')