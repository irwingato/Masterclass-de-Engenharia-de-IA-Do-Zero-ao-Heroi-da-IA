# ============================================
# PRÁTICA ADICIONAL 1 — Contar vogais em uma string
# ============================================

def contar_vogais(texto):
    # Define as vogais (maiúsculas e minúsculas) para comparar
    vogais = "aeiouAEIOU"

    # Inicializa o contador em zero
    contador = 0

    # Percorre cada caractere da string
    for char in texto:
        # Se o caractere estiver na lista de vogais, incrementa o contador
        if char in vogais:
            contador += 1

    return contador

# Programa principal 1
print("=" * 40)
print("PRÁTICA ADICIONAL 1 — Contar Vogais")
print("=" * 40)
texto1 = input("Digite um texto: ")
total = contar_vogais(texto1)
print(f'Número de vogais: {total}')
print()


# ============================================
# PRÁTICA ADICIONAL 2 — Encontrar e substituir emails com regex
# ============================================

import re

def encontrar_emails(texto):
    # Padrão regex para encontrar emails:
    # \w+        → um ou mais caracteres (nome do usuário)
    # @          → literalmente o caractere @
    # \w+        → um ou mais caracteres (domínio)
    # \.         → literalmente um ponto
    # \w+        → um ou mais caracteres (extensão: com, org, etc.)
    padrao = r'\w+@\w+\.\w+'
    # re.findall() retorna uma lista com todos os emails encontrados
    return re.findall(padrao, texto)

def substituir_emails(texto, substituto):
    # Usa re.sub() para substituir todos os emails que casam com o padrão
    # pelo texto de substituição fornecido
    padrao = r'\w+@\w+\.\w+'
    return re.sub(padrao, substituto, texto)

# Programa principal 2
print("=" * 40)
print("PRÁTICA ADICIONAL 2 — Emails com Regex")
print("=" * 40)
texto2 = input("Digite um texto com emails: ")
emails = encontrar_emails(texto2)
texto_substituido = substituir_emails(texto2, "[EMAIL OCULTO]")
print(f"Emails encontrados: {emails}")
print(f'Texto substituído: "{texto_substituido}"')
print()


# ============================================
# PRÁTICA ADICIONAL 3 — Inverter as palavras de uma sentença
# ============================================

def inverter_palavras(frase):
    # split() quebra a frase em lista de palavras (separadas por espaço)
    # Ex: "Eu gosto de Python" → ["Eu", "gosto", "de", "Python"]
    palavras = frase.split()

    # [::-1] inverte a ordem da lista
    # ["Eu", "gosto", "de", "Python"] → ["Python", "de", "gosto", "Eu"]
    palavras_invertidas = palavras[::-1]

    # join() junta as palavras de volta numa string com espaço entre elas
    return " ".join(palavras_invertidas)

# Programa principal 3
print("=" * 40)
print("PRÁTICA ADICIONAL 3 — Inverter Palavras")
print("=" * 40)
frase3 = input("Digite uma frase: ")
frase_invertida = inverter_palavras(frase3)
print(f'Frase invertida: "{frase_invertida}"')
