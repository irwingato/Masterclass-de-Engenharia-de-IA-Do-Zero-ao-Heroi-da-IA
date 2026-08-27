# Importa o módulo "string", que tem a lista de sinais de pontuação (!, ? , . etc.)
import string

def contar_palavra(texto, palavra_alvo):
    # Passo 1: deixa o texto todo em letras minúsculas (para não diferenciar maiúsculas de minúsculas)
    texto = texto.lower()
    palavra_alvo = palavra_alvo.lower()

    # Passo 2: divide o texto em palavras, usando espaços como separador
    # (split() já remove os espaços e transforma o texto numa lista de palavras)
    palavras = texto.split()

    # Passo 3: remove a pontuação do final de cada palavra
    # Exemplo: "hoje?" vira "hoje", "dia," vira "dia"
    palavras_limpas = []
    for p in palavras:
        p_limpa = p.strip(string.punctuation)  # tira pontuação das pontas
        if p_limpa:  # só adiciona se não ficou vazia
            palavras_limpas.append(p_limpa)

    # Passo 4: conta quantas palavras são iguais à palavra-alvo
    # Agora contamos PALAVRAS EXATAS, não pedaços de texto
    contador = 0
    for p in palavras_limpas:
        if p == palavra_alvo:
            contador = contador + 1

    return contador

# Exemplo de uso
texto = "Bom dia, belo dia hoje não, hoje está quente não?"
palavra_alvo = "hoje"
num_ocorrencias = contar_palavra(texto, palavra_alvo)
print(f"A palavra '{palavra_alvo}' aparece {num_ocorrencias} vezes no texto.")
