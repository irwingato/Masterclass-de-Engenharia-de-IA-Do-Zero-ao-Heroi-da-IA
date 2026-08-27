import re  # Importa o módulo de expressões regulares

def clean_text(text):
    # Remove pontuação: [^\w\s] significa "qualquer coisa que NÃO seja letra/número (\w) ou espaço (\s)"
    # Tudo que casar é substituído por "" (apagado)
    text = re.sub(r'[^\w\s]', "", text)

    # Remove espaços extras:
    # text.split() quebra a string em lista (ignora múltiplos espaços)
    # " ".join() junta tudo de volta com um único espaço entre cada palavra
    text = " ".join(text.split())

    # Converte o texto todo para letras minúsculas
    # Ex: "Hello" → "hello"
    return text.lower()

input_text = "Hello, World.!!! Welcome to Python, Programming....    "
cleaned_text = clean_text(input_text)
print("Cleaned Text:", cleaned_text)
# Saída: "hello world welcome to python programming"
