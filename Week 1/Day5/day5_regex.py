import re  # Importa o módulo "re" (regular expressions) do Python

text = "Contact me at 123-456-7890"  # Texto com um número de telefone

# re.findall() — encontra TODAS as ocorrências de um padrão no texto
# r"\d+" significa: \d = qualquer dígito (0-9), + = um ou mais repetidos
# Resultado: ["123", "456", "7890"] (cada grupo de dígitos separado)
digits = re.findall(r"\d+", text)
print(digits)

# re.sub() — substitui o que casar com o padrão por um novo texto
# r"\d" significa: \d = qualquer dígito único
# Cada dígito individual é trocado por "X"
# Resultado: "Contact me at XXX-XXX-XXXX"
updated_text = re.sub(r"\d", "X", text)
print(updated_text)
