# Copia o conteúdo de "sample.txt" para "sample_copy.txt".
# Usa encoding utf-8 para garantir acentos e caracteres especiais.
try:
    # Forma combinada de gerenciadores de contexto (abre origem e destino juntos).
    with open("sample.txt", "r", encoding="utf-8") as origem, \
         open("sample_copy.txt", "w", encoding="utf-8") as destino:
        # Copia em blocos de 4096 bytes, eficiente mesmo para arquivos grandes.
        while True:
            bloco = origem.read(4096)
            if not bloco:
                break
            destino.write(bloco)
    print("Arquivo copiado com sucesso para sample_copy.txt")
except FileNotFoundError:
    print("Erro: o arquivo 'sample.txt' não foi encontrado!")
except IOError as e:
    print(f"Erro ao copiar o arquivo: {e}")
