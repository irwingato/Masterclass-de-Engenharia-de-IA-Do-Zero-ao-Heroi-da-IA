# Importa o módulo 'os' para interagir com o sistema operacional (ex.: verificar se um arquivo existe)
import os
# Importa 'datetime' (mais precisamente a classe 'datetime') para validar datas fornecidas pelo usuário
from datetime import datetime
# Importa 'json' para salvar/exportar os dados no formato JSON
import json
# Importa 'csv' para exportar as tarefas em um arquivo CSV (planilha)
import csv

# Nome do arquivo onde as tarefas serão salvas de forma persistente
NOME_ARQUIVO = "tasks.txt"

# Carrega as tarefas salvas do arquivo e retorna um dicionário
def carregar_tarefas():
    tarefas = {}  # Dicionário: id_da_tarefa (int) -> {"titulo": str, "status": str, "prioridade": str, "prazo": str}
    if os.path.exists(NOME_ARQUIVO):  # Só lê o arquivo se ele já existir
        with open(NOME_ARQUIVO, "r") as arquivo:  # Abre o arquivo em modo leitura
            for linha in arquivo:
                # Cada linha tem o formato: "id | titulo | status | prioridade | prazo" -> separa pelos " | "
                partes = linha.strip().split(" | ")

                # Caso o arquivo já tenha o formato antigo (apenas 3 campos: id | titulo | status)
                if len(partes) == 5:
                    id_tarefa, titulo, status, prioridade, prazo = partes
                    tarefas[int(id_tarefa)] = {
                        "titulo": titulo,
                        "status": status,
                        "prioridade": prioridade,
                        "prazo": prazo
                    }
                # Compatibilidade com o formato antigo (3 campos), garantindo que tarefas antigas funcionem
                elif len(partes) == 3:
                    id_tarefa, titulo, status = partes
                    tarefas[int(id_tarefa)] = {
                        "titulo": titulo,
                        "status": status,
                        "prioridade": "MEDIA",      # valor padrão para tarefas antigas
                        "prazo": "Sem prazo"        # valor padrão para tarefas antigas
                    }
    return tarefas

# Salva todas as tarefas no arquivo (modo escrita sobrescreve o conteúdo anterior)
def salvar_tarefas(tarefas):
    with open(NOME_ARQUIVO, "w") as arquivo:
        for id_tarefa, tarefa in tarefas.items():
            # Grava cada tarefa em uma linha, no formato "id | titulo | status | prioridade | prazo"
            arquivo.write(f"{id_tarefa} | {tarefa['titulo']} | {tarefa['status']} | {tarefa['prioridade']} | {tarefa['prazo']}\n")

# Adiciona uma nova tarefa à lista
def adicionar_tarefa(tarefas):
    titulo = input("Digite o título da tarefa: ")  # Lê o título fornecido pelo usuário

    # Validação da Prioridade
    print("Selecione a prioridade: 1. ALTA | 2. MÉDIA | 3. BAIXA")
    opcao_p = input("Digite sua escolha (1/2/3): ")
    prioridade = "MEDIA"  # valor padrão
    if opcao_p == "1":
        prioridade = "ALTA"
    elif opcao_p == "3":
        prioridade = "BAIXA"

    # Validação do Prazo (data de entrega)
    prazo = input("Digite um prazo (DD/MM/AAAA) ou pressione Enter para sem prazo: ").strip()
    if prazo:
        try:
            # Valida se o usuário digitou uma data real no formato correto (Dia/Mês/Ano)
            datetime.strptime(prazo, "%d/%m/%Y")
        except ValueError:
            print("Formato de data inválido. Tarefa criada com 'Sem prazo'.")
            prazo = "Sem prazo"
    else:
        prazo = "Sem prazo"

    # Gera um novo ID: o maior ID existente + 1 (default=0 caso a lista esteja vazia)
    id_tarefa = max(tarefas.keys(), default=0) + 1

    # Cria a tarefa com todos os campos definidos
    tarefas[id_tarefa] = {
        "titulo": titulo,
        "status": "incompleta",
        "prioridade": prioridade,
        "prazo": prazo
    }
    print(f"Tarefa '{titulo}' adicionada.")

# Exibe todas as tarefas cadastradas
def exibir_tarefas(tarefas):
    if not tarefas:  # Se o dicionário estiver vazio
        print("Nenhuma tarefa disponível.")
    else:
        for id_tarefa, tarefa in tarefas.items():  # Percorre todas as tarefas
            # Exibe todos os campos: id, título, status, prioridade e prazo
            print(f"[{id_tarefa}] {tarefa['titulo']} - {tarefa['status']} (prioridade: {tarefa['prioridade']}, prazo: {tarefa['prazo']})")

# Marca uma tarefa como concluída
def marcar_tarefa_completa(tarefas):
    id_tarefa = int(input("Digite o ID da tarefa para marcar como completa: "))  # ID informado pelo usuário
    if id_tarefa in tarefas:  # Verifica se a tarefa existe
        tarefas[id_tarefa]["status"] = "completa"  # Atualiza o status para "completa"
        print(f"Tarefa '{tarefas[id_tarefa]['titulo']}' marcada como completa.")
    else:
        print(f"ID da tarefa não encontrado.")

# Remove (exclui) uma tarefa da lista
def excluir_tarefa(tarefas):
    id_tarefa = int(input("Digite o ID da tarefa para excluir: "))  # ID informado pelo usuário
    if id_tarefa in tarefas:  # Verifica se a tarefa existe
        # pop() remove o item do dicionário e retorna o valor removido (o dicionário da tarefa)
        tarefa_excluida = tarefas.pop(id_tarefa)
        # Exibe o título da tarefa excluída usando o dicionário retornado
        print(f"Tarefa '{tarefa_excluida['titulo']}' excluída.")
    else:
        print(f"ID da tarefa não encontrado.")

# Exporta o dicionário de tarefas para um arquivo JSON estruturado
def exportar_para_json(tarefas):
    """Exporta o dicionário de tarefas para um arquivo JSON estruturado."""
    if not tarefas:
        print("Nenhuma tarefa disponível para exportar.")
        return

    # Nome padrão do arquivo de saída
    nome_arquivo = "exported_tasks.json"

    # O JSON exige chaves como strings, então convertemos o ID (int) para string
    dados_exportacao = {str(id_tarefa): tarefa for id_tarefa, tarefa in tarefas.items()}

    # 'encoding' deve ser escrito corretamente (sem o typo 'enconding')
    with open(nome_arquivo, "w", encoding="utf-8") as arquivo_json:
        # indent=4 deixa o arquivo legível para humanos (com quebras de linha e espaçamentos)
        # ensure_ascii=False permite caracteres acentuados (como ã, ç) serem salvos normalmente
        json.dump(dados_exportacao, arquivo_json, indent=4, ensure_ascii=False)

    print(f"Tarefas exportadas com sucesso para {nome_arquivo}!")

# Exporta o dicionário de tarefas para um arquivo CSV
def exportar_para_csv(tarefas):
    """Exporta o dicionário de tarefas para um arquivo CSV."""
    if not tarefas:
        print("Nenhuma tarefa disponível para exportar.")
        return

    # Nome padrão do arquivo de saída
    nome_arquivo = "exported_tasks.csv"

    with open(nome_arquivo, "w", newline="", encoding="utf-8") as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow(["ID Tarefa", "Título", "Status", "Prioridade", "Prazo"])  # cabeçalho
        for id_tarefa, tarefa in tarefas.items():
            escritor.writerow([id_tarefa, tarefa["titulo"], tarefa["status"], tarefa["prioridade"], tarefa["prazo"]])

    print(f"Tarefas exportadas com sucesso para {nome_arquivo}!")

# Menu principal que controla o fluxo do programa
def main():
    tarefas = carregar_tarefas()  # Carrega as tarefas salvas ao iniciar
    while True:  # Loop infinito até o usuário escolher sair
        print("\nMenu do Gerenciador de Tarefas:")
        print("1. Adicionar Tarefa")
        print("2. Exibir Tarefas")
        print("3. Marcar Tarefa como Completa")
        print("4. Excluir Tarefa")
        print("5. Exportar Tarefas (JSON / CSV)")
        print("6. Sair")

        escolha = input("Digite sua escolha (1/2/3/4/5/6): ")  # Lê a opção escolhida

        # Redireciona para a função correspondente à opção escolhida
        if escolha == "1":
            adicionar_tarefa(tarefas)
        elif escolha == "2":
            exibir_tarefas(tarefas)
        elif escolha == "3":
            marcar_tarefa_completa(tarefas)
        elif escolha == "4":
            excluir_tarefa(tarefas)
        elif escolha == "5":
            print("\nEscolha o formato de exportação:")
            print("1. Exportar para JSON")
            print("2. Exportar para CSV")
            formato = input("Digite sua escolha (1/2): ")
            if formato == "1":
                exportar_para_json(tarefas)
            elif formato == "2":
                exportar_para_csv(tarefas)
            else:
                print("Formato inválido.")
        elif escolha == "6":  # Encerra o programa
            salvar_tarefas(tarefas)  # Salva antes de sair para não perder as alterações
            print("Até logo!")
            break  # Encerra o loop
        else:
            print("Escolha inválida. Tente novamente.")  # Opção inválida

# Garante que main() só rode quando o script for executado diretamente
# (e não quando for importado como módulo em outro arquivo)
if __name__ == "__main__":
    main()
