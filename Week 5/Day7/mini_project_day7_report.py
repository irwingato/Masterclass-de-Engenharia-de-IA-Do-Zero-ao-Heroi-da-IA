# =============================================================================
# MINI PROJETO DAY 7 — Parte 3: RELATÓRIO (Tarefa 3)
# Objetivo: imprimir no terminal um resumo estruturado dos Projetos 1 e 2.
# =============================================================================
# COMPUTAÇÃO:
# - Arquivo puramente de APRESENTAÇÃO: zero ML, zero matemática de modelo.
# - Define uma função que faz prints formatados e a chama no bloco main.
# - `if __name__ == "__main__":` é o idioma Python para "só execute isso quando o
#   arquivo for rodado diretamente" (`python mini_project_day7_report.py`).
#   Se o arquivo for importado (`import mini_project_day7_report`), a função fica
#   disponível mas NÃO executa sozinha — evita efeito colateral no import.
# =============================================================================

def gerar_relatorio_tarefa3():
    """
    Gera o resumo estruturado dos achados dos projetos de
    Aprendizado Supervisionado (Tarefa 1 e Tarefa 2).

    Computação: sequência de print() — O(1), sem loops nem I/O de arquivos.
    Cada bloco print descreve um projeto (objetivo, atributos, modelo, split, métrica).
    """
    # --- Cabeçalho ---
    # "=" * 60 repete o caractere 60x (multiplicação de string) para fazer moldura.
    print("=" * 60)
    print("        RELATÓRIO DE ACHADOS - APRENDIZADO SUPERVISIONADO        ")
    print("=" * 60)

    # --- MODELO 1: REGRESSÃO (California Housing) ---
    # Resumo do mini_project_day7.py: alvo contínuo MedHouseVal (US$100k),
    # 3 features (MedInc, HouseAge, AveRooms), Regressão Linear, split 80/20 (seed 42),
    # métrica MSE = (1/n)*soma((y-y_hat)²). Referência esperada: MSE ~0.52-0.55.
    print("\n[PROJETO 1] Predição de Preços de Imóveis (Regressão)")
    print("-" * 50)
    print("• Objetivo: Prever o valor médio de imóveis ('MedHouseVal').")
    print("• Atributos Utilizados: Renda Média ('MedInc'), Idade do Imóvel ('HouseAge')")
    print("                        e Média de Cômodos ('AveRooms').")
    print("• Modelo Aplicado: Regressão Linear.")
    print("• Divisão dos Dados: 80% Treino / 20% Teste (Random State: 42).")
    print("• Avaliação: Métrica de Erro Quadrático Médio (MSE).")
    print("• Observações do Código: O script possui etapas comentadas para")
    print("  Análise Exploratória de Dados (EDA), plotagem de gráficos com pairplot")
    print("  e checagem de valores ausentes.")

    # --- MODELO 2: CLASSIFICAÇÃO (Telco Customer Churn) ---
    # Resumo do mini_project_day7_telco.py: alvo binário churn (0/1, ~31% positivos),
    # pipeline get_dummies (one-hot, drop_first) + StandardScaler z-score com fit só
    # no treino + stratify, modelos LogReg (sigmoide + log-loss) vs k-NN (k=5, voto
    # por distância euclidiana), métricas precision/recall/F1 + matriz [[TN FP],[FN TP]].
    # NOTA: o texto original citava "LabelEncoder + fillna(mean)" — mantido abaixo por
    # fidelidade ao relatório, mas o código comentado explica por que one-hot +
    # fillna só em numéricas é o correto para LogReg/k-NN (LabelEncoder cria ordem falsa).
    print("\n[PROJETO 2] Previsão de Churn de Clientes (Classificação)")
    print("-" * 50)
    print("• Objetivo: Identificar a perda de clientes ('churn' - binário).")
    print("• Pré-processamento de Dados:")
    print("  - Tratamento de Nulos: Substituição de valores ausentes pela média.")
    print("  - Codificação: Uso do LabelEncoder para variáveis categóricas")
    print("    ('churn', 'gender', 'contract_type', 'payment_method').")
    print("  - Escalonamento: Uso do StandardScaler para padronizar os atributos.")
    print("• Modelos Aplicados: Regressão Logística vs. K-Nearest Neighbors (k-NN).")
    print("• Divisão dos Dados: 80% Treino / 20% Teste (Random State: 42).")
    print("• Avaliação: Relatório de Classificação (Precision, Recall, F1-Score)")
    print("             e Matriz de Confusão para a Regressão Logística.")

    print("\n" + "=" * 60)
    print("                    FIM DO RESUMO DA TAREFA 3                   ")
    print("=" * 60)

# Bloco main: ponto de entrada do script.
# `__name__` vale "__main__" só em execução direta; em import vale o nome do módulo.
if __name__ == "__main__":
    gerar_relatorio_tarefa3()  # chamada única: imprime o relatório no stdout
