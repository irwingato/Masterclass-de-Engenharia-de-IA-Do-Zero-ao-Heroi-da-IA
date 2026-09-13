"""
day5_ex1.py
===========
OBJETIVO: Aplicar o TEOREMA DE BAYES a um problema médico clássico.

Problema:
- Uma doença afeta 1% da população (prior).
- O teste acerta 95% nos doentes (sensibilidade) e 90% nos saudáveis (especificidade).
- Pergunta: se uma pessoa testou POSITIVO, qual a probabilidade REAL de ela estar doente?

Resposta intuitiva errada: "95%". Resposta correta de Bayes: ~8.76%.
Por quê? Porque a doença é rara, então a maioria dos positivos são falsos positivos.
"""

# --- 1. DEFINIÇÃO DA FUNÇÃO DO TEOREMA DE BAYES ---
def bayes_theorem(prior, sensitivity, specificity):
    """
    Calcula P(Doente | Teste Positivo) pela fórmula de Bayes.

    Parâmetros:
    - prior: P(Doente) = prevalência na população (ex: 0.01).
    - sensitivity: P(Positivo | Doente) = taxa de verdadeiros positivos (ex: 0.95).
    - specificity: P(Negativo | Saudável) = taxa de verdadeiros negativos (ex: 0.90).

    Retorna:
    - posterior: P(Doente | Positivo) = probabilidade atualizada após ver o teste.
    """
    # Teorema da Probabilidade Total: P(Positivo) = positivos verdadeiros + falsos positivos.
    # - (sensitivity * prior) = P(Positivo E Doente) = 0.95 * 0.01 = 0.0095
    # - (1 - specificity) = P(Positivo | Saudável) = taxa de FALSO positivo = 1 - 0.90 = 0.10
    # - (1 - prior) = P(Saudável) = 0.99
    # - Logo: evidence = 0.0095 + (0.10 * 0.99) = 0.0095 + 0.099 = 0.1085
    evidence = (sensitivity * prior) + ((1 - specificity) * (1 - prior))

    # Fórmula de Bayes: Posterior = (Verossimilhança * Prior) / Evidência
    # Ou seja: P(Doente|Positivo) = P(Positivo|Doente) * P(Doente) / P(Positivo)
    posterior = (sensitivity * prior) / evidence

    # Devolve o valor calculado para quem chamou a função
    return posterior


# --- 2. DEFINIÇÃO DOS PARÂMETROS DO PROBLEMA ---
prior = 0.01  # P(Doente) = 1% da população tem a doença (prevalência baixa = raro)
sensitivity = 0.95  # P(Positivo|Doente) = 95% de acerto quando a pessoa ESTÁ doente
specificity = 0.90  # P(Negativo|Saudável) = 90% de acerto quando a pessoa NÃO está doente

# --- 3. CÁLCULO E EXIBIÇÃO DO RESULTADO ---
# Chama a função com os 3 parâmetros e guarda o resultado na variável posterior
posterior = bayes_theorem(prior, sensitivity, specificity)

# Imprime a probabilidade final em formato decimal (ex: 0.0875...)
print("Probability of Disease Given Positive Test: ", posterior)

# Imprime também em porcentagem para leitura intuitiva (~8.76%)
print(f"Em porcentagem: {posterior:.2%}")

# Mensagem de interpretação: explica por que o valor é baixo apesar do teste ser "95% preciso"
print("Interpretação: mesmo com teste positivo, a chance real é baixa porque a doença é rara")
print("e os falsos positivos (10% de 99% saudáveis) superam os verdadeiros positivos.")
