# ============================================================================
# additional_practice_1.py
# OBJETIVO: Simular o lançamento de uma moeda 10.000 vezes e calcular a
# probabilidade experimental (frequentista) de Cara e Coroa.
# IDEIA CENTRAL (Lei dos Grandes Números): quanto mais lançamentos fizermos,
# mais a frequência observada se aproxima da probabilidade teórica (50%/50%).
# ============================================================================

import random

# --- 1. CONFIGURAÇÃO DA SIMULAÇÃO ---
# Definindo a quantidade de lançamentos (10 mil, conforme o enunciado).
# Um número grande reduz o erro estatístico da estimativa.
total_lancamentos = 10000

# --- 2. SIMULAÇÃO DOS LANÇAMENTOS ---
# Simulamos cada lançamento com random.randint(0, 1):
#   - 0 representa Cara
#   - 1 representa Coroa
# Como randint(0, 1) gera 0 ou 1 com igual chance, modelamos uma moeda justa.
# A list comprehension repete esse sorteio 'total_lancamentos' vezes.
resultados = [random.randint(0, 1) for _ in range(total_lancamentos)]

# --- 3. CONTAGEM DOS RESULTADOS ---
# O método .count() percorre a lista e conta quantas vezes cada valor apareceu.
total_caras = resultados.count(0)   # Quantas vezes saiu Cara
total_coroas = resultados.count(1)  # Quantas vezes saiu Coroa

# --- 4. CÁLCULO DAS PROBABILIDADES EXPERIMENTAIS ---
# Probabilidade frequentista: P(evento) = (nº de sucessos) / (nº total de tentativas).
# Com 10.000 lançamentos, esperamos valores próximos de 0.50 (50%).
prob_cara = total_caras / total_lancamentos
prob_coroa = total_coroas / total_lancamentos

# --- 5. EXIBIÇÃO DOS RESULTADOS ---
# ':.2%' formata o número como porcentagem com 2 casas decimais.
print(f"Quantidade de lançamentos: {total_lancamentos:}")
print(f"Cara: {total_caras} vezes | Probabilidade: {prob_cara:.2%}")
print(f"Coroa: {total_coroas} vezes | Probabilidade: {prob_coroa:.2%}")
# Valor teórico esperado para comparação: 50.00% para cada lado.