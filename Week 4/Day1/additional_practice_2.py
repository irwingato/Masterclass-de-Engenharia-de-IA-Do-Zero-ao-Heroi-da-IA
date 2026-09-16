# ============================================================================
# additional_practice_2.py
# OBJETIVO: Calcular a EXPECTATIVA (média teórica) e a VARIÂNCIA de um
# DADO PONDERADO (viciado), usando probabilidades ENVIESADAS (não uniformes).
#
# CONCEITOS:
#   - Expectativa: E(X) = soma de [x * P(x)] -> valor médio esperado no longo prazo.
#   - Variância: Var(X) = E(X²) - [E(X)]² -> mede a dispersão em torno da média.
#   - Dado justo teria P(x) = 1/6 ≈ 16.67% para cada face (E(X) = 3.5).
#     Aqui usamos um dado viciado com tendência a números altos.
# ============================================================================

# --- 1. DEFINIÇÃO DO ESPAÇO AMOSTRAL E DAS PROBABILIDADES ENVIESADAS ---
# 'faces' são os valores possíveis da variável aleatória X (resultado do dado).
faces = [1, 2 ,3, 4, 5, 6]
# 'probabilidades' é a função de massa de probabilidade P(X = x).
# Exemplo: o dado tem tendência a cair nos números mais altos (5 e 6 somam 50%).
# REGRA: a soma das probabilidades deve ser sempre 1.0 (100%).
# 0.10 + 0.10 + 0.15 + 0.15 + 0.20 + 0.30 = 1.00 -> válido.
probabilidades = [0.10, 0.10, 0.15, 0.15, 0.20, 0.30]

# Verificação de sanidade: garante que as probabilidades somam 1 (a menos de
# erro de ponto flutuante). Se a soma estiver errada, o cálculo não é válido.
assert abs(sum(probabilidades) - 1.0) < 1e-9, "As probabilidades devem somar 1.0!"

# --- 2. CÁLCULO DA EXPECTATIVA: E(X) = Σ [x * P(x)] ---
# Multiplicamos cada face pelo seu peso (probabilidade) e somamos tudo.
# zip(faces, probabilidades) emparelha cada face com sua probabilidade.
# Como o dado favorece valores altos, esperamos E(X) > 3.5 (média do dado justo).
expectativa = sum(x * p for x, p in zip(faces, probabilidades))

# --- 3. CÁLCULO DO E(X²) (passo intermediário para a variância) ---
# E(X²) = Σ [x² * P(x)]. Precisamos dele para aplicar a fórmula da variância.
# Note que E(X²) NÃO é igual a [E(X)]² — por isso calculamos separadamente.
expectativa_x2 = sum((x**2) * p for x, p in zip(faces, probabilidades))

# --- 4. CÁLCULO DA VARIÂNCIA: Var(X) = E(X²) - [E(X)]² ---
# A variância quantifica o espalhamento: quanto maior, mais imprevisível o dado.
# O desvio-padrão (raiz quadrada da variância) estaria na mesma unidade do dado.
variancia = expectativa_x2 - (expectativa**2)
desvio_padrao = variancia ** 0.5  # Informação extra: desvio-padrão = sqrt(Var)

# --- 5. EXIBIÇÃO DOS RESULTADOS ---
print(f"--- Estatísticas do Dado Ponderado ---")
for f, p in zip(faces, probabilidades):
    print(f"Face {f}: {p:.1%}")

print(f"\nExpectativa Matemática E(X): {expectativa:.2f}")
print(f"Variância Var(X): {variancia:.2f}")
print(f"Desvio-padrão DP(X): {desvio_padrao:.2f}")
# Para referência: dado justo -> E(X) = 3.50 e Var(X) ≈ 2.92.
# Aqui a expectativa deve ser maior que 3.50 por causa do viés para cima.