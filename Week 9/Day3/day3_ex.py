import numpy as np

# ============================================================
# Importação: NumPy (biblioteca de computação científica)
# Fornece arrays multidimensionais e operações vetorizadas.
# Vetorização = executar operações em blocos de dados de uma vez,
# usando instruções SIMD (Single Instruction, Multiple Data) em C
# sob o Python, o que evita loops lentos em Python puro.
# ============================================================

# Mean Squared Error (MSE) - Erro Quadrático Médio
# Função de perda para regressão.
# Matemática: L_MSE = (1/n) * Σᵢ (y_trueᵢ - y_predᵢ)²
# - (y_true - y_pred) = resíduo (erro de previsão)
# - Elevado ao quadrado: penaliza erros grandes mais que pequenos
#   (crescimento quadrático), e elimina o sinal (erro sempre positivo)
# - Média (1/n): normaliza pelo número de amostras para que a perda
#   não dependa do tamanho do dataset.
# Ciência da Computação: np.mean e ** 2 são operações vetorizadas
# (O(n) no número de elementos), implementadas em C.
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Binary Cross-Entropy (BCE) - Entropia Cruzada Binária
# Função de perda para classificação binária (saída entre 0 e 1).
# Matemática: L_BCE = -(1/n) * Σᵢ [y_trueᵢ * log(y_predᵢ) +
#                                 (1 - y_trueᵢ) * log(1 - y_predᵢ)]
# - y_true ∈ {0, 1}: apenas um dos dois termos se mantém ativo
# - log é monotônica crescente: quanto mais perto y_pred de y_true,
#   menor a perda. Se y_pred → y_true, perda → 0.
# - Se y_pred → 0 quando y_true = 1 (ou y_pred → 1 quando y_true = 0),
#   log → -∞, perda explode (penaliza fortemente previsões confiantes
#   e erradas).
# - base natural (ln): derivada de ln(x) é 1/x, o que simplifica o
#   cálculo do gradiente.
# Ciência da Computação: np.clip evita log(0) = -inf e division by zero,
# limitando y_pred ao intervalo [ε, 1-ε] com ε = 1e-15.
def binary_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ============================================================
# Dados de exemplo (vetores NumPy)
# y_true: rótulos reais (ground truth), valores 0 ou 1
# y_pred: previsões do modelo (probabilidades entre 0 e 1)
# ============================================================
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.2, 0.8, 0.7])

# Cálculo das perdas para os dados de exemplo
mse = mse_loss(y_true, y_pred)
bce = binary_cross_entropy_loss(y_true, y_pred)

print(f"MSE LOSS: {mse:.4f}")
print(f"Binary Cross-Entropy LOSS: {bce:.4f}")

# ============================================================
# Gradientes (derivadas) das funções de perda
# Usados em otimização por descida do gradiente:
#   θ ← θ - η * ∇L(θ)   (atualização de parâmetros)
# onde η é a taxa de aprendizado (learning rate).
# ============================================================

# Derivada do MSE em relação a y_pred:
# dL/dy_pred = d/dy_pred [(1/n) Σ (y_pred - y_true)²]
#            = (1/n) Σ 2 * (y_pred - y_true) * d(y_pred)/dy_pred
#            = 2 * (y_pred - y_true) / n
# O gradiente é linear no erro: erros grandes → gradiente grande.
def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

# Derivada da BCE em relação a y_pred:
# L = -[y * ln(p) + (1-y) * ln(1-p)]
# dL/dp = -[y/p + (1-y) * (-1)/(1-p)]
#       = -y/p + (1-y)/(1-p)
#       = [-y(1-p) + p(1-y)] / [p(1-p)]
#       = [-y + yp + p - py] / [p(1-p)]
#       = (p - y) / [p(1-p)]
# O gradiente cresce à medida que p se aproxima de 0 ou 1 (denominador
# tende a 0), forçando correções fortes quando o modelo está confiante
# e errado.
def bce_gradient(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

# Cálculo dos gradientes para os dados de exemplo
grad_mse = mse_gradient(y_true, y_pred)
grad_bce = bce_gradient(y_true, y_pred)

print(f"MSE Gradients: {grad_mse}")
print(f"BCE Gradients: {grad_bce}")

import matplotlib.pyplot as plt

# ============================================================
# Visualização comparativa das funções de perda
# ============================================================

# predictions: 100 pontos espaçados uniformemente entre 0 e 1
# np.linspace(início, fim, n) gera n valores com espaçamento igual.
# Matemática: pᵢ = início + i * (fim - início) / (n - 1)
predictions = np.linspace(0, 1, 100)

# Rótulo fixo igual a 1 para isolar o comportamento da perda
# como função apenas da previsão p.
true_label = 1

# Calcula a perda MSE para cada previsão:
# L_MSE(p) = (1 - p)²  (quando y_true = 1)
# Gráficamente é uma parábola com mínimo em p = 1.
mse_losses = [(true_label - p)**2 for p in predictions]

# Calcula a perda BCE para cada previsão:
# L_BCE(p) = -[1 * ln(p) + 0 * ln(1-p)] = -ln(p)
# Gráficamente tende a +∞ quando p → 0 e a 0 quando p → 1.
# max(p, 1e-15) evita log(0).
bce_losses = [-true_label * np.log(max(p , 1e-15)) - (1 - true_label) * np.log(max(1-p, 1e-15)) for p in predictions]

# Cria a figura com tamanho 10x6 polegadas
plt.figure(figsize=(10,6))

# Plota as curvas de perda
plt.plot(predictions, mse_losses, label="MSE Loss")
plt.plot(predictions, bce_losses, label="Binary Cross-Entropy Loss")

# Títulos e rótulos dos eixos
plt.title('Loss Functions Comparison')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
