import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Prática 2: Vanilla SGD (batch=1) vs Mini-Batch SGD  --  Regressão Linear
# ----------------------------------------------------------------------------
# MATEMÁTICA:
#   Modelo linear:  ŷ = X·w  onde X é a matriz (n_amostras, 2) e w = [w1, w2].
#
#   Função de custo MSE em todo o conjunto:
#     MSE(w) = (1/n)·Σᵢ (x_i·w - y_i)²
#
#   Gradiente do MSE:
#     ∇MSE(w) = -(2/n)·Xᵀ·(y - X·w)
#   Para um único exemplo i (Vanilla SGD): ∇ = -2·(y_i - x_i·w)·x_i
#   Para um mini-batch B (Mini-Batch SGD):  ∇ = -(2/|B|)·Σ_{i∈B} (y_i - x_i·w)·x_i
#
#   Atualização comum a ambos:   w ← w - lr·∇
# ============================================================================

# 1. Geração de Dados Sintéticos (Problema de Regressão Linear)
np.random.seed(42)
num_samples = 200
X_data = np.random.uniform(-3, 3, (num_samples, 2))     # 2 features em [-3, 3]
true_weights = np.array([2.0, 3.0])  # O objetivo é encontrar estes valores
# y = X·w_true + ruído N(0, 0.5)  -> alvo com leve variação aleatória
y_data = X_data.dot(true_weights) + np.random.normal(0, 0.5, num_samples)

# Função para calcular o MSE global na superfície
def compute_loss_surface(w1, w2):
    # Calcula a perda para a malha do gráfico 3D/Contorno
    # Avalia MSE(w) para cada par (w1, w2) da malha, célula por célula.
    # Isso gera o "mapa de custo" que visualizamos no contorno.
    loss = np.zeros_like(w1)
    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
            w = np.array([w1[i, j], w2[i, j]])
            loss[i, j] = np.mean((X_data.dot(w) - y_data) ** 2)
    return loss

# 2. Implementação do Vanilla SGD (Batch Size = 1)
def vanilla_sgd(w_start, lr=0.005, epochs=3):
    w = w_start.copy()
    path = [w.copy()]

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)  # Embaralha a cada época

        for idx in indices:                        # Cada amostra, uma a uma
            xi = X_data[idx]                       # Um único exemplo
            yi = y_data[idx]
            # Gradiente de uma única amostra:  ∇ = -2·(y_i - x_i·w)·x_i
            grad = -2 * (yi - xi.dot(w)) * xi
            w -= lr * grad                         # Atualização (batch = 1)
            path.append(w.copy())
    return np.array(path)

# 3. Implementação do Mini-Batch SGD
def minibatch_sgd(w_start, batch_size=16, lr=0.01, epochs=3):
    w = w_start.copy()
    path = [w.copy()]

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)                 # Embaralha a cada época

        for i in range(0, num_samples, batch_size):   # Fatia o conjunto em blocos
            batch_idx = indices[i:i+batch_size]       # Índices do mini-batch
            X_batch = X_data[batch_idx]
            y_batch = y_data[batch_idx]

            # Gradiente médio do mini-batch:
            #   ∇ = -(2/|B|)·Σ (y_i - x_i·w)·x_i
            # Broadcasting: (y_batch - X_batch·w)[:, None] dá shape (|B|,1),
            # multiplicado por X_batch (|B|,2) via broadcast, e a média é
            # sobre as amostras (axis=0) -> vetor (2,).
            grad = -2 * np.mean((y_batch - X_batch.dot(w))[:, None] * X_batch, axis=0)
            w -= lr * grad
            path.append(w.copy())
    return np.array(path)

# 4. Executando as Otimizações
w_initial = np.array([-2.0, -1.0])  # Ponto de partida distante do ideal (2.0, 3.0)

# Chamadas reais (diferentes lr/épocas por serem métodos distintos):
path_vanilla = vanilla_sgd(w_initial, lr=0.005, epochs=2)      # 400 atualizações
path_minibatch = minibatch_sgd(w_initial, batch_size=20, lr=0.02, epochs=5)
#   200/20 = 10 batches por época x 5 épocas = 50 atualizações (menos ruído)

# 5. Criando a Superfície de Contorno
w1_vals = np.linspace(-3, 4, 100)
w2_vals = np.linspace(-2, 5, 100)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z_loss = compute_loss_surface(W1, W2)

# 6. Plotagem e Comparação Visual
plt.figure(figsize=(10, 7))
contours = plt.contour(W1, W2, Z_loss, levels=25, cmap='viridis')
plt.colorbar(contours, label='Custo (MSE)')

# Plotar caminhos
# Vanilla SGD: muitas atualizações => trajetória ruidosa e ziguezagueante
plt.plot(path_vanilla[:, 0], path_vanilla[:, 1], color='red', alpha=0.6,
         linewidth=1, marker='o', markersize=3, label='Vanilla SGD (Batch=1)')
# Mini-Batch: gradiente mais estável => caminho mais suave e rápido ao alvo
plt.plot(path_minibatch[:, 0], path_minibatch[:, 1], color='blue', alpha=0.9,
         linewidth=2, marker='s', markersize=4, label='Mini-Batch SGD (Batch=20)')

# Destaques
plt.scatter(w_initial[0], w_initial[1], color='black', s=150, marker='X',
            label='Início', zorder=5)
plt.scatter(2.0, 3.0, color='gold', s=200, marker='*', edgecolor='black',
            label='Alvo Real (2.0, 3.0)', zorder=5)

plt.title('Comparação de Otimização: Vanilla SGD vs Mini-Batch SGD')
plt.xlabel('Peso 1 ($w_1$)')
plt.ylabel('Peso 2 ($w_2$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()