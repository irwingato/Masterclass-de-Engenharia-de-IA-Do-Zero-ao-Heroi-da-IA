import numpy as np

# ============================================================================
# Exemplo: Regressão Linear com SGD (Stochastic Gradient Descent)
# ----------------------------------------------------------------------------
# Modelo real (gerador dos dados):  y = 4 + 3x + ruído
# Queremos estimar os parâmetros θ = [b (bias), w (peso)] ≈ [4, 3].
#
# MODELO LINEAR:  ŷ = X_b · θ   com X_b = [1, x]
#
# MATEMÁTICA DO GRADIENTE:
#   Função de custo (erro quadrático para UMA amostra i):
#     E_i(θ) = (1/2) * (x_iᵀθ - y_i)²      (o 1/2 simplifica a derivada)
#   Derivada em relação a θ (regra da cadeia):
#     ∇E_i(θ) = 2·x_i·(x_iᵀθ - y_i)
#   => É exatamente a operação feita abaixo: 2·xi.T·(xi@θ - yi)
#
# O SGD usa APENAS UMA amostra aleatória por atualização:
#     θ ← θ - lr · ∇E_i(θ)
# (ao contrário do GD, que usa a média de TODAS as amostras).
# ============================================================================

# Generate synthetic data
np.random.seed(42)                     # Reprodutibilidade
X = 2 * np.random.rand(100, 1)         # 100 pontos, feature x ∈ [0, 2)
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + ruído gaussiano N(0,1)

# Add bias term to X
# Concatena uma coluna de 1's para representar o termo de bias (intercepto).
# X_b vira a matriz do modelo: [1, x], shape (100, 2).
X_b = np.c_[np.ones((100, 1)), X]

# SGD Implementation
def stochastic_gradient_descendent(X, y, theta, learning_rate, n_epocs):
    m = len(y)                          # Número de amostras = 100
    for epoch in range(n_epocs):        # Repete todo o conjunto |n_epocs| vezes
        for i in range(m):              # --- 'i' NÃO é usado (o índice é sorteado)
            random_index = np.random.randint(m)          # Sorteia UMA amostra
            xi = X[random_index:random_index+1]          # x da amostra (1x2)
            yi = y[random_index:random_index+1]          # y da amostra (1x1)

            # Gradiente do erro quadrático da amostra sorteada:
            #   ∇E = 2·xiᵀ·(xi·θ - yi)   (vetor 2x1)
            gradients = 2 * xi.T.dot(xi @ theta - yi)

            # Atualização SGD: θ ← θ - lr·∇E
            theta = theta - learning_rate * gradients
    return theta

# initialize parameters
theta = np.random.randn(2, 1)          # Chute inicial aleatório (2 parâmetros)
learning_rate = 0.01
n_epochs = 50

# Perform SGD
theta_opt = stochastic_gradient_descendent(X_b, y, theta, learning_rate, n_epochs)
print("Optimized parameters: ", theta_opt)   # Esperado ≈ [[4.], [3.]]