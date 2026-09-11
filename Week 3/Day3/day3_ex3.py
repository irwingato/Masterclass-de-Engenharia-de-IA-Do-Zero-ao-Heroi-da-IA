"""
PROGRAMA: Gradiente Descendente para Regressão Linear (mínimos quadrados)

OBJETIVO: Encontrar os parâmetros ótimos (θ₀, θ₁) de um modelo linear
          y = θ₀ + θ₁·x que melhor ajusta aos dados de treino, usando
          Gradiente Descendente para minimizar o Erro Quadrático Médio (MSE).

CONCEITO MATEMÁTICO:
  - MODELO LINEAR: ŷ = θ₀ + θ₁·x, onde θ₀ é o intercepto e θ₁ a inclinação.
  - MATRIZ DE DESIGN X: inclui uma coluna de 1s (para o bias θ₀) e as variáveis.
  - FUNÇÃO DE CUSTO: J(θ) = (1/2m)·Σ(ŷᵢ − yᵢ)² → Erro Quadrático Médio.
  - GRADIENTE: ∇J(θ) = (1/m)·Xᵀ·(Xθ − y) → direção que mais aumenta o erro.
  - ATUALIZAÇÃO: θ ← θ − lr·∇J(θ) → passo na direção que DIMINUI o erro.
"""

# ─── BIBLIOTECA ────────────────────────────────────────────────────────────────
# NumPy: operações matriciais, produto escalar, algebra linear.
import numpy as np


# ─── FUNÇÃO: GRADIENTE DESCENDENTE PARA REGRESSÃO LINEAR ───────────────────────
def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    X             → matriz de design (m × n+1): cada linha é uma amostra,
                    primeira coluna = 1 (bias), demais = variáveis.
    y             → vetor alvo (m × 1): valores reais observados.
    theta         → vetor de parâmetros (n+1 × 1): pesos do modelo.
    learning_rate → taxa de aprendizado (tamanho do passo em cada atualização).
    iterations    → número de repetições do algoritmo.
    """

    m = len(y)  # número de amostras (usado para normalizar o gradiente: média)

    for _ in range(iterations):
        # 1) PREVISÕES: calcula ŷ = X·θ (produto matriz × vetor)
        #    Em notação: ŷᵢ = θ₀·1 + θ₁·xᵢ₁ + θ₂·xᵢ₂ + ...
        #    np.dot(X, theta) faz a multiplicação de matrizes X(m×n) · theta(n×1) = vet(m×1)
        predictions = np.dot(X, theta)

        # 2) ERROS: diferença entre previsão e valor real (resíduo)
        #    errorsᵢ = ŷᵢ − yᵢ
        #    Se o erro > 0 → o modelo prevê demais; se < 0 → prevê de menos.
        errors = predictions - y

        # 3) GRADIENTE DA FUNÇÃO DE CUSTO:
        #    ∇J(θ) = (1/m) · Xᵀ · errors
        #
        #    MATEMÁTICA: derivada do MSE em relação a θ:
        #      ∂J/∂θⱼ = (1/m) · Σ xᵢⱼ · (ŷᵢ − yᵢ)  (para cada peso j)
        #
        #    X.T → transposta da matriz (n×m); X.T @ errors → (n×1)
        #    Divide por m para obter a MÉDIA dos gradientes (normalização).
        #    Sem dividir por m, amostras extras influenciariam demais.
        gradients = (1/m) * np.dot(X.T, errors)

        # 4) ATUALIZAÇÃO DOS PARÂMETROS:
        #    θ ← θ − lr · ∇J(θ)
        #    Na prática: θⱼ ← θⱼ − lr · ∂J/∂θⱼ
        #    Isso move θ na direção que REDUZ o erro (oposta ao gradiente).
        theta -= learning_rate * gradients

    # Retorna os parâmetros já ajustados (após todas as iterações)
    return theta


# ─── DADOS DE EXEMPLO ──────────────────────────────────────────────────────────
# MODELO: y = θ₀ + θ₁·x (reta)
#
# X: matriz 3×2 (3 amostras, 2 colunas: bias + variável x)
#    coluna 1 toda = 1 (representa o bias/intercepto θ₀)
#    coluna 2 = valores de x: [1, 2, 3]
#    Na prática: cada linha é [1, xᵢ]
X = np.array([[1, 1], [1, 2], [1, 3]])

# y: valores reais observados (alvos)
#    y₀ = 2  (para x=1),  y₁ = 2.5  (para x=2),  y₂ = 3.5  (para x=3)
y = np.array([2, 2.5, 3.5])

# theta: chute inicial dos parâmetros
#    θ₀ = 0.1 (intercepto),  θ₁ = 0.1 (inclinação inicial)
#    O GD vai corrigir estes valores até encontrar a reta ideal.
theta = np.array([0.1, 0.1])

# learning_rate = 0.1 → tamanho do passo
iterations = 1000


# ─── EXECUÇÃO ──────────────────────────────────────────────────────────────────
# Roda o GD 1000 vezes; a cada iteração, θ se move em direção ao menor MSE.
optimized_theta = gradient_descent(X, y, theta, learning_rate, iterations)

# ─── RESULTADO ─────────────────────────────────────────────────────────────────
# optimized_theta[0] = θ₀ (intercepto),  optimized_theta[1] = θ₁ (inclinação)
# Espera-se algo perto de: θ₀ ≈ 1.17,  θ₁ ≈ 0.75 (reta que melhor se ajusta)
print("Optimized Parameters:", optimized_theta)
