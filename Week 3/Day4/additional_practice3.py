import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Prática 3: Adam vs SGD  --  Otimização em Superfície Não-Convexa (Rastrigin)
# ----------------------------------------------------------------------------
# FUNÇÃO DE PERDA (Rastrigin 2D):
#   L(x, y) = (x² - 2·cos(2π·x)) + (y² - 2·cos(2π·y)) + 4
#   Possui MÍNIMOS LOCAIS em grade inteira: (0,0), (1,0), (-1,2), etc.
#   O mínimo GLOBAL é (0,0) com L = 0.
#
# GRADIENTE (verdadeiro):
#   ∂L/∂x = 2x + 4π·sin(2πx)       (regra do produto na derivada do cosseno)
#   ∂L/∂y = 2y + 4π·sin(2πy)
#
# ADAM - Adaptive Moment Estimation (Kingma & Ba, 2014):
#   m_t = β₁·m_{t-1} + (1-β₁)·g_t              --> média móvel (1º momento)
#   v_t = β₂·v_{t-1} + (1-β₂)·g_t²             --> variância móvel (2º momento)
#   m̂_t = m_t / (1 - β₁ᵗ)                       --> correção de viés
#   v̂_t = v_t / (1 - β₂ᵗ)                       --> correção de viés
#   θ_t = θ_{t-1} - lr·m̂_t / (√v̂_t + ε)       --> atualização adaptativa
#
# β₁=0.9, β₂=0.999, ε=1e-8 são os valores padrão recomendados no artigo.
# O Adam combina o Momementum (via m_t) com o RMSProp (via v_t):
#   - m_t acelera o SGD na direção de gradiente consistente.
#   - v_t reduz o step nas direções onde o gradiente oscila (normaliza).
#   - Correção de viés evita que m_t e v_t sejam subestimados no início.
# ============================================================================

# 1. Superfície de Perda Complexa (Baseada na Função de Rastrigin)
def complex_loss(x, y):
    # Rastrigin 2D: bowl quadrático + oscilações periódicas (mínimos locais)
    return (x**2 - 2 * np.cos(2 * np.pi * x)) + (y**2 - 2 * np.cos(2 * np.pi * y)) + 4

def complex_gradient(x, y):
    # Derivadas parciais analíticas com termos oscilatórios
    #   ∂L/∂x = 2x + 4π·sin(2πx)
    #   ∂L/∂y = 2y + 4π·sin(2πy)
    grad_x = 2 * x + 4 * np.pi * np.sin(2 * np.pi * x)
    grad_y = 2 * y + 4 * np.pi * np.sin(2 * np.pi * y)
    return np.array([grad_x, grad_y])

# 2. Implementação do Otimizador Adam
def adam_optimizer(start_pos, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=80):
    pos = np.array(start_pos, dtype=float)
    m = np.zeros_like(pos)  # 1º Momento (Média móvel do gradiente)
    v = np.zeros_like(pos)  # 2º Momento (Média móvel do gradiente ao quadrado)

    path = [pos.copy()]

    for t in range(1, steps + 1):
        grad = complex_gradient(pos[0], pos[1])

        # Atualização dos momentos com base nos hiperparâmetros clássicos
        #   m_t = 0.9·m_{t-1} + 0.1·g_t
        #   v_t = 0.999·v_{t-1} + 0.001·g_t²
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Correção de viés: compensa a inicialização m₀=0, v₀=0
        #   m̂_t = m_t / (1 - 0.9ᵗ)   → para t grande, ≈ m_t
        #   v̂_t = v_t / (1 - 0.999ᵗ)  → para t grande, ≈ v_t
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Atualização adaptativa dos pesos
        #   θ ← θ - lr · m̂_t / (√v̂_t + ε)
        pos -= lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(pos.copy())

    return np.array(path)

# 3. Implementação do SGD Tradicional para comparação
def sgd_optimizer(start_pos, lr=0.02, steps=80):
    pos = np.array(start_pos, dtype=float)
    path = [pos.copy()]

    for _ in range(steps):
        grad = complex_gradient(pos[0], pos[1])
        pos -= lr * grad             # Atualização SGD padrão: θ ← θ - lr·∇
        path.append(pos.copy())

    return np.array(path)

# 4. Geração do espaço tridimensional complexo
x_vals = np.linspace(-2.5, 2.5, 200)
y_vals = np.linspace(-2.5, 2.5, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = complex_loss(X, Y)

# Ponto de início (distante do centro e cercado por vales/mínimos locais)
start_point = [2.1, 1.9]

# Execução das simulações
path_adam = adam_optimizer(start_point, lr=0.08, steps=100)
path_sgd = sgd_optimizer(start_point, lr=0.015, steps=100)

# 5. Plotagem das trajetórias sobre o mapa de contorno
plt.figure(figsize=(11, 8))
contours = plt.contourf(X, Y, Z, levels=35, cmap='twilight_r', alpha=0.85)
plt.colorbar(contours, label='Custo / Perda')

# Caminhos de Otimização
# SGD: oscila entre mínimos locais mas geralmente fica preso num deles.
plt.plot(path_sgd[:, 0], path_sgd[:, 1], color='#FF3E3E', linewidth=2.5,
         marker='o', markersize=3, label='SGD (Preso em mínimo local)')
# Adam: graças à adaptação de lr, escapa de mínimos rasos e alcança (0,0).
plt.plot(path_adam[:, 0], path_adam[:, 1], color='#00FFCC', linewidth=2.5,
         marker='s', markersize=3, label='ADAM (Escapa de armadilhas)')

# Marcos visuais
plt.scatter(start_point[0], start_point[1], color='white', edgecolor='black',
            s=200, marker='X', label='Início', zorder=5)
plt.scatter(0, 0, color='gold', edgecolor='black', s=250, marker='*',
            label='Mínimo Global (0,0)', zorder=5)

plt.title('Adam vs SGD em Superfície Não-Convexa de Alta Complexidade', fontsize=14,
          fontweight='bold')
plt.xlabel('Parâmetro X')
plt.ylabel('Parâmetro Y')
plt.legend(loc='lower left', framealpha=0.9)
plt.grid(True, alpha=0.2)
plt.show()