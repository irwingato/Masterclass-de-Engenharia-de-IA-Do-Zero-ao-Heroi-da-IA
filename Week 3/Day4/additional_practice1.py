import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Prática 1: Superfície de Perda Quadrática + Caminho do SGD
# ----------------------------------------------------------------------------
# MATEMÁTICA:
#   Função de perda:  L(x, y) = (1/2)·(x² + 10y²)      --> elipse/vale estreito.
#   O fator 10 em y² torna o vale "íngreme" na direção y (eixo anisotrópico),
#   o que faz o SGD oscilar fortemente na direção de maior curvatura.
#
#   Gradiente (verdadeiro):  ∇L = [∂L/∂x, ∂L/∂y] = [x, 10y]
#
#   Atualização do SGD com ruído estocástico (simula mini-batches):
#     x ← x - lr·(∇L_x + ruído)
#     y ← y - lr·(∇L_y + ruído)
# ============================================================================

# 1. Definição da Função de Perda (Superfície Quadrática Assimétrica)
def loss_function(x, y):
    # Formato de elipse/vale, ideal para ver o comportamento do SGD
    return 0.5 * (x**2 + 10 * y**2)

def true_gradient(x, y):
    # Derivados parciais em relaçõa a x e y
    #   dL/dx = x     (derivada de 0.5·x²)
    #   dL/dy = 10y   (derivada de 0.5·10y² = 5y² -> 10y)
    return np.array([x, 10 * y])

# 2. Algoritmo do SGD (com ruído estocástico simulado)
def simulate_sgd(start_x, start_y, lr=0.12, steps=25, noise_scale=0.4):
    np.random.seed(42) # Garante reprodutibilidade
    path_x = [start_x]   # Lista que guarda TODOS os x visitados (trajetória)
    path_y = [start_y]   # Lista que guarda TODOS os y visitados (trajetória)

    x, y = start_x, start_y
    for _ in range(steps):
        # Gradiente verdadeiro no ponto atual (aponta de volta ao mínimo)
        grad = true_gradient(x, y)

        # Adiciona ruído para simular a variação de mini-batches do SGD
        # Ruído gaussiano N(0, noise_scale) em cada coordenada.
        # É isso que diferencia SGD de GD: cada passo é uma estimativa
        # ruidosa do gradiente, causando a "dança" oscilatória do caminho.
        noise = np.random.normal(0, noise_scale, size=2)
        sgd_grad = grad + noise

        # Atualização dos parâmetros (descida do gradiente)
        #   x ← x - lr·sgd_grad[0]
        #   y ← y - lr·sgd_grad[1]
        x = x - lr * sgd_grad[0]
        y = y - lr * sgd_grad[1]

        # Registra a nova posição na trajetória
        path_x.append(x)
        path_y.append(y)

    return np.array(path_x), np.array(path_y)

# 3. Geração do cenário e simulação
# Malha de pontos (x, y) para desenhar a "tigela" da função de perda.
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = loss_function(X, Y)          # Avalia a perda em toda a malha

# Executa a otimização saindo do ponto inicial (4.0, 1.5)
path_x, path_y = simulate_sgd(start_x=4.0, start_y=1.5, lr=0.15, steps=20)
path_z = loss_function(path_x, path_y)   # Perda correspondente a cada passo

# 4. Plotagem dos Gráficos (Gráfico 3D e Curvas de Nível)
fig = plt.figure(figsize=(14, 6))

# Subplot 1: Superfície 3D
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
ax1.plot(path_x, path_y, path_z, color='red', marker='o', markersize=4,
         label='Caminho SGD', zorder=10)   # Trajetória descendo o vale
ax1.set_title('Superfície de Perda em 3D')
ax1.set_xlabel('Peso X')
ax1.set_ylabel('Peso Y')
ax1.set_zlabel('Perda (Loss)')
ax1.legend()

# Subplot 2: Curvas de Nível (Contour Plot) - melhor par aver a oscilação do SGD
ax2 = fig.add_subplot(1, 2, 2)
contours = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contours, ax=ax2, label='Valor da Perda')
ax2.plot(path_x, path_y, color='red', marker='o', linestyle='-', linewidth=1.5,
         label='Passos do SGD')            # Vê-se a oscilação em zigue-zague
ax2.scatter(path_x[0], path_y[0], color='blue', s=100, label='Início', zorder=5)
ax2.scatter(0, 0, color='gold', marker='*', s=200, label='Mínimo Global', zorder=5)
ax2.set_title('Mapa de Contorno e Caminho de Otimização')
ax2.set_xlabel('Peso X')
ax2.set_ylabel('Peso Y')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()