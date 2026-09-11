"""
PROGRAMA: Visualização dos Caminhos do Gradiente Descendente

OBJETIVO: Mostrar graficamente (em 3 gráficos lado a lado) como diferentes
          taxas de aprendizado afetam a trajetória do Gradiente Descendente
          ao minimizar f(x, y) = x² + 5 + y².

CONCEITO MATEMÁTICO:
  - A função é um PARABOLOIDE (tigela) com mínimo global em (0, 0).
  - CURVAS DE NÍVEL/CONTORNO: linhas que unem os pontos onde a função tem o
    MESMO valor. No paraboloide, são círculos concêntricos: quanto mais perto
    do centro, menor o valor de f.
  - GRADIENTE DESCENDENTE: partimos de um ponto e, a cada iteração, andamos
    na direção oposta ao gradiente: (x, y) ← (x, y) − lr·∇f(x, y).
  - A trajetória mostrada no gráfico revela:
      * lr baixo  → passos minúsculos, caminho lento porém reto
      * lr ideal  → passos grandes, chega rápido ao centro
      * lr alto   → passos grandes DEMAIS, o algoritmo "pula" o mínimo e
                    fica ziguezagueando (ora para um lado, ora para o outro)
"""

import numpy as np
import matplotlib.pyplot as plt


# ─── PASSO 1: FUNÇÃO DE CUSTO E SEU GRADIENTE ─────────────────────────────────
def f(x, y):
    # f(x, y) = x² + 5 + y²
    #
    # MATEMÁTICA: paraboloide simétrico; curvas de nível = círculos.
    # Mínimo em (0, 0), onde f = 5. Convexa → único mínimo global.
    return x**2 + 5 + y**2

def grad_f(x, y):
    # ∇f = ( ∂f/∂x , ∂f/∂y ) = ( 2x , 10y )
    #
    #   ∂f/∂x = 2x   → derivada de x² em relação a x (o resto é constante)
    #   ∂f/∂y = 10y  → aqui o fator 10 "engrossa" a derivada de y² (que seria
    #                  2y) de propósito, para amplificar o efeito do passo na
    #                  coordenada y e tornar as oscilações dos gráficos visíveis.
    #                  Serve apenas ao propósito didático, não muda a direção
    #                  da descida.
    return 2 * x, 10 * y


# ─── PASSO 2: SIMULAÇÃO DO CAMINHO DO GRADIENTE DESCENDENTE ───────────────────
def obter_caminho_gd(lr, x_init=4.0, y_init=4.0, passos=15):
    # lr: taxa de aprendizado (tamanho do passo).
    # x_init, y_init: ponto de partida (4, 4).
    # passos: quantas iterações de atualização serão feitas.
    x, y = x_init, y_init
    caminho = [(x, y)]  # lista que guarda os pontos visitados (histórico)

    for _ in range(passos):
        # Calcula o gradiente no ponto atual (a direção de MAIOR subida)
        gx, gy = grad_f(x, y)

        # Atualização: andamos na direção OPOSTA ao gradiente (−∇f)
        #   x ← x − lr·2x  = x·(1 − 2·lr)
        #   y ← y − lr·10y = y·(1 − 10·lr)
        #
        # MATEMÁTICA: se o fator multiplicador (1 − 2·lr) tiver módulo < 1,
        # a distância ao mínimo diminui a cada passo (converge). Se o módulo
        # for > 1, a distância AUMENTA (diverge). Se for NEGATIVO, o ponto
        # pulará para o outro lado do eixo (causa o ziguezague!).
        x = x - lr * gx
        y = y - lr * gy

        # Registra o novo ponto na trajetória
        caminho.append((x, y))

    # Converte a lista de tuplas em um array Nx2 para facilitar o plot:
    # caminho[:, 0] = todas as coordenadas X  /  caminho[:, 1] = todas as Y
    return np.array(caminho)


# ─── PASSO 3: CRIAÇÃO DA MALHA PARA AS CURVAS DE NÍVEL ────────────────────────
# np.linspace(-5, 5, 400) → 400 valores igualmente espaçados entre -5 e 5.
x_val = np.linspace(-5, 5, 400)
y_val = np.linspace(-5, 5, 400)

# np.meshgrid gera todas as combinações (x, y) da grade:
# X e Y são matrizes 400x400 em que X[i,j] = x_val[i] e Y[i,j] = y_val[j],
# formando uma "malha" de pontos no plano XY.
X, Y = np.meshgrid(x_val, y_val)

# Avalia a função em TODOS os pontos da malha de uma só vez (vetorização).
# Z é uma matriz 400x400 com o valor de f em cada ponto da grade —
# usada para desenhar as curvas de nível.
Z = f(X, Y)


# ─── PASSO 4: SIMULAÇÃO DE TRÊS CENÁRIOS DE TAXA DE APRENDIZADO ───────────────
# Cada caminho tem 16 pontos (início + 15 iterações).

# lr = 0.02 → taxa BAIXA.
#   Fatores: x·(1 − 0.04) = x·0.96 → encolhe 4% por passo;
#            y·(1 − 0.2)  = y·0.80 → encolhe 20% por passo.
#   Passos curtos, trajetória reta e lenta (não alcança o centro em 15 passos).
caminho_lento = obter_caminho_gd(lr=0.02)

# lr = 0.12 → taxa "ideal" (escolha equilibrada).
#   Fatores: x·(1 − 0.24) = x·0.76 → encolhe 24% por passo;
#            y·(1 − 1.20) = y·(−0.20) → MÓDULO 0.20 < 1, mas NEGATIVO;
#   em y o ponto alterna de lado do eixo toda iteração, porém encolhendo,
#   chegando quase no fundo em poucos passos.
caminho_otimo = obter_caminho_gd(lr=0.12)

# lr = 0.19 → taxa ALTA (ziguezague forte).
#   Fatores: x·(1 − 0.38) = x·0.62;
#            y·(1 − 1.90) = y·(−0.90) → módulo 0.90 < 1 mas próximo de 1:
#   y oscila violentamente de um lado para o outro (quase explode), criando
#   o padrão ziguezague no gráfico.
caminho_oscilante = obter_caminho_gd(lr=0.19)


# ─── PASSO 5: CONSTRUÇÃO DOS TRÊS GRÁFICOS LADO A LADO ────────────────────────
# plt.subplots(1, 3) → 1 linha, 3 colunas de gráficos.
# fig = a figura inteira; axes = lista com os 3 subgráficos.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Títulos de cada subgráfico. $...$ ativa a escrita matemática (LaTeX) do matplotlib.
titulos = [
    "Taxa Baixa ($\\alpha=0.02$\nPasso curtos e lentos",
    "Taxa Ideal ($\\alpha=0.12$)\nDireto ao ponto mínimo",
    "Taxa Alta ($\\alpha=0.19$)\nOscilação e ziguezague"
]

# As três trajetórias e suas cores (vermelho, verde, laranja)
caminhos = [caminho_lento, caminho_otimo, caminho_oscilante]
cores = ['#e74c3c', '#2ecc71', '#f39c12']

# zip junta os três conjuntos: para cada (gráfico, caminho, título, cor)...
for ax, caminho, titulo, cor in zip(axes, caminhos, titulos, cores):
    # ── Curvas de nível ──
    # ax.contour desenha as linhas onde f(x,y) é constante.
    # levels=15 → desenha 15 círculos concêntricos.
    # cmap='viridis' → paleta de cores do fundo. alpha=0.6 → transparência.
    contorno = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)

    # Coloca os valores de f DENTRO das curvas de nível (ex.: 5, 10, 20...)
    ax.clabel(contorno, inline=True, fontsize=8)

    # ── Trajetória do GD ──
    # Conecta os pontos visitados com uma linha marcada por círculos, mostrando
    # como o algoritmo "anda" pela superfície.
    ax.plot(caminho[:, 0], caminho[:, 1], color=cor, marker='o', linewidth=2, markersize=5, label="Trajetória GD")

    # ── Marcadores especiais ──
    # Ponto inicial (4, 4) em vermelho; mínimo global (0, 0) em asterisco preto.
    ax.plot(caminho[0, 0], caminho[0, 1], 'ro', markersize=8, label='Início (4, 4)')
    ax.plot(0, 0, 'k*', markersize=12, label='Mínimo (0, 0)')

    # ── Formatação do gráfico ──
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.grid(True, linestyle='--', alpha=0.5)   # grade tracejada suave
    ax.legend(loc='upper right')               # legenda no canto superior direito
    ax.set_xlim(-5, 5)                         # faixa de valores do eixo X
    ax.set_ylim(-5, 5)                         # faixa de valores do eixo Y

# Ajusta os espaçamentos para nada ficar sobreposto e exibe os gráficos.
plt.tight_layout()
plt.show()

# ─── O QUE ESPERAR DO GRÁFICO ──────────────────────────────────────────────────
#   1º gráfico (0.02): reta curta descendo devagar, longe do centro (0,0).
#   2º gráfico (0.12): chega quase no centro rapidamente (mas note alguns
#      "pulos" de lado no eixo Y, causados pelo fator −0.20).
#   3º gráfico (0.19): ziguezague acentuado no eixo Y — o ponto não se aproxima
#      de (0,0) de forma reta; fica "dançando" entre lados opostos.
#
#   REPARO IMPORTANTE: com o fator 10·y, nenhuma das taxas aqui ultrapassa o
#   limiar de divergência (10·lr > 1 → lr > 0.20). Com o gradiente exato 2·y,
#   o limite seria lr > 0.5 na coordenada y. O fator 10 só foi usado para
#   AMPLIFICAR visualmente o ziguezague.