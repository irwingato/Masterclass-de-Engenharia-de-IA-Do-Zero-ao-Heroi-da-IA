"""
day5_ex2.py
===========
OBJETIVO: Visualizar 3 distribuições de probabilidade fundamentais usando scipy.stats.

1. GAUSSIANA (Normal): dados CONTÍNUOS em forma de sino (ex: altura, erros).
2. BINOMIAL: nº de SUCESSOS em n tentativas discretas (ex: 10 moedas, p=0.5).
3. POISSON: nº de EVENTOS por intervalo (ex: 3 clientes/hora em média).

No arquivo original, Gaussiana e Binomial estavam comentadas e só Poisson executava.
Aqui as 3 foram ativadas lado a lado (1x3 subplots) para comparação didática.
"""

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
import numpy as np  # Gera os eixos X: linspace (contínuo) e arange (discreto)
import matplotlib.pyplot as plt  # Cria a figura, os 3 subplots e as barras/linhas
from scipy.stats import norm, binom, poisson  # norm=Gaussiana, binom=Binomial, poisson=Poisson

# --- 2. CRIAÇÃO DA FIGURA COM 3 GRÁFICOS LADO A LADO ---
# 1 linha x 3 colunas, tamanho 18x5 para cada distribuição ter espaço próprio
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 3. GRÁFICO 1 (esquerda): DISTRIBUIÇÃO GAUSSIANA (Normal padrão) ---
# np.linspace(-4, 4, 100): cria 100 pontos igualmente espaçados de -4 a 4 no eixo X.
# É contínua, por isso usamos linspace (e não arange de inteiros).
x_gauss = np.linspace(-4, 4, 100)

# norm.pdf(x, loc=0, scale=1): calcula a DENSIDADE de probabilidade em cada ponto x.
# loc=0 é a média (centro do sino), scale=1 é o desvio-padrão (largura do sino).
y_gauss = norm.pdf(x_gauss, loc=0, scale=1)

# Plota curva contínua (linha) azul com espessura 2.5 no primeiro eixo
axes[0].plot(x_gauss, y_gauss, label="Gaussian", color="royalblue", linewidth=2.5)
axes[0].set_title("Gaussian Distribution\nContínua: N(0,1)")  # Título: nome + tipo + parâmetros
axes[0].set_xlabel("Valores (x)")  # Eixo X: valores contínuos possíveis
axes[0].set_ylabel("Densidade pdf(x)")  # Eixo Y: densidade (área total = 1)
axes[0].legend()  # Mostra a legenda "Gaussian"
axes[0].grid(alpha=0.3)  # Grade suave para facilitar a leitura

# --- 4. GRÁFICO 2 (centro): DISTRIBUIÇÃO BINOMIAL ---
n, p = 10, 0.5  # n=10 tentativas (ex: 10 moedas), p=0.5 prob. de sucesso em cada uma

# np.arange(0, n+1): cria os inteiros [0,1,2,...,10] = nº possíveis de sucessos.
# É discreta, por isso valores inteiros (não faz sentido "2.5 caras em 10 moedas").
x_binom = np.arange(0, n + 1)

# binom.pmf(x, n, p): calcula P(X=x) = prob. de obter EXATAMENTE x sucessos.
# pmf = Probability Mass Function (função de massa, usada para discretas).
y_binom = binom.pmf(x_binom, n, p)

# Plota barras verdes no segundo eixo, uma barra por nº de sucessos
axes[1].bar(x_binom, y_binom, label="Binomial", color="green", alpha=0.7)
axes[1].set_title(f"Binomial Distribution\nDiscreta: n={n}, p={p}")  # Título com parâmetros
axes[1].set_xlabel("Nº de sucessos (k)")  # Eixo X: quantos sucessos (0 a 10)
axes[1].set_ylabel("Probabilidade pmf(k)")  # Eixo Y: probabilidade de cada k (soma = 1)
axes[1].set_xticks(x_binom)  # Marca todos os inteiros 0..10 no eixo X
axes[1].legend()  # Mostra a legenda "Binomial"
axes[1].grid(axis="y", alpha=0.3)  # Grade só no eixo Y

# --- 5. GRÁFICO 3 (direita): DISTRIBUIÇÃO DE POISSON ---
lam = 3  # lambda=3 é a TAXA MÉDIA de eventos por intervalo (ex: 3 chegadas/hora)

# np.arange(0, 10): inteiros [0..9] = nº possíveis de eventos no intervalo.
x_pois = np.arange(0, 10)

# poisson.pmf(x, lam): calcula P(X=x) dado a média lam. Também é discreta (pmf).
y_pois = poisson.pmf(x_pois, lam)

# Plota barras laranjas no terceiro eixo (era o único gráfico ativo no original)
axes[2].bar(x_pois, y_pois, label="Poisson", color="darkorange", alpha=0.7)
axes[2].set_title(f"Poisson Distribution\nDiscreta: λ={lam}")  # Título com lambda
axes[2].set_xlabel("Nº de eventos")  # Eixo X: quantos eventos ocorreram
axes[2].set_ylabel("Probabilidade pmf(k)")  # Eixo Y: probabilidade de cada contagem
axes[2].set_xticks(x_pois)  # Marca os inteiros 0..9 no eixo X
axes[2].legend()  # Mostra a legenda "Poisson"
axes[2].grid(axis="y", alpha=0.3)  # Grade só no eixo Y

# --- 6. AJUSTE FINAL E EXIBIÇÃO ---
plt.tight_layout()  # Ajusta os espaçamentos para os títulos não se sobreporem
plt.show()  # Abre a janela com os 3 gráficos
