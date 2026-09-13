"""
day5_samples.py
===============
OBJETIVO: Arquivo de AMOSTRAS (rascunho didático) com 4 distribuições clássicas.

1. GAUSSIANA manual: calcula a fórmula do sino na mão (sem scipy).
2. BERNOULLI: 1 única tentativa com 2 resultados (0=falha, 1=sucesso, p=0.6).
3. BINOMIAL: n=10 repetições de Bernoulli com p=0.5 (nº de sucessos 0..10).
4. POISSON: contagem de eventos por intervalo com média λ=3.

No original, só o Poisson executava; os outros 3 estavam comentados.
Aqui os 4 foram ativados em grade 2x2 para estudo comparativo.
"""

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
import numpy as np  # Cria eixos X e calcula a fórmula matemática da Gaussiana
import matplotlib.pyplot as plt  # Cria a grade 2x2 de gráficos (linhas e barras)
from scipy.stats import binom  # Calcula a Binomial via binom.pmf (massa discreta)
from scipy.stats import poisson  # Calcula a Poisson via poisson.pmf (massa discreta)

# --- 2. CRIAÇÃO DA FIGURA EM GRADE 2x2 ---
# 2 linhas x 2 colunas = 4 posições: [0,0]=Gauss, [0,1]=Bernoulli, [1,0]=Binomial, [1,1]=Poisson
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 3. GRÁFICO [0,0]: GAUSSIANA COM FÓRMULA MANUAL ---
mu, sigma = 0, 1  # mu=0 é a média (centro), sigma=1 é o desvio-padrão (largura)

# np.linspace(-4, 4, 100): 100 pontos contínuos de -4 a 4 (cobre ±4 desvios = 99.99% da massa)
x_gauss = np.linspace(-4, 4, 100)

# Fórmula da densidade Normal calculada NA MÃO (equivalente a scipy.stats.norm.pdf):
# y = 1/sqrt(2*pi*sigma²) * exp(-0.5 * ((x-mu)/sigma)²)
# - 1/sqrt(...) normaliza para área total = 1
# - exp(...) dá o formato de sino centrado em mu
y_gauss = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-0.5 * ((x_gauss - mu) / sigma)**2)

axes[0, 0].plot(x_gauss, y_gauss, color="royalblue", linewidth=2.5)  # Curva contínua azul
axes[0, 0].set_title("Gaussian Distribution\nFórmula manual N(0,1)")  # Título
axes[0, 0].set_xlabel("Valores (x)")  # Eixo X: valores contínuos
axes[0, 0].set_ylabel("Densidade pdf(x)")  # Eixo Y: densidade
axes[0, 0].grid(alpha=0.3)  # Grade suave

# --- 4. GRÁFICO [0,1]: BERNOULLI (1 tentativa, 2 resultados) ---
p = 0.6  # p=0.6 = prob. de sucesso (1); logo 1-p=0.4 = prob. de falha (0)

# plt.bar([0,1], [1-p, p]): duas barras — altura 0.4 em x=0 e 0.6 em x=1
axes[0, 1].bar([0, 1], [1 - p, p], color="blue", alpha=0.7)
axes[0, 1].set_title(f"Bernoulli Distribution\n1 tentativa: p={p}")  # Título com p
axes[0, 1].set_xlabel("Resultado")  # Eixo X: 0 ou 1
axes[0, 1].set_ylabel("Probabilidade")  # Eixo Y: 0.4 e 0.6 (soma = 1)
axes[0, 1].set_xticks([0, 1])  # Garante só as marcas 0 e 1 no eixo X
axes[0, 1].set_xticklabels(["0 (Failure)", "1 (Success)"])  # Troca 0/1 por rótulos legíveis
axes[0, 1].grid(axis="y", alpha=0.3)  # Grade no eixo Y

# --- 5. GRÁFICO [1,0]: BINOMIAL (10 tentativas, p=0.5) ---
n, p_binom = 10, 0.5  # n=10 moedas, p=0.5 chance de cara em cada uma

# np.arange(n+1): inteiros [0..10] = nº possíveis de caras
x_binom = np.arange(n + 1)

# binom.pmf(x, n, p): P(X=x) = C(n,x) * p^x * (1-p)^(n-x), calculado pelo scipy
y_binom = binom.pmf(x_binom, n, p_binom)

axes[1, 0].bar(x_binom, y_binom, color="green", alpha=0.7)  # Barras verdes discretas
axes[1, 0].set_title(f"Binomial Distribution\nn={n}, p={p_binom}")  # Título com parâmetros
axes[1, 0].set_xlabel("Nº de sucessos (k)")  # Eixo X: 0 a 10 sucessos
axes[1, 0].set_ylabel("Probabilidade pmf(k)")  # Eixo Y: prob. de cada k
axes[1, 0].set_xticks(x_binom)  # Marca todos os inteiros 0..10
axes[1, 0].grid(axis="y", alpha=0.3)  # Grade no eixo Y

# --- 6. GRÁFICO [1,1]: POISSON (média λ=3) ---
lam = 3  # lambda=3 = nº médio de eventos por intervalo (ex: 3/hora)

# np.arange(0, 10): inteiros [0..9] = contagens possíveis no intervalo
x_pois = np.arange(0, 10)

# poisson.pmf(x, lam): P(X=x) = (lam^x * e^-lam) / x!, calculado pelo scipy
y_pois = poisson.pmf(x_pois, lam)

axes[1, 1].bar(x_pois, y_pois, color="orange", alpha=0.7)  # Barras laranjas (único ativo no original)
axes[1, 1].set_title(f"Poisson Distribution\nλ={lam}")  # Título com lambda
axes[1, 1].set_xlabel("Nº de eventos")  # Eixo X: quantos eventos ocorreram
axes[1, 1].set_ylabel("Probabilidade pmf(k)")  # Eixo Y: prob. de cada contagem
axes[1, 1].set_xticks(x_pois)  # Marca os inteiros 0..9
axes[1, 1].grid(axis="y", alpha=0.3)  # Grade no eixo Y

# --- 7. AJUSTE FINAL E EXIBIÇÃO ---
plt.tight_layout()  # Ajusta margens para os 4 títulos não colidirem
plt.show()  # Abre a janela com a grade 2x2
