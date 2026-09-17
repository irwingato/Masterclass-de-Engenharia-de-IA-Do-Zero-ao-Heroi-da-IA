"""
day2_samples.py — CATÁLOGO VISUAL das 4 distribuições fundamentais
==================================================================
OBJETIVO:
    Desenhar lado a lado 1 distribuição CONTÍNUA simétrica (Gaussiana),
    2 DISCRETAS de contagem (Binomial e Poisson) e 1 CONTÍNUA sem forma
    (Uniforme amostrada + histograma), para fixar PDF vs PMF vs amostragem.

MATEMÁTICA DAS 4:
    1) GAUSSIANA N(mu, sigma²) — contínua, sino simétrico:
       PDF: f(x) = 1/(sigma*sqrt(2*pi)) * exp(-(x-mu)²/(2*sigma²)).
       Aqui mu=0, sigma=1 (Normal padrão). skew=0, kurt=0 (referência).

    2) BINOMIAL Bin(n,p) — discreta, "k sucessos em n tentativas":
       PMF: P(X=k) = C(n,k) * p^k * (1-p)^(n-k), k=0..n.
       Aqui n=10, p=0.5 (10 moedas justas). Média = n*p = 5,
       simétrica porque p=0.5. Se p≠0.5, fica assimétrica.

    3) POISSON Pois(lambda) — discreta, "eventos por intervalo":
       PMF: P(X=k) = e^-lambda * lambda^k / k!, k=0,1,2,...
       Aqui lambda=3 (média 3 eventos). Assimétrica à direita por natureza
       (não tem valor negativo, mas tem cauda infinita à direita).
       Quando lambda cresce (~20+), vira quase Gaussiana (TCL).

    4) UNIFORME U(a,b) — contínua, "todos iguais":
       PDF: f(x) = 1/(b-a) para a<=x<=b. Aqui a=0, b=10.
       Diferente das 3 acima (que desenham a fórmula exata com pdf/pmf),
       aqui AMOSTRAMOS 1000 pontos e o histograma + KDE reconstrói a forma
       retangular. skew=0, kurt=-1.2 (achatada = platicúrtica).

    PDF (contínua) vs PMF (discreta): PDF é curva (prob. = área), PMF é
    barra (prob. = altura). Por isso Gaussiana usa plt.plot e as discretas
    usam plt.bar.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson, uniform
import seaborn as sns

# Gaussian Distribution: desenha a PDF exata, não amostra.
# x de -4 a 4 cobre mu ± 4*sigma (99.99% da massa está em ±3sigma).
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, loc=0, scale=1), label="Gaussian (u=0, s=1)")

# Binomial Distribution: PMF exata para k=0..10 (n+1 valores possíveis).
# alpha=0.7 deixa transparente para barras sobrepostas se misturarem.
n, p = 10, 0.5
x = np.arange(n + 1)
plt.bar(x, binom.pmf(x, n, p), alpha=0.7, label="Binomial (n=10, p=0.5)")

# Poisson Distribution: PMF exata para k=0..9 (corta a cauda infinita;
# P(X>=10 | lambda=3) é desprezível, <0.1%).
lam = 3
x = np.arange(0, 10)
plt.bar(x, poisson.pmf(x, lam), alpha=0.7, label="Poisson (l=3)")

# Uniform Distribution: aqui é AMOSTRAGEM (Monte Carlo), não fórmula.
# Gera 1000 pontos U(0,10); o histograma deve ficar "retangular" ≈ 0.1.
# kde=True desenha a densidade estimada — deve ficar plana em ~0.1 = 1/(10-0).
x = np.random.uniform(low=0, high=10, size=1000)
sns.histplot(x, kde=True, label="Uniform", color="red")


plt.title("Probability Distributions")
plt.legend()
plt.show()
