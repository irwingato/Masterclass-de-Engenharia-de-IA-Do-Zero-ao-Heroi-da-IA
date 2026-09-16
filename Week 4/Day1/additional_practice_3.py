# ============================================================================
# additional_practice_3.py
# OBJETIVO: Explorar OUTRAS DISTRIBUIÇÕES de probabilidade além do dado/moeda:
#   1. NORMAL (contínua, curva do sino) — ex.: alturas, erros de medição.
#   2. BINOMIAL (discreta, nº de sucessos em n tentativas) — ex.: caras em 10 moedas.
#   3. POISSON (discreta, nº de eventos raros por intervalo) — ex.: clientes/minuto.
#   4. UNIFORME (contínua, mesma chance em todo intervalo) — ex.: sorteio entre 0 e 1.
#   5. EXPONENCIAL (contínua, tempo entre eventos) — ex.: tempo até a próxima chegada.
#
# ESTRATÉGIA: para cada distribuição mostramos (a) a TEORIA via scipy.stats
# (PMF/PDF, média e variância teóricas) e (b) a PRÁTICA via simulação Monte Carlo
# com numpy (média e variância amostrais). Pela Lei dos Grandes Números, os dois
# devem convergir quando simulamos 10.000 amostras.
# No final, geramos gráficos comparando histograma simulado x curva teórica.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- CONFIGURAÇÃO GERAL ---
# seed fixa => resultados reproduzíveis (mesmos números a cada execução).
# N = tamanho da amostra simulada (10 mil, coerente com os practices 1 e 2).
np.random.seed(42)
N = 10_000

print("=" * 60)
print("EXPLORAÇÃO DE DISTRIBUIÇÕES DE PROBABILIDADE")
print("=" * 60)

# ==========================================================================
# 1. DISTRIBUIÇÃO NORMAL: X ~ N(mu=170, sigma=10) — ex.: altura em cm
# PDF (função densidade): f(x) = 1/(σ√2π) * exp(-(x-μ)² / 2σ²)
# Teoria: E(X) = μ = 170 | Var(X) = σ² = 100
# ==========================================================================
mu, sigma = 170, 10  # média e desvio-padrão escolhidos para o exemplo

# (a) TEORIA: usamos scipy.stats.norm para média/variância exatas e a PDF.
media_teorica_normal = stats.norm.mean(loc=mu, scale=sigma)
var_teorica_normal = stats.norm.var(loc=mu, scale=sigma)

# (b) PRÁTICA: simulamos N amostras com numpy (Monte Carlo).
amostras_normal = np.random.normal(loc=mu, scale=sigma, size=N)
media_amostral_normal = np.mean(amostras_normal)      # deve ficar ≈ 170
var_amostral_normal = np.var(amostras_normal, ddof=0)  # deve ficar ≈ 100

print("\n--- 1) NORMAL N(170, 10) ---")
print(f"Teórica : média = {media_teorica_normal:.2f} | variância = {var_teorica_normal:.2f}")
print(f"Simulada: média = {media_amostral_normal:.2f} | variância = {var_amostral_normal:.2f}")
# Ex.: P(X > 180) — probabilidade teórica de alguém ter mais de 1,80 m.
p_maior_180 = stats.norm.sf(180, loc=mu, scale=sigma)  # sf = 1 - CDF (sobrevivência)
print(f"P(X > 180) teórica = {p_maior_180:.2%} | simulada = {np.mean(amostras_normal > 180):.2%}")

# ==========================================================================
# 2. DISTRIBUIÇÃO BINOMIAL: X ~ Bin(n=10, p=0.5) — nº de caras em 10 moedas
# PMF: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
# Teoria: E(X) = n*p = 5 | Var(X) = n*p*(1-p) = 2.5
# ==========================================================================
n, p = 10, 0.5  # 10 tentativas independentes, 50% de sucesso cada

# (a) TEORIA: PMF para cada k = 0..10 e momentos exatos.
k_binom = np.arange(0, n + 1)
pmf_binom = stats.binom.pmf(k_binom, n, p)  # probabilidade teórica de cada k
media_teorica_binom = stats.binom.mean(n, p)
var_teorica_binom = stats.binom.var(n, p)

# (b) PRÁTICA: cada amostra = contar caras em 10 lançamentos; repetimos N vezes.
amostras_binom = np.random.binomial(n=n, p=p, size=N)
media_amostral_binom = np.mean(amostras_binom)
var_amostral_binom = np.var(amostras_binom, ddof=0)

print("\n--- 2) BINOMIAL Bin(n=10, p=0.5) ---")
print(f"Teórica : média = {media_teorica_binom:.2f} | variância = {var_teorica_binom:.2f}")
print(f"Simulada: média = {media_amostral_binom:.2f} | variância = {var_amostral_binom:.2f}")
print(f"P(X = 5) teórica = {stats.binom.pmf(5, n, p):.2%} | "
      f"simulada = {np.mean(amostras_binom == 5):.2%}")

# ==========================================================================
# 3. DISTRIBUIÇÃO DE POISSON: X ~ Poisson(lambda=3) — eventos por intervalo
# Ex.: nº de clientes que chegam por minuto, com média de 3/min.
# PMF: P(X=k) = (λ^k * e^-λ) / k!
# Teoria: E(X) = λ = 3 | Var(X) = λ = 3 (média = variância!)
# ==========================================================================
lam = 3  # taxa média de ocorrência por intervalo

# (a) TEORIA
k_pois = np.arange(0, 12)  # olhamos k = 0..11 (cobre quase toda a massa)
pmf_pois = stats.poisson.pmf(k_pois, lam)
media_teorica_pois = stats.poisson.mean(lam)
var_teorica_pois = stats.poisson.var(lam)

# (b) PRÁTICA
amostras_pois = np.random.poisson(lam=lam, size=N)

print("\n--- 3) POISSON Poisson(lambda=3) ---")
print(f"Teórica : média = {media_teorica_pois:.2f} | variância = {var_teorica_pois:.2f}")
print(f"Simulada: média = {np.mean(amostras_pois):.2f} | variância = {np.var(amostras_pois, ddof=0):.2f}")
print(f"P(X = 0) teórica = {stats.poisson.pmf(0, lam):.2%} | "
      f"simulada = {np.mean(amostras_pois == 0):.2%} (prob. de minuto vazio)")

# ==========================================================================
# 4. DISTRIBUIÇÃO UNIFORME (contínua): X ~ U(a=0, b=10)
# PDF: f(x) = 1/(b-a) dentro de [a, b], zero fora. Toda faixa tem igual chance.
# Teoria: E(X) = (a+b)/2 = 5 | Var(X) = (b-a)²/12 ≈ 8.33
# ==========================================================================
a, b = 0, 10

# (a) TEORIA
media_teorica_unif = stats.uniform.mean(loc=a, scale=b - a)
var_teorica_unif = stats.uniform.var(loc=a, scale=b - a)

# (b) PRÁTICA
amostras_unif = np.random.uniform(low=a, high=b, size=N)

print("\n--- 4) UNIFORME U(0, 10) ---")
print(f"Teórica : média = {media_teorica_unif:.2f} | variância = {var_teorica_unif:.2f}")
print(f"Simulada: média = {np.mean(amostras_unif):.2f} | variância = {np.var(amostras_unif, ddof=0):.2f}")

# ==========================================================================
# 5. DISTRIBUIÇÃO EXPONENCIAL: X ~ Exp(escala=2.0) — tempo entre chegadas
# Ex.: tempo médio de 2 min entre clientes. Parametrização do numpy/scipy por
# 'scale' = 1/λ (média). PDF: f(x) = λ * e^(-λx), para x >= 0.
# Teoria: E(X) = scale = 2 | Var(X) = scale² = 4
# ==========================================================================
scale_exp = 2.0  # tempo médio entre eventos (equivale a λ = 0.5 por minuto)

# (a) TEORIA
media_teorica_exp = stats.expon.mean(scale=scale_exp)
var_teorica_exp = stats.expon.var(scale=scale_exp)

# (b) PRÁTICA
amostras_exp = np.random.exponential(scale=scale_exp, size=N)

print("\n--- 5) EXPONENCIAL Exp(média=2.0) ---")
print(f"Teórica : média = {media_teorica_exp:.2f} | variância = {var_teorica_exp:.2f}")
print(f"Simulada: média = {np.mean(amostras_exp):.2f} | variância = {np.var(amostras_exp, ddof=0):.2f}")
print(f"P(X > 5) teórica = {stats.expon.sf(5, scale=scale_exp):.2%} | "
      f"simulada = {np.mean(amostras_exp > 5):.2%} (esperar > 5 min)")

# ==========================================================================
# 6. GRÁFICOS: histograma simulado (empírico) x curva/pontos teóricos
# - Distribuições contínuas (Normal, Uniforme, Exponencial): histograma com
#   density=True + linha da PDF teórica.
# - Distribuições discretas (Binomial, Poisson): frequência relativa simulada
#   (barras) + marcadores da PMF teórica.
# ==========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Explorando Distribuições: Simulação (10 mil amostras) x Teoria", fontsize=13)
axes = axes.ravel()  # achata a grade 2x3 numa lista de 6 eixos (usaremos 5)

# -- Gráfico 1: Normal --
x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)  # grade para a PDF
axes[0].hist(amostras_normal, bins=40, density=True, alpha=0.6, label="Simulado")
axes[0].plot(x_norm, stats.norm.pdf(x_norm, loc=mu, scale=sigma), "r-", lw=2, label="PDF teórica")
axes[0].set_title("Normal N(170, 10)")
axes[0].legend()

# -- Gráfico 2: Binomial --
freq_binom = [np.mean(amostras_binom == k) for k in k_binom]  # freq. relativa simulada
axes[1].bar(k_binom - 0.2, freq_binom, width=0.4, alpha=0.6, label="Simulado")
axes[1].bar(k_binom + 0.2, pmf_binom, width=0.4, alpha=0.6, label="PMF teórica")
axes[1].set_title("Binomial Bin(10, 0.5)")
axes[1].set_xticks(k_binom)
axes[1].legend()

# -- Gráfico 3: Poisson --
freq_pois = [np.mean(amostras_pois == k) for k in k_pois]
axes[2].bar(k_pois - 0.2, freq_pois, width=0.4, alpha=0.6, label="Simulado")
axes[2].bar(k_pois + 0.2, pmf_pois, width=0.4, alpha=0.6, label="PMF teórica")
axes[2].set_title("Poisson(λ=3)")
axes[2].set_xticks(k_pois)
axes[2].legend()

# -- Gráfico 4: Uniforme --
x_unif = np.linspace(a, b, 300)
axes[3].hist(amostras_unif, bins=30, density=True, alpha=0.6, label="Simulado")
axes[3].plot(x_unif, stats.uniform.pdf(x_unif, loc=a, scale=b - a), "r-", lw=2, label="PDF teórica")
axes[3].set_title("Uniforme U(0, 10)")
axes[3].legend()

# -- Gráfico 5: Exponencial --
x_exp = np.linspace(0, np.quantile(amostras_exp, 0.99), 300)  # até o percentil 99 p/ enxugar a cauda
axes[4].hist(amostras_exp, bins=40, density=True, alpha=0.6, label="Simulado")
axes[4].plot(x_exp, stats.expon.pdf(x_exp, scale=scale_exp), "r-", lw=2, label="PDF teórica")
axes[4].set_title("Exponencial (média=2)")
axes[4].legend()

# O 6º painel da grade 2x3 fica vazio — removemos para não poluir a figura.
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

print("\nConclusão: com 10.000 amostras, as estatísticas simuladas ficam muito")
print("próximas das teóricas em todas as 5 distribuições (Lei dos Grandes Números).")
