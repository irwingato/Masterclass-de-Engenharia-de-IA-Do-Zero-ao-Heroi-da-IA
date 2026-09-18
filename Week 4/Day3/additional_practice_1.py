import numpy as np
import scipy.stats as stats

# Objetivo: criar ICs de 95% para OUTRAS estatísticas além da média: desvio-padrão, proporção e mediana.
# Ideia geral de IC: intervalo que, se repetíssemos o experimento muitas vezes, conteria o parâmetro
# verdadeiro em ~95% das vezes.

# Semente fixa para reprodutibilidade: sempre gera os mesmos 30 valores.
np.random.seed(42)

# Simula 30 observações de uma Normal(5.0, 1.5). Ex: salários, notas, medidas.
dados = np.random.normal(loc=5.0, scale=1.5, size=30)
n = len(dados)

print(f"Tamanho da amostra: {n}\n")

# =========================================================================
# 1. IC PARA O DESVIO PADRÃO (Utiliza a Distribuição Qui-Quadrado)
# Matemática: (n-1)*s²/σ² ~ Qui² com df=n-1. Invertendo:
#   IC(σ²) = [(n-1)s² / qui2_sup, (n-1)s² / qui2_inf]
# Note a inversão: o quantil superior vai no denominador do limite inferior.
# Depois tira a raiz para voltar de variância para desvio-padrão.
# =========================================================================
# Variância amostral com ddof=1 (divide por n-1, correção de Bessel).
variancia = np.var(dados, ddof=1)
# Quantis 2.5% e 97.5% da Qui² com df=29. Deixam 95% no centro.
qui2_inferior = stats.chi2.ppf(0.025, df=n-1)
qui2_superior = stats.chi2.ppf(0.975, df=n-1)

# Limites para a variância e depois extraindo a raiz para o desvio padrão
ic_var = ((n-1)* variancia / qui2_superior, (n-1)*variancia / qui2_inferior)
ic_std = (np.sqrt(ic_var[0]), np.sqrt(ic_var[1]))

print(f"Desvio Padrão Amostral: {np.sqrt(variancia):.3f}")
print(f"IC 95% do Desvio Padrão: ({ic_std[0]:.3f}, {ic_std[1]:.3f})\n")


# =========================================================================
# 2. IC PARA UMA PROPORÇÃO (Método de Wald / Aproximação Normal)
# Matemática: p_hat ± z * SE, onde SE = sqrt(p_hat*(1-p_hat)/n), z≈1.96.
# Vale quando n*p_hat e n*(1-p_hat) >= 5 (aproximação Normal da Binomial).
# =========================================================================
# Suponha que queremos estimar a proporção de valores na população maiores que 5.0
# Conta quantos dos 30 valores passam de 5.0 = número de sucessos.
sucessos = np.sum(dados > 5.0)
# Proporção amostral: sucessos / n.
p_hat = sucessos / n
# Erro-padrão da proporção: mede a incerteza de p_hat. Cai com sqrt(n).
erro_padrao_prop = np.sqrt(p_hat * (1 - p_hat) / n)

# norm.interval(0.95, loc=p_hat, scale=SE) = p_hat ± 1.96*SE automaticamente.
ic_prop = stats.norm.interval(0.95, loc=p_hat, scale=erro_padrao_prop)

print(f"Proporção Amostral (valores > 5.0): {p_hat:.3f}")
print(f"IC 95% da Proporção: ({ic_prop[0]:.3f}, {ic_prop[1]:.3f})\n")


# =========================================================================
# 3. IC PARA A MEDIANA (Método Não-Paramétrico: Bootstrapping)
# Matemática: sem fórmula teórica simples, então estima a distribuição da mediana
# por reamostragem: sorteia n valores COM reposição 1000x, calcula a mediana de cada,
# e pega os percentis 2.5 e 97.5 como limites (método percentil).
# =========================================================================
# Útil quando a estatística não possui uma distribuição teórica simples
boot_medianas = []
for _ in range(1000):
    # Reamostra do próprio dado, com reposição, mesmo tamanho n.
    amostra_boot = np.random.choice(dados, size=n, replace=True)
    boot_medianas.append(np.median(amostra_boot))

# Coletando os percentis 2.5 e 97.5 para fechar os 95% de confiança
ic_mediana = np.percentile(boot_medianas, [2.5, 97.5])

print(f"Mediana Amostral: {np.median(dados):.3f}")
print(f"IC 95% da Mediana (Bootstrap): ({ic_mediana[0]:.3f}, {ic_mediana[1]:.3f})\n")
