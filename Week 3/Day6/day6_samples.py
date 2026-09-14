"""Cálculo MANUAL (sem NumPy) + IC 95% para a média + bloco comentado de referência.
CÓDIGO ATIVO (linhas 1-14): média/var/desvio na mão e IC com z=1.96.
MATEMÁTICA: x̄=(1/n)Σxᵢ; σ²=(1/n)Σ(xᵢ-x̄)² (populacional); σ=√σ².
  IC: x̄ ± z·EP, EP=σ/√n (erro-padrão da média, cai com 1/√n). z_{0.975}=Φ⁻¹(0.975)≈1.96.
  ATENÇÃO: com n=5 o correto seria t de Student t_{0.975,4}≈2.78 (σ desconhecido);
  usar z=1.96 subestima a largura. Aqui: x̄=30, σ≈14.14, EP≈6.32 -> IC≈[17.6,42.4].
  Bloco comentado mostra mediana (q50, regra par/ímpar) e moda (statistics.mode).
"""
import scipy.stats as stats

data = [10, 20, 30, 40, 50]
# CÓDIGO: média manual = sum/len (equivale a np.mean).
# MATEMÁTICA: (10+20+30+40+50)/5 = 150/5 = 30.
mean = sum(data) / len(data)

# CÓDIGO: variância populacional manual (/len) e desvio = raiz.
# MATEMÁTICA: soma dos quadrados=1000 -> σ²=1000/5=200, σ=√200≈14.14.
variance = sum((x - mean) ** 2 for x in data) / len(data)
std_dev = variance ** 0.5

# CÓDIGO: IC 95% com z=1.96 (aproximação Normal, σ tratado como conhecido).
# MATEMÁTICA: EP=σ/√n≈14.14/√5≈6.32; margem=1.96·EP≈12.4; IC=[30-12.4, 30+12.4].
sample_mean = mean
z_score = 1.96

ci = (sample_mean - z_score * (std_dev / len(data) ** 0.5),
      sample_mean + z_score * (std_dev / len(data) ** 0.5))
print("95% Confidence Interval: ", ci)


# Bloco de referência (comentado): média/mediana/moda/var/desvio sem NumPy.
# MATEMÁTICA mediana: ordena; se n ímpar pega o central, se par média dos 2 centrais.
# MATEMÁTICA moda: statistics.mode = valor mais frequente (aqui todos têm freq 1,
#   então a moda é arbitrária/primeiro valor — por isso o exemplo usa dados sem moda clara).
# from statistics import mode

# data = [10, 20, 30, 40, 50]
# mean = sum(data) / len(data)
# print("Mean:", mean)

# sorted_data = sorted(data)
# median = sorted_data[len(data) // 2] if len(data) % 2 != 0 else \
#     (sorted_data[len(data) // 2 - 1] + sorted_data[len(data) // 2]) / 2
# print("Median: ", median)

# print("Mode: ", mode(data))

# variance = sum((x - mean) ** 2 for x in data) / len(data)
# print("Variance: ", variance)
# std_dev = variance ** 0.5
# print("Standard Deviation: ", std_dev)
