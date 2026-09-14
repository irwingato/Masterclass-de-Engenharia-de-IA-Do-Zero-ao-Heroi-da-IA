"""Estatística descritiva básica com NumPy.
CÓDIGO: calcula média, variância e desvio-padrão de [10,20,30,40,50].
MATEMÁTICA: média x̄=(1/n)Σxᵢ; variância POPULACIONAL σ²=(1/n)Σ(xᵢ-x̄)²;
  desvio σ=√σ² (mesma unidade dos dados). np.var/std usam ddof=0 (populacional).
  Para AMOSTRA (inferência) usa-se s²=(1/(n-1))Σ(xᵢ-x̄)² (ddof=1, correção de Bessel).
  Aqui: x̄=30, σ²=200, σ≈14.14; com ddof=1 seria s²=250, s≈15.81.
"""
import numpy as np

# Dataset
# CÓDIGO: lista fixa, n=5, simétrica em torno de 30.
data = [10, 20, 30, 40, 50]

# Calculate stats
# CÓDIGO: np.mean/var/std aplicam as fórmulas acima com ddof=0.
# MATEMÁTICA: desvios: -20,-10,0,10,20 -> quadrados: 400,100,0,100,400 -> soma=1000.
#   σ²=1000/5=200; σ=√200≈14.14. Variância em unidades², desvio na unidade original.
mean = np.mean(data)
variance = np.var(data)
std_dev = np.std(data)

print("Mean", mean)
print("Variance", variance)
print("Standard Deviation", std_dev)