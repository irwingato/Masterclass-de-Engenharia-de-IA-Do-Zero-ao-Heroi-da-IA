import numpy as np
from scipy.stats import norm

# Objetivo: calcular o IC de 95% para a média de uma amostra grande simulada.
# Matemática: IC = media ± margem, onde margem = z * (s / sqrt(n))

# Gera 100 valores aleatórios de uma Normal com media 50 e desvio 10.
# Simula coletar uma amostra grande na prática.
data = np.random.normal(loc=50, scale=10, size=100)

# Média amostral: centro do intervalo, deve dar perto de 50.
mean = np.mean(data)
# Desvio-padrão amostral com ddof=1 (correção de Bessel, divide por n-1).
std = np.std(data, ddof=1)
# Tamanho da amostra.
n = len(data)

# Como n=100 (grande), pelo Teorema Central do Limite podemos usar a Normal padrão.
# Não precisa da t-Student, pois s já estima bem o sigma.
# Quantil 97.5% da Normal padrão: ~1.96. Deixa 2.5% em cada cauda = 95% no centro.
z_value = norm.ppf(0.975)
# Margem de erro: erro-padrão (std / sqrt(n)) escalado por 1.96.
margin_of_error = z_value * (std / np.sqrt(n))
# OBS versão NumPy: o professor usa NumPy <2.0, que imprime (x, y) limpo.
# Meu ambiente usa NumPy 2.5.2, onde repr(np.float64(x)) virou np.float64(x).
# O float() converte para float Python puro só para a exibição ficar igual. Não muda a conta.
ci = (float(mean - margin_of_error), float(mean + margin_of_error))

print("Sample Mean: ", mean)
print("95% Confidence Interval:", ci)
