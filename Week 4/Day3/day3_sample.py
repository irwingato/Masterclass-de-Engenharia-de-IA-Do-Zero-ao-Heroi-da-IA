import numpy as np
from scipy.stats import norm, t

# Objetivo: calcular o Intervalo de Confiança (IC) de 95% para a média de uma amostra pequena.
# Matemática: IC = media ± margem, onde margem = valor_critico * (s / sqrt(n))
# - media (x-barra): estimativa pontual da média populacional
# - s / sqrt(n): erro-padrão, mede a incerteza da média. Diminui quando n cresce.
# - valor_critico: quantil que deixa 95% da distribuição no centro.

# Dados amostrais fixos
data = [12, 14, 15, 16, 17, 18, 19]

# Média amostral: soma / n. Aqui ~15.857. É o centro do intervalo.
mean = np.mean(data)
# Desvio-padrão amostral com ddof=1 (divide por n-1, correção de Bessel).
# Usamos n-1 porque estimamos a média a partir da própria amostra.
std = np.std(data, ddof=1)

# Como n=7 (<30) e o desvio populacional é desconhecido, usamos a t-Student e não a Normal.
# A t tem caudas mais pesadas, gerando um intervalo mais largo e conservador.
n = len(data)
# Quantil 97.5% da t com df=n-1=6. Deixa 2.5% em cada cauda = 95% no centro.
# Aqui vale ~2.447 (maior que 1.96 da Normal, por isso o intervalo abre mais).
t_value = t.ppf(0.975, df=n-1)
# Margem de erro = quanto andamos para cada lado da média.
margin_of_error = t_value * (std / np.sqrt(n))
# OBS versão NumPy: o professor usa NumPy <2.0, que imprime (13.62..., 18.08...).
# Meu ambiente usa NumPy 2.5.2, onde repr(np.float64(x)) virou np.float64(x),
# então sem o float() imprimia (np.float64(...), np.float64(...)).
# O float() converte de escalar NumPy para float Python puro só para a exibição ficar igual.
# O valor matemático não muda.
ci = (float(mean - margin_of_error), float(mean + margin_of_error))
print("95% Confidence Interval:", ci)
