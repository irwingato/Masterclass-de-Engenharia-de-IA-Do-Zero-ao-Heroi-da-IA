# url  = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

import pandas as pd
from scipy.stats import norm
import numpy as np

# Objetivo: calcular o IC de 95% para a média real (sepal_length da Iris) a partir de uma subamostra.
# Matemática: IC = media ± margem, onde margem = z * (s / sqrt(n))

# Carrega o dataset Iris direto da internet (150 linhas, 4 medidas + espécie).
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Sorteia 30 linhas da coluna sepal_length com semente fixa (random_state=42).
# Semente fixa = resultado reproduzível, sempre os mesmos 30 valores.
sample = df["sepal_length"].sample(30, random_state=42)

# Média amostral da subamostra: centro do intervalo.
mean = sample.mean()
# Desvio-padrão amostral do pandas (já usa ddof=1 por padrão, divide por n-1).
std = sample.std()
# Tamanho da subamostra: n=30, o limite clássico para aproximar pela Normal.
n = len(sample)

# Usa a Normal padrão porque n=30 já é considerado suficiente para o Teorema Central do Limite.
# Quantil 97.5%: ~1.96.
z_value = norm.ppf(0.975)
# Margem de erro = 1.96 * erro-padrão.
margin_of_error = z_value * (std / np.sqrt(n))
# OBS versão NumPy/pandas: o professor usa NumPy <2.0, que imprime (x, y) limpo.
# Meu ambiente usa NumPy 2.5.2, onde repr(np.float64(x)) virou np.float64(x),
# então sem o float() imprimia (np.float64(...), np.float64(...)).
# O float() converte para float Python puro só para a exibição ficar igual. Não muda a conta.
ci = (float(mean - margin_of_error), float(mean + margin_of_error))

print("Sample Mean: ", mean)
print("95% Confidence Interval:", ci)
