"""
day2_ex_1.py — Assimetria e Curtose na PRÁTICA (dataset Iris)
=============================================================
OBJETIVO:
    Carregar um dataset real famoso e medir a forma da distribuição de
    UMA feature: 'sepal_length' (comprimento da sépala em cm).

MATEMÁTICA:
    - Skewness = E[(X-mu)³]/sigma³. Diz para onde a cauda aponta.
    - Kurtosis (Fisher) = E[(X-mu)^4]/sigma^4 - 3. Diz o peso das caudas.
      scipy já devolve a versão "em excesso" (Normal = 0).
    - Iris tem 150 flores (50 de cada espécie). 'sepal_length' costuma dar
      skew levemente positivo (algumas flores bem grandes) e kurt próxima
      de 0 / levemente negativa (distribuição razoavelmente "normalzinha",
      mas com mistura de 3 espécies por trás).
"""

from scipy.stats import skew, kurtosis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data: lê o CSV da Iris direto do GitHub (150 linhas, 5 colunas).
# Colunas: sepal_length, sepal_width, petal_length, petal_width, species.
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Analyze sepal_length: extrai a coluna como Série (vetor de 150 valores).
feature = df['sepal_length']
# skew(feature): momento padronizado de ordem 3. >0 = cauda à direita.
print("Skewness: ", skew(feature))
# kurtosis(feature): Fisher (excesso). >0 = caudas pesadas, <0 = achatada.
print("Kurtosis: ", kurtosis(feature))

# Visualize distribution: histograma + KDE (curva de densidade suave).
# Leitura: se a KDE for sino simétrico -> skew~0; se pender p/ direita -> skew>0.
sns.histplot(feature, kde=True)
plt.title("Distribution of Sepal Length")
plt.show()
