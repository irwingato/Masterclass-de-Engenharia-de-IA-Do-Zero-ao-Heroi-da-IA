"""Distribuição Log-Normal + média, mediana e moda.
CÓDIGO: gera 1000 amostras, calcula as 3 medidas e plota histograma + linhas verticais.
MATEMÁTICA: se Y ~ Normal(mu, sigma²), então X = exp(Y) ~ LogNormal(mu, sigma).
  f(x) = 1/(x*sigma*sqrt(2pi)) * exp(-(ln x - mu)²/(2*sigma²)), x > 0.
  Média = exp(mu + sigma²/2) | Mediana = exp(mu) | Moda = exp(mu - sigma²).
  Com assimetria à direita: moda < mediana < média (a média é puxada pela cauda).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Gerar dados fictícios com assimetria (Distribuição Log-Normal)
# CÓDIGO: seed fixa p/ reprodutibilidade; mean/sigma são mu/sigma da Normal subjacente.
# MATEMÁTICA: mu=0.5, sigma=0.5 -> teórica: mediana=exp(0.5)~1.65,
#   média=exp(0.5+0.125)~1.87, moda=exp(0.5-0.25)~1.28.
np.random.seed(42)
dados = np.random.lognormal(mean=0.5, sigma=0.5, size=1000)

# 2. Calcular as medidas de tendência central
# CÓDIGO: mean = soma/n; median = valor central ordenado; mode = valor mais frequente.
# MATEMÁTICA: média x̄=(1/n)Σxᵢ (sensível a outliers); mediana=q50 (robusta);
#   moda=argmax f(x) (pico da distribuição). Em dados contínuos a prob. de empate é ~0,
#   por isso arredondamos a 1 casa -> histograma discreto implícito p/ estimar o pico.
media = np.mean(dados)
mediana = np.median(dados)

# Para dados contínuos, aproximamos a moda agrupando os valores próximos
dados_arredondados = np.round(dados, 1)
moda = stats.mode(dados_arredondados, keepdims=True).mode[0]

# 3. Criar a visualização com Matplotlib
# CÓDIGO: hist(bins=40) estima a densidade; axvline marca cada medida; xlim(0,6) corta
#   a cauda longa p/ focar no corpo da distribuição onde estão as 3 medidas.
plt.figure(figsize=(10, 6))

# Plotar o histograma dos dados
plt.hist(dados, bins=40, edgecolor='black', color='skyblue', alpha=0.7, label='Dados')

# Adicionar as linhas verticais para Média, Mediana e Moda
plt.axvline(media, color='blue', linestyle='dashed', linewidth=2, label=f'Média: {media:.2f}')
plt.axvline(mediana, color='green', linestyle='dashdot', linewidth=2, label=f'Mediana: {mediana:.2f}')
plt.axvline(moda, color='red', linestyle='dotted', linewidth=2, label=f'Moda (aprox): {moda:.2f}')

# Configurações estéticas do gráfico
plt.title('Distribuição de Dados com Média, Mediana e Moda', fontsize=14, fontweight='bold')
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.xlim(0, 6) # Limitando o eixo X para melhor visualização da cauda
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Exibir o gráfico
plt.show()