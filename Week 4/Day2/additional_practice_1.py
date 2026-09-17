"""
additional_practice_1.py — Efeito da ASSIMETRIA (skew) e CURTOSE (kurtosis)
===========================================================================
OBJETIVO:
    Comparar visual e numericamente como assimetria e curtose deformam
    diferentes conjuntos de dados em relação à Normal.

MATEMÁTICA ENVOLVIDA:
    - Média:            mu = E[X]
    - Variância:        sigma² = E[(X - mu)²]
    - Assimetria:       skew = E[(X - mu)³] / sigma³
        * skew ≈ 0  -> simétrica (ex: Normal). média ≈ mediana.
        * skew > 0  -> cauda longa à DIREITA. média > mediana.
        * skew < 0  -> cauda longa à ESQUERDA. média < mediana.
    - Curtose (Fisher, "em excesso"): kurt = E[(X - mu)^4] / sigma^4 - 3
        * kurt ≈ 0  -> caudas como a Normal (mesocúrtica).
        * kurt > 0  -> leptocúrtica: pico alto + caudas pesadas (outliers).
        * kurt < 0  -> platicúrtica: achatada, caudas leves.
      O "-3" serve para a Normal dar exatamente 0 (curtose bruta da Normal = 3).

CONJUNTOS GERADOS AQUI (n = 5000 cada):
    1. Normal           -> referência simétrica, curtose ~0.
    2. LogNormal        -> assimetria positiva clássica (renda, preços).
    3. LogNormal espelhada -> assimetria negativa artificial.
    4. Johnson SU       -> alta curtose (caudas pesadas) sem muita assimetria.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, lognorm, johnsonsu

# Semente fixa: garante que todo mundo gere os mesmos números aleatórios.
np.random.seed(42)
n_samples = 5000  # Tamanho de cada conjunto. Grande => estimativas estáveis.

# ---------------------------------------------------------------------
# 1. Dados Normais (referência): X ~ N(mu=0, sigma=1)
# PDF: f(x) = 1/sqrt(2*pi) * exp(-x²/2). Simétrica, skew=0, kurt=0.
# ---------------------------------------------------------------------
dados_normais = np.random.normal(loc=0, scale=1, size=n_samples)

# ---------------------------------------------------------------------
# 2. Assimetria POSITIVA (cauda à direita): X ~ LogNormal(s=0.7)
# Se Z ~ N(0,1), então X = exp(s*Z) * scale. Suporte: [0, +inf).
# PDF: f(x) = 1/(x*s*sqrt(2*pi)) * exp(-ln(x/scale)²/(2s²)).
# Por que é assimétrica? A exponencial "esticada" cria poucos valores
# muito grandes -> média puxada para a direita, mediana fica menor.
# Exemplo real: salários, preços de imóveis.
# ---------------------------------------------------------------------
dados_assimetria_pos = lognorm.rvs(s=0.7, loc=0, scale=1, size=n_samples)

# ---------------------------------------------------------------------
# 3. Assimetria NEGATIVA (cauda à esquerda)
# MATEMÁTICA DO TRUQUE: a LogNormal só tem cauda à direita e exige s > 0
# (o código antigo usava s=-0.7, que dá ValueError no scipy).
# Para criar cauda à esquerda, ESPELHAMOS: Y = C - X, onde X é LogNormal.
# Espelhar inverte o sinal do skew: skew(C - X) = -skew(X).
# C=4 centraliza os valores numa faixa parecida com a Normal para comparar.
# ---------------------------------------------------------------------
dados_assimetria_neg = 4 - lognorm.rvs(s=0.7, loc=0, scale=1, size=n_samples)

# ---------------------------------------------------------------------
# 4. Alta CURTOSE (leptocúrtica): X ~ Johnson SU(a=0, b=1.5)
# A Johnson SU é uma Normal transformada: X = loc + scale*sinh((Z-a)/b),
# onde Z ~ N(0,1) e sinh é o seno hiperbólico.
# Com b pequeno, o sinh "esticada" as caudas -> muitos outliers,
# pico central alto, mas skew ≈ 0 (simétrica).
# Exemplo real: retornos financeiros (maioria perto de 0, crashes raros).
# ---------------------------------------------------------------------
dados_alta_curtose = johnsonsu.rvs(a=0, b=1.5, loc=0, scale=1, size=n_samples)

# Junta tudo num DataFrame: cada coluna é um "cenário" para comparar.
df = pd.DataFrame({
    'Normal': dados_normais,
    'Assimetria Positiva': dados_assimetria_pos,
    'Assimetria Negativa': dados_assimetria_neg,
    'Alta Curtose': dados_alta_curtose})

# ---------------------------------------------------------------------
# Métricas: scipy.stats.skew = momento padronizado de ordem 3,
# scipy.stats.kurtosis = Fisher (já subtrai 3). Logo Normal ≈ 0.
# ---------------------------------------------------------------------
print("--- Métricas Estatísticas Calculadas ---")
for coluna in df.columns:
    s = skew(df[coluna])
    k = kurtosis(df[coluna])  # Curtose em excesso (0 significa normal)
    print(f"{coluna}: Assimetria = {s:.2f}, Curtose em Excesso = {k:.2f}")

# ---------------------------------------------------------------------
# Plotagem: histograma (densidade) + KDE + média vs mediana.
# LEITURA DO GRÁFICO:
# - Se média (vermelho --) > mediana (preto :) -> skew > 0.
# - Se média < mediana -> skew < 0.
# - Pico fino + caudas longas -> kurt > 0.
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

cores = ['#4A90E2', '#E24A8D', '#50E3C2', '#F5A623']
titulos = df.columns

for i, col in enumerate(df.columns):
    plt.subplot(2, 2, i + 1)
    # stat="density": área total = 1, logo é uma estimativa da PDF.
    sns.histplot(df[col], kde=True, color=cores[i], stat="density", alpha=0.6)

    # Linhas de referência (Média e Mediana)
    media = df[col].mean()
    mediana = df[col].median()
    plt.axvline(media, color='red', linestyle='--', linewidth=1.5, label=f'Média: {media:.2f}')
    plt.axvline(mediana, color='black', linestyle=':', linewidth=1.5, label=f'Mediana: {mediana:.2f}')

    plt.title(f"{titulos[i]}\n(Assimetria: {skew(df[col]):.2f} | Curtose: {kurtosis(df[col]):.2f})", fontsize=12)
    plt.xlabel('Valores')
    plt.ylabel('Densidade')
    plt.legend()

plt.tight_layout()
plt.show()
