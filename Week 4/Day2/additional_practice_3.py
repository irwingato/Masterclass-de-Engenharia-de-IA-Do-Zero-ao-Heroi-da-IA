"""
additional_practice_3.py — DISTRIBUIÇÕES em DADOS REAIS
=======================================================
OBJETIVO:
    Sair do sintético e ver assimetria/curtose em 2 datasets reais clássicos:
    imóveis (cauda longa) e consumo em restaurante (assimetria moderada).

MATEMÁTICA QUE VAMOS VER NA PRÁTICA:
    - Em dados reais de valores >= 0 com "poucos gigantes" (mansões, contas
      altas), quase sempre skew > 0 e média > mediana.
    - Distância média-mediana mede o "puxão" da cauda.
    - Curtose alta aqui = presença de outliers distantes (mansões caríssimas).
    - Modelo teórico típico: LogNormal (X = exp(Z), Z Normal) — nasce quando
      efeitos se MULTIPLICAM (juros, valorização, gorjetas %), não se somam.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing

# =====================================================================
# CONTROLE DE DADOS REAIS
# =====================================================================

# Cenário 1: Preço de Casas na Califórnia (típica cauda longa à direita)
# Dataset: 20.640 blocos da Califórnia (censo 1990). Cada linha = um bloco.
# CORREÇÃO: a coluna correta no sklearn é 'MedHouseVal' (não 'MedHouseValue').
# Ela vem em unidades de US$ 100 mil (ex: 2.5 = US$ 250 mil). Por isso *100000.
# Por que esperar skew > 0? Poucas mansões na costa puxam a média para cima,
# enquanto a maioria das casas é "comum". É o padrão renda/patrimônio.
california = fetch_california_housing(as_frame=True)
df_casas = california.frame
# O preço original está em centenas de milhares (multiplicamos por 100k para ficar real)
precos_casas = df_casas['MedHouseVal'] * 100000

# Cenário 2: Valor Total de Contas em Restaurantes (distribuição de consumo)
# Dataset 'tips' do seaborn: 244 contas reais. Coluna 'total_bill' em US$.
# Também assimétrica à direita (poucas mesas gastam muito), mas menos extrema
# que imóveis: ninguém paga 10x a conta mediana num jantar típico.
df_restaurante = sns.load_dataset('tips')
valor_contas = df_restaurante['total_bill']

# =====================================================================
# CÁLCULO DAS MÉTRICAS ESTATÍSTICAS
# skew = E[(X-mu)³]/sigma³ ; kurtosis Fisher = E[(X-mu)^4]/sigma^4 - 3.
# REGRA DE LEITURA: se média >> mediana, a cauda direita está puxando.
# =====================================================================
def extrair_metricas(dados, nome_cenario):
    """Calcula e imprime média, mediana, skew e curtose de uma série."""
    sk = stats.skew(dados)
    kt = stats.kurtosis(dados)  # Curtose em excesso (Fisher)
    print(f"--- {nome_cenario} ---")
    print(f"Média:   R$ {dados.mean():.2f}")
    print(f"Mediana: R$ {dados.median():.2f}")
    print(f"Assimetria (Skewness): {sk:.2f} (Positiva - Cauda à direita)")
    print(f"Curtose em Excesso:    {kt:.2f}\n")
    return sk, kt

sk_casas, kt_casas = extrair_metricas(precos_casas, "Preço de Imóveis (Califórnia)")
sk_contas, kt_contas = extrair_metricas(valor_contas, "Valor das Contas (Restaurante)")

# =====================================================================
# PLOTAGEM DOS GRÁFICOS REAIS
# histograma com stat="density" (área=1 ≈ PDF) + KDE (curva suave) +
# média (--) vs mediana (:). O vão entre as duas linhas É o skew visual.
# =====================================================================
plt.figure(figsize=(14, 5))
sns.set_theme(style="whitegrid")

# Gráfico 1: Mercado Imobiliário — esperar cauda direita longa e kurt alta.
plt.subplot(1, 2, 1)
sns.histplot(precos_casas, kde=True, color="#2C3E50", stat="density", alpha=0.7)
plt.axvline(precos_casas.mean(), color='red', linestyle='--', label=f'Média')
plt.axvline(precos_casas.median(), color='blue', linestyle=':', label=f'Mediana')
plt.title(f"Mercado Imobiliário\nSkew: {sk_casas:.2f} | Kurtosis: {kt_casas:.2f}")
plt.xlabel("Valor do Imóvel ($)")
plt.legend()

# Gráfico 2: Setor de Alimentação/Consumo — skew positivo mais suave.
plt.subplot(1, 2, 2)
sns.histplot(valor_contas, kde=True, color="#E67E22", stat="density", alpha=0.7)
plt.axvline(valor_contas.mean(), color='red', linestyle='--', label=f'Média')
plt.axvline(valor_contas.median(), color='blue', linestyle=':', label=f'Mediana')
plt.title(f"Consumo em Restaurantes\nSkew: {sk_contas:.2f} | Kurtosis: {kt_contas:.2f}")
plt.xlabel("Valor da Conta ($)")
plt.legend()

plt.tight_layout()
plt.show()
