"""
additional_practice_3.py — Seleção de características com Lasso (alpha=0.05)
=============================================================================
OBJETIVO: usar o Lasso NÃO para prever melhor, mas como SELETOR de variáveis:
identificar, entre os 9 termos polinomiais, quais são os preditores relevantes
e quais podem ser descartados (coeficiente = 0).

MATEMÁTICA DA SELEÇÃO L1:
- Problema: J(w) = (1/(2n))||y - Xw||^2 + alpha*||w||1.
  O termo ||w||1 = |w1|+...+|w9| é a "norma losango": as curvas de nível são
  losangos com vértices exatamente sobre os eixos. Ao expandir o losango até
  tocar a elipse do erro quadrático, o contacto cai frequentemente NUM VÉRTICE
  -> uma ou mais coordenadas = 0. É por isso que L1 gera esparsidade e L2 não
  (a bola L2 é redonda, o contacto é raramente no eixo).
- Condição de zeramento (soft-thresholding univariado, intuição):
  se |correlação_j com resíduo| < alpha -> w_j = 0 (variável eliminada).
  Logo, alpha é o "limiar de relevância": alpha=0.05 mantém efeitos moderados,
  alpha=1+ mataria quase tudo (visto no practice_2).
- Como os dados estão PADRONIZADOS (média 0, desvio 1), os coeficientes w_j
  ficam comparáveis: |w_j| grande = impacto grande no preço. Sem padronizar,
  um w pequeno poderia corresponder a uma variável de escala enorme.
- Métricas do script:
  Relevante  = NOT isclose(w, 0)  -> sobreviveu à penalização.
  Magnitude  = |w|                -> ordena por importância.
  Sinal      = sinal(w): positivo (teal) aumenta o preço, negativo (crimson) reduz.
- Resultado típico (alpha=0.05): 5 sobreviventes
  MedInc (+0.81, dominante), HouseAge (+0.16), AveRooms (-0.036),
  MedInc^2 (-0.016, saturação: rendimento extra rende menos no topo),
  AveRooms^2 (+0.002, quase irrelevante); 4 interações zeradas.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset California Housing (20640 amostras)
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Selecionar os recursos múltiplos a o alvo
# Mesmos 3 do practice_2 para comparar: rendimento, idade, nº quartos.
features = ['MedInc', 'HouseAge', 'AveRooms']
X = df[features]
y = df['MedHouseVal']

# 3. Padronizar os dados (Essencial para o Lasso comparar os pesos de forma justa)
# Matemática: z = (x - mu)/sigma por coluna. Sem isto, |w| não mede importância
# (ver docstring). Fit aqui no total por simplicidade didática.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Criar características polinomiais (Grau 2)
# include_bias=False para não criar uma coluna de 1s (o Lasso já tem o intercepto)
# Gera 9 colunas: 3 lineares + 3 quadráticas + 3 interações (C(5,2)-1 = 9).
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Obter o nome exato de cada nova característica gerada
# Ex: ['MedInc', 'HouseAge', ..., 'MedInc^2', 'MedInc HouseAge', ...]. O espaço
# significa interação (produto), ^2 significa quadrado.
feature_names = poly.get_feature_names_out(features)

# 5. Dividir em treino e teste
# random_state=42 = reproduzível. O Lasso é ajustado só no treino (sem leakage aqui).
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 6. Treinar a Regressão Lasso para Seleção de Características
# Usamos um alpha de 0.05 para aplicar uma penalidade moderada, ideal para seleção:
# alpha pequeno (0.001) manteria tudo (sem seleção); alpha grande (1+) zeraria
# quase tudo (ver practice_2: 8-9 zeros). 0.05 é o ponto de equilíbrio empírico.
# random_state só afeta selection='random'; com default 'cyclic' é ignorado.
lasso = Lasso(alpha=0.05, random_state=42)
lasso.fit(X_train, y_train)  # coordinate descent: atualiza cada w_j por soft-thresholding

# 7. Criar um DataFrame associando cada Recurso ao seu Coeficiente obtido
# lasso.coef_ tem 9 valores (um por termo polinomial); lasso.intercept_ é o w0
# (preço médio quando todas as variáveis estão na média, por causa da padronização).
df_coeficientes = pd.DataFrame({
    'Características': feature_names,
    'Coeficiente': lasso.coef_
})

# Filtrar para ver quem sobreviveu e quem foi zerado (eliminado)
# np.isclose(x, 0) com tolerância default ~1e-8: |w|<1e-8 conta como zero numérico.
df_coeficientes['Relevante'] = df_coeficientes['Coeficiente'].apply(lambda x: "Sim" if not np.isclose(x, 0) else "Eliminado (Zero)")
# Magnitude = |w|: como X está padronizado, maior |w| = maior impacto no preço.
df_coeficientes['Magnitude'] = df_coeficientes['Coeficiente'].abs()
# Ordenar por importância decrescente para o relatório (MedInc deve ficar no topo).
df_coeficientes = df_coeficientes.sort_values(by='Magnitude', ascending=False)

# -------------------------------------------------------------------------
# Visualização das Características Selecionadas
# Gráfico de barras horizontais divergente: eixo x = valor de w (sinal importa),
# eixo y = nome do termo. teal = efeito positivo (sobe o preço), crimson = negativo.
# Linha tracejada em 0 marca a fronteira; só os "Sim" aparecem (os zeros foram filtrados).
# -------------------------------------------------------------------------
# Filtrando apenas as que não foram zeradas para o gráfico, ordenadas por valor
# crescente para o barh ficar em escada legível (mais negativo em baixo).
df_relevantes = df_coeficientes[df_coeficientes['Relevante'] == "Sim"].sort_values(by='Coeficiente', ascending=True)

plt.figure(figsize=(10, 6))
# Lista de cores por barra: crimson se w<0 (ex. AveRooms reduz preço), teal se w>0.
color = ['crimson' if c < 0 else 'teal' for c in df_relevantes['Coeficiente']]
plt.barh(df_relevantes['Características'], df_relevantes['Coeficiente'], color=color, edgecolor='black')
plt.axvline(0, color='black', linestyle='--', linewidth=1)  # referência do zero
plt.title("Preditores Mais Relevantes Selecionados pelo Lasso (Alpha = 0.05)")
plt.xlabel("Valor do Coeficiente (Impacto no preço da casa)")
plt.ylabel("Características Polinomiais")
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Exibir relatório completo no console (ordenado por |w|: do mais ao menos importante)
print("\n" + "-"*65)
print("  RELATÓRIO LASSO DE SELEÇÃO DE CARACTERÍSTICAS")
print("="*65)
print(df_coeficientes[['Características', 'Coeficiente', 'Relevante']].to_string(index=False, formatters={'Coeficiente': '{:,.4f}'.format}))
print("-"*65)
