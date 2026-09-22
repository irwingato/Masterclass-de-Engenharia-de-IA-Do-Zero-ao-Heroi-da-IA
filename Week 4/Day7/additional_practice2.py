"""
additional_practice2.py — Regressão Linear MÚLTIPLA em dados reais de saúde.
===========================================================================
DATASET: smoking.csv (55.692 linhas) — age, weight(kg), smoking (Y/N),
  Cholesterol (alvo) + dezenas de outras colunas clínicas.
URL: https://raw.githubusercontent.com/aliabdallah7/smoking-detection/refs/heads/main/smoking.csv

OBJETIVO: prever colesterol a partir de 3 variáveis (idade, peso, fumo).
TÉCNICAS: dummy encoding manual, train_test_split 80/20, OLS múltiplo,
  R² fora da amostra, interpretação ceteris paribus + boxplot.

MODELO MATEMÁTICO:
  Cholesterol = b0 + b1*age + b2*weight + b3*smoking_num + erro
  onde smoking_num = 1 se 'Y', 0 se 'N'. b3 = efeito médio do fumo
  mantendo idade e peso fixos (paradigma "tudo o mais constante").
"""

import matplotlib.pyplot as plt  # figuras
import numpy as np  # (reservado p/ operações numéricas)
import pandas as pd  # CSV -> DataFrame, mapeamento, seleção de colunas
import seaborn as sns  # boxplot estatístico
from sklearn.linear_model import LinearRegression  # OLS múltiplo
from sklearn.metrics import r2_score  # R² no conjunto de teste
from sklearn.model_selection import train_test_split  # divisão treino/teste

# ---------------------------------------------------------------------------
# 1. CARGA DO DATASET REAL
# Computacional: CSV grande (~5 MB). pd.read_csv infere: age/int, weight/float,
# smoking/object(Y/N), Cholesterol/int. Nomes são case-sensitive:
# 'Cholesterol' com C maiúsculo, 'weight(kg)' com dois t — erros aqui dão KeyError.
# ---------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/aliabdallah7/smoking-detection/refs/heads/main/smoking.csv"
df = pd.read_csv(url)

# ---------------------------------------------------------------------------
# Inspeção: lista colunas e mostra 5 linhas das variáveis de interesse.
# Por que inspecionar? Detecta encoding inesperado (ex: smoking Y/N e não 0/1).
# ---------------------------------------------------------------------------
print("--- Colunas Disponíveis no Dataset ---")
print(df.columns.tolist()[:10])  # Exibe as 10 primeiras colunas
print("\n--- Primeiras Linhas ---")
print(df[['age', 'weight(kg)', 'smoking', 'Cholesterol']].head(), "\n")

# ---------------------------------------------------------------------------
# 2. ENGENHARIA DE RECURSOS + DEFINIÇÃO X/y
# Matemática — dummy encoding: variável categórica binária vira 0/1 para entrar
# na equação linear. Sem isso, 'Y'/'N' não tem ordem nem distância numérica.
#   smoking_num = 1_{smoking=='Y'} (função indicadora).
# Computacional: (df['smoking']=='Y') gera Series bool; .astype(int) vira 0/1.
#   .astype(float) garante matriz float64 para o solver numérico (SVD).
# X shape = (n,3), y shape = (n,). Cada linha = um paciente.
# ---------------------------------------------------------------------------
# Alvo (y): Colesterol (Variável contínua, coluna 'Cholesterol' com C maiúsculo)
# Recursos (X): Idade, Peso e Smoking (smoking vem como 'Y'/'N', então convertemos para 1/0)
df['smoking_num'] = (df['smoking'] == 'Y').astype(int)
X = df[['age', 'weight(kg)', 'smoking_num']].astype(float)
y = df['Cholesterol']

# ---------------------------------------------------------------------------
# 3. DIVISÃO TREINO (80%) / TESTE (20%)
# Matemática/ML: treinar e avaliar nos MESMOS dados infla R² (overfitting
# otimista). Separar 20% "inéditos" estima erro de generalização.
# Computacional: train_test_split embaralha (shuffle) e reparte.
# random_state=42 = semente fixa -> resultado reproduzível (mesma partição).
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# 4. TREINO OLS MÚLTIPLO
# Matemática: minimiza S(b) = ||y - Xb||². Solução: b = (X^T X)^-1 X^T y
# (sklearn usa SVD, estável mesmo com colinearidade). Custo O(n*p²), p=3 aqui.
# b0 = intercepto (colesterol basal p/ age=0, weight=0, não-fumante — só
# referência matemática, sem sentido clínico); b1,b2,b3 = efeitos marginais.
# ---------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 5. AVALIAÇÃO FORA DA AMOSTRA
# y_pred = X_test @ coef_ + intercept_ (produto matricial).
# R² = 1 - soma((y-y_pred)²)/soma((y-media)²). R²=1 perfeito, 0 = pior que média,
# negativo = pior que chutar a média. Em saúde, R² baixo (~0.05-0.15) é comum:
# colesterol depende de genética/dieta não incluídas (viés de variável omitida).
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("--- Resultados da Regressão (Previsão de Colesterol) ---")
print(f"Intercepto (Linha de Base): {model.intercept_:.2f}")
print(f"R² (Poder de Explicação): {r2:.4f}\n")

print("Coeficientes:")
print(f"  - Idade (age): {model.coef_[0]:.4f}")  # +mg/dL por ano extra, peso/fumo fixos
print(f"  - Peso (weight): {model.coef_[1]:.4f}")  # +mg/dL por kg extra, idade/fumo fixos
print(f"  - Fumante (smoking): {model.coef_[2]:.4f}")  # salto médio Y vs N

# ---------------------------------------------------------------------------
# 6. INTERPRETAÇÃO CETERIS PARIBUS ("tudo o mais constante")
# Essa é a força da regressão múltipla vs comparar médias brutas: b3 isola o
# fumo porque idade e peso estão controlados na equação. Se b3>0, fumantes têm
# +b3 mg/dL em média. Cuidado causal: é associação observacional, não ensaio
# randomizado — confounders (ex: dieta) podem viesar b3.
# ---------------------------------------------------------------------------
print("\n--- Análise Prática ---")
if model.coef_[2] > 0:
    print(f"• Mantendo idade e peso constantes, indivíduos fumantes possuem, em média, {model.coef_[2]:.2f} mg/dL a MAIS de colesterol comparados e não fumantes.")
else:
    print(f"• Mantendo idade e peso constantes, indivíduos fumantes possuem, em média,  {abs(model.coef_[2]):.2f} mg/dL a MENOS e colesterol comparados a não fumantes.")

# ---------------------------------------------------------------------------
# 7. VISUALIZAÇÃO: boxplot colesterol por status de fumo
# Matemática: mesma lógica do practice1 (Q1/mediana/Q3/IQR/outliers) por grupo
# N vs Y. order=['N','Y'] fixa ordem lógica (não-fumante primeiro).
# xticks mapeia posição 0->'N', 1->'Y' para rótulos amigáveis.
# Complementa a regressão: mostra distribuição bruta (sem ajuste por idade/peso),
# enquanto b3 mostra efeito AJUSTADO. Se divergirem, há confundimento.
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='smoking', y='Cholesterol', palette='Set1', order=['N', 'Y'])
plt.title("Impacto do Status de Fumante no Colesterol")
plt.xlabel("Fumante (N = Não, Y = Sim)")
plt.ylabel("Colesterol Total")
plt.xticks([0, 1], ['Não Fumante (N)', 'Fumante (Y)'])
plt.show()
