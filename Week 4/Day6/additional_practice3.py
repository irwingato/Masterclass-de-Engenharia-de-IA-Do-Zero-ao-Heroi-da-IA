import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Carregar o conjunto de dados real de habitação da Califórnia
california = fetch_california_housing(as_frame=True)
df = california.frame

# Definindo as variáveis: X (Renda Mediana) e Y (Valor Médio do Imóvel em $100k)
X = df[['MedInc']].values
y = df['MedHouseVal'].values

# Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 📊 ABORDAGEM 1: REGRESSÃO LINEAR
# ==========================================
modelo_linear = LinearRegression()
modelo_linear.fit(X_train, y_train)
y_pred_linear = modelo_linear.predict(X_test)

# ==========================================
# 📈 ABORDAGEM 2: REGRESSÃO POLINOMIAL (GRAU 2)
# ==========================================
transformador_poly = PolynomialFeatures(degree=2)
X_train_poly = transformador_poly.fit_transform(X_train)
X_test_poly = transformador_poly.transform(X_test)

modelo_poly = LinearRegression()
modelo_poly.fit(X_train_poly, y_train)
y_pred_poly = modelo_poly.predict(X_test_poly)

# ==========================================
# 🔍 AVALIAÇÃO DE RESULTADOS
# ==========================================
print("--- DESEMPENHO NO MODELO LINEAR ---")
print(f"R² (Explicação da variância): {r2_score(y_test, y_pred_linear):.4f}")
print(f"Erro Médio Absoluto (MAE): ${mean_absolute_error(y_test, y_pred_linear) * 100000:.2f}")

print("\n--- DESEMPENHO NO MODELO POLINOMIAL (NÃO LINEAR) ---")
print(f"R² (Explicação da variância): {r2_score(y_test, y_pred_poly):.4f}")
print(f"Erro Médio Absoluto (MAE): ${mean_absolute_error(y_test, y_pred_poly) * 100000:.2f}")