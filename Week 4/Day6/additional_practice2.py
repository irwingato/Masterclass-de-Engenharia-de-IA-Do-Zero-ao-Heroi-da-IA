import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 1. Gerando dados com uma relação não linear (Parábola: Y = X² + ruído)
np.random.seed(42)
X = np.random.uniform(-10, 10, 200)
y = X**2 + np.random.normal(0, 10, 100)

df = pd.DataFrame({'X': X, 'Y': y})

# ==========================================
# 🔍 PARTE 1: COMPARAÇÃO DE CORRELAÇÃO
# ==========================================
# Pearson mede apenas relações lineares. Spearman mede relações monotônicas (não lineares que só sobem ou só descem).
corr_pearson, _ = pearsonr(df['X'], df['Y'])
corr_spearman, _ = spearmanr(df['X'], df['Y'])

print("--- CORRELAÇÕES ---")
print(f"Correlação de Pearson (Linear): {corr_pearson:.4f} (Falhou dem detectar a parábola)")
print(f"Correlação de Spearman (Monotônica): {corr_spearman:.4f} (Também falhou, pois a curva desce e sobe)")

# ==========================================
# 📊 PARTE 2: COMPARAÇÃO DE REGRESSÃO
# ==========================================
X_arr = df[['X']].values
y_arr = df[['Y']].values

# Regressão Linear Simples
modelo_linear = LinearRegression()
modelo_linear.fit(X_arr, y_arr)
Y_pred_linear = modelo_linear.predict(X_arr)
r2_linear = r2_score(y_arr, Y_pred_linear)

# Regressão Polinomial (Não Linear) - Grau 2
transformador_poly = PolynomialFeatures(degree=2)
X_poly = transformador_poly.fit_transform(X_arr)

modelo_poly = LinearRegression()
modelo_poly.fit(X_poly, y_arr)
Y_pred_poly = modelo_poly.predict(X_poly)
r2_poly = r2_score(y_arr, Y_pred_poly)

print("\n--- REGRESSÕES (R²) ---")
print(f"R² da Regressão Linear: {r2_linear:.4f} (Não consegue explicar a variação)")
print(f"R² da Regressão Polinomial (Grau 2): {r2_poly:.4f} (Ajuste quase perfeito)")