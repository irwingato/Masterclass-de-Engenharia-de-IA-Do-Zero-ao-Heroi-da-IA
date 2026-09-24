# =============================================================================
# REGRESSAO LINEAR SIMPLES - Explicacao matematica geral
# -----------------------------------------------------------------------------
# Modelo: y = beta_0 + beta_1 * x + epsilon
#   beta_1 = slope (inclinacao), beta_0 = intercept (viés)
#   epsilon ~ N(0, sigma^2) = ruido gaussiano
#
# Objetivo do treino (OLS - Ordinary Least Squares):
#   Minimizar S(beta_0, beta_1) = soma_{i=1..n} (y_i - beta_0 - beta_1*x_i)^2
# Solucao fechada:
#   beta_1 = Cov(X, y) / Var(X)
#   beta_0 = media(y) - beta_1 * media(x)
#   Forma matricial: beta_hat = (X^T X)^-1 X^T y (sklearn usa SVD)
#
# Metricas:
#   MSE = (1/n) * soma (y_i - y_pred_i)^2  -> erro quadratico medio
#   R2 = 1 - soma(y_i - y_pred_i)^2 / soma(y_i - media(y))^2 = 1 - SSres/SStot
# =============================================================================

import numpy as np  # arrays e numeros aleatorios
import pandas as pd  # importado mas NAO usado neste script
from sklearn.model_selection import train_test_split  # divisao treino/teste
from sklearn.linear_model import LinearRegression  # modelo OLS: y = b0 + b1*x
from sklearn.metrics import mean_squared_error, r2_score  # MSE e R2 (formulas acima)
import matplotlib.pyplot as plt  # grafico dispersao + reta

# --- 1. Geracao de dados sinteticos ---
# Matematica verdadeira: y = 3*x + epsilon, com epsilon ~ N(0, 2^2)
np.random.seed(42)  # seed = reprodutibilidade (mesmos numeros aleatorios sempre)
X = np.random.rand(100, 1) * 100  # 100 valores uniformes em [0,100). Shape (100,1) pois sklearn exige 2D.
y = 3 * X + np.random.randn(100, 1) * 2  # beta_1=3, beta_0=0, ruido gaussiano com sigma=2 (variancia=4)

# --- 2. Divisao treino/teste ---
# 80% treino (ajusta beta_0, beta_1) e 20% teste (avalia generalizacao em dados novos).
# random_state=42 garante mesma divisao a cada execucao.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Ajuste da Regressao Linear (OLS) ---
model = LinearRegression()  # cria modelo y_hat = b0 + b1*x
model.fit(X_train, y_train)  # minimiza S(b0,b1) = soma(y_train - b0 - b1*X_train)^2 so no treino

# --- 4. Predicao ---
# Aplica a equacao aprendida: y_pred = beta_0_hat + beta_1_hat * X_test
y_pred = model.predict(X_test)

# --- 5. Coeficientes aprendidos ---
# beta_1_hat ~ 2.99 (perto do 3 real) e beta_0_hat ~ 0.28 (perto do 0 real).
# Desvio vem do ruido epsilon.
print("Slope : ", model.coef_[0][0])  # beta_1 (inclinacao): Cov(X,y)/Var(X)
print("Intercept : ", model.intercept_[0])  # beta_0 (intercepto): media(y) - beta_1*media(x)

# --- 6. Visualizacao ---
plt.scatter(X_test, y_test, color="blue", label="Actual")  # pontos reais y_test (com ruido)
plt.plot(X_test, y_pred, color="red", label="Predicted")  # reta predita y_hat. OBS: como X_test nao esta ordenado, a linha fica serrilhada; ideal seria ordenar X_test.
plt.title("Linear Regression Model")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# --- 7. Avaliacao de performance ---
# MSE: media dos erros ao quadrado. Esperado ~4 (= sigma^2 do ruido). Aqui ~2.6.
mse = mean_squared_error(y_test, y_pred)
# R2: fracao da variancia explicada. 1 = perfeito, 0 = preve so a media. Aqui ~0.999 pois sinal (0-300) >> ruido (sigma=2).
r2 = r2_score(y_test, y_pred)
print("MSE: ", mse)
print("R-Squared: ", r2)