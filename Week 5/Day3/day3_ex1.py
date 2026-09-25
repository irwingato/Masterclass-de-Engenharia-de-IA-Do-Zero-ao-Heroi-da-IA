"""
day3_ex1.py — Ridge vs Lasso em dados REAIS (California Housing, 1 recurso)
============================================================================
OBJETIVO: ponte entre o sintético (samples2) e os practices: aplicar Ridge e
Lasso ao dataset real California Housing usando só MedInc + polinómio grau 2.

PIPELINE: MedInc -> [x, x^2] -> split 80/20 (seed 42) -> Ridge(α=1) e Lasso(α=0.1)
-> scatter previsões -> MSE de teste.

MATEMÁTICA E ESCOLHAS:
1) Porquê α diferentes (Ridge=1, Lasso=0.1)? Porque as escalas das penalizações
   NÃO são comparáveis: Ridge minimiza ||y-Xw||² + α||w||² enquanto o Lasso
   sklearn minimiza (1/2n)||y-Xw||² + α||w||₁. O fator 1/(2n) (≈1/33000 no treino!)
   torna o termo de ajuste minúsculo no Lasso, logo o MESMO α numérico pune
   muito mais no Lasso. α_lasso=0.1 equivale grosseiramente a um α_ridge bem maior.
   Resultado observado: Ridge MSE≈0.703, Lasso(0.1) MSE≈0.721 — próximos, com o
   Lasso ligeiramente pior por já estar a encolher forte (ver practice_1: com
   α=1 o Lasso zera w1 e o MSE salta para 0.81).

2) OLS como referência (código comentado no fim, linhas 55-74): sem penalização,
   w*=(X^T X)^-1 X^T y. Com [x,x^2] correlacionados, o OLS teria variância alta;
   Ridge(α=1) estabiliza com viés mínimo. O bloco comentado ajustava no dataset
   TODO (sem split) -> MSE otimista; o código ativo corrige isso avaliando no teste.

3) Visualização por scatter (não por linha): cada ponto verde/laranja é um
   ŷ_teste plotado contra x_teste. Como o teste está embaralhado, os pontos
   previstos formam a "nuvem da parábola" mas com dispersão vertical do ruído
   real. Uma linha ordenada (como no additional_practice_1.py com sort_idx)
   mostraria melhor a curva média ŷ(x); aqui optou-se por nuvem vs nuvem.

4) MSE realista: ~0.70 (em unidades (100k$)²). Melhor que Lasso forte, pior que
   o modelo multirrecurso do practice_2 (0.64) — prova de que 1 recurso limita o
   poder explicativo (viés por omissão de variáveis: HouseAge, AveRooms, etc.).

5) Sem StandardScaler aqui: com 1 variável em escala única (MedInc 0-15, x^2 0-225)
   a distorção existe mas é pequena para α's escolhidos; com 3+ variáveis (practice_2/3)
   o scaler passa a obrigatório.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# Carregar California Housing (20640 casas; alvo MedHouseVal em $100k)
data = fetch_california_housing(as_frame=True)
df = data.frame

# Selecionar feature (MedInc = rendimento mediano) e alvo (valor mediano da casa)
# 1D de propósito: permite scatter ŷ vs x interpretável.
X = df[['MedInc']]
y = df['MedHouseVal']

# Expandir para [x, x^2]: ŷ = w0 + w1·x + w2·x^2 (parábola: capta saturação do preço
# em rendimentos altos). Sem coluna de bias (intercepto tratado nos estimadores).
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split 80/20, seed 42: teste mede generalização fora da amostra.
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ridge α=1 (L2): J=||y-Xw||²+1·||w||². Encolhimento suave, nunca zera.
# Esperado: w≈[0.53,-0.01], MSE≈0.703 (ver tabela do practice_1).
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

# Lasso α=0.1 (L1): J=(1/2n)||y-Xw||²+0.1·||w||₁. Já encolhe forte (w≈[0.31,0.007]),
# mas ainda sem zerar. Se α fosse 1, w1→0 e MSE→0.81 (practice_1).
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

# Visualizar: azul=dados reais, verde=Ridge, laranja=Lasso (todos vs MedInc de teste).
# Leitura: as duas nuvens previstas sobrepõem-se bastante (α's moderados); com α=10
# ver-se-ia a nuvem laranja colapsar para quase-horizontal (underfit).
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], y_test, color="blue", label="Actual Data", alpha=0.5)
plt.scatter(X_test[:, 0], ridge_predictions, color="green", label="Ridge Predictions", alpha=0.5)
plt.scatter(X_test[:, 0], lasso_predictions, color="orange", label="Lasso Predictions", alpha=0.5)
plt.title("Ridge vs Lasso Regressions")
plt.xlabel("Median Income (Transformed)")
plt.ylabel("Median House Value in California")
plt.legend()
plt.show()

# Avaliar Ridge: MSE_teste = (1/n_teste)Σ(y-ŷ_ridge)². Menor = melhor generalização.
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("Ridge Regression MSE:", ridge_mse)

# Avaliar Lasso: mesma fórmula com ŷ_lasso. Comparação direta e honesta (mesmo teste).
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print("Lasso Regression MSE:", lasso_mse)

# Bloco comentado (referência OLS sem regularização):
# Matemática: min ||y-Xw||², w=(X^T X)^-1 X^T y. Sem split (ajusta+avalia em tudo)
# -> MSE otimista e w instável pela colinearidade [x,x^2]. Mantido comentado para
# não poluir a comparação Ridge vs Lasso acima; para reativar, adicionar split.
# Fit polynomial regression model
# model = LinearRegression()
# model.fit(X_poly, y)

# # Make Predictions
# y_pred = model.predict(X_poly)

# # Plot actual vs predicted values
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, color="blue", label="Actual Data", alpha=0.5)
# plt.scatter(X, y_pred, color="red", label="Predicted Curve", alpha=0.5)
# plt.title("Polynomial Regression")
# plt.xlabel("Median Income in California")
# plt.ylabel("Median House Value in California")
# plt.legend()
# plt.show()

# # Evaluate model performance
# mse = mean_squared_error(y, y_pred)
# print("Mean Squared Error (MSE):", mse)
