"""
day3_samples.py — Regressão Polinomial OLS em dados sintéticos (base de tudo)
==============================================================================
OBJETIVO: mostrar a regressão polinomial SEM regularização, em dados onde
conhecemos a verdade — para depois comparar com Ridge/Lasso.

DADOS SINTÉTICOS (verdade conhecida):
    x ~ Uniforme(0, 10),  100 pontos
    y = 3*x^2 + 2*x + eps,  eps ~ Normal(0, 5^2)  [ruído gaussiano, sigma=5]
Ou seja: w1_verdade=2 (linear), w2_verdade=3 (quadrático). O modelo estimado
deveria recuperar valores próximos destes.

MATEMÁTICA:
1) Expansão polinomial: phi(x) = [x, x^2]. O modelo continua LINEAR nos
   parâmetros: ŷ = w0 + w1*x + w2*x^2. "Polinomial" refere-se às features,
   não aos pesos — por isso o OLS aplica-se na mesma.

2) OLS (Ordinary Least Squares):  min_w ||y - X_poly w||^2
   Solução fechada (equações normais):  w* = (X^T X)^-1 X^T y
   Teorema Gauss-Markov: sob ruído homocedástico não-correlacionado, o OLS é o
   melhor estimador linear não-enviesado (BLUE). Mas Var(w*) = sigma^2 (X^T X)^-1
   explode se as colunas [x, x^2] forem quase colineares -> motiva Ridge/Lasso.

3) MSE: MSE = (1/n) Σ(y_i - ŷ_i)^2. Estima sigma^2 + viés^2 + variância.
   ATENÇÃO metodológica: aqui o MSE é calculado NOS MESMOS dados do treino
   (in-sample, sem train_test_split). É otimista: mede ajuste, não generalização.
   Nos ficheiros seguintes corrige-se isto com train_test_split.

4) Ruído: np.random.randn*5 gera eps com desvio 5. Mesmo o modelo perfeito teria
   MSE ≈ 25 (variância irredutível). Se o MSE obtido for ~20-30, o modelo está
   a recuperar bem a verdade; muito abaixo disso seria sobreajuste ao ruído.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Gerar dados sintéticos (seed=42 = reproduzível: mesmos 100 pontos sempre)
# X uniforme 0-10 (100,1); y = parábola verdadeira 3x^2+2x + ruído N(0,25).
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5

# Transformar features para polinómio grau 2: cada linha x -> [x, x^2]
# include_bias=False: sem coluna de 1s (o LinearRegression já estima w0=intercept_).
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Ajustar OLS: resolve (X^T X)^-1 X^T y por SVD internamente (mais estável que
# inverter diretamente). Esperamos coef_ ≈ [2, 3] e intercept_ ≈ 0.
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)  # ŷ = X_poly @ coef_ + intercept_ (nos dados de treino)

# Plotar: azul = dados ruidosos, vermelho = parábola ajustada.
# NOTA: scatter de ŷ vs X mostra a curva amostrada nos 100 x's (com buracos onde
# não há pontos); o ideal seria uma grelha densa ordenada para linha contínua.
plt.scatter(X, y, color="blue", label="Data")
plt.scatter(X, y_pred, color="red", label="Predicted Data")
plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Avaliar: MSE in-sample (treino). Referência: ~25 = variância do ruído.
# MSE >> 25 = underfit (modelo não capturou a parábola); MSE << 25 = sobreajuste.
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)
# Para inspeção didática: devia imprimir também model.coef_ (≈[2,3]) e intercept_.
