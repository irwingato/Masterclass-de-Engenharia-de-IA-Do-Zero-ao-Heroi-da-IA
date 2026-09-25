"""
day3_samples2.py — Ridge vs Lasso nos mesmos dados sintéticos (com teste)
============================================================================
OBJETIVO: repetir o day3_samples.py MAS com (1) divisão treino/teste e
(2) regularização — para ver o preço da penalização quando a verdade é densa.

DADOS: os mesmos do samples.py: y = 3x^2 + 2x + eps, eps~N(0,25), 100 pontos.
Verdade densa: w1=2 e w2=3, AMBOS ≠ 0. Nenhuma variável deveria ser eliminada.

MATEMÁTICA:
1) Divisão 80/20 (random_state=42): treino ajusta w, teste estima o erro de
   generalização E[(y_novo - ŷ)^2] = viés^2 + variância + sigma^2. É a medida
   honesta; o MSE de treino do samples.py era otimista.

2) Ridge (L2):  J = ||y-Xw||^2 + α||w||²,  w* = (X^T X + αI)^-1 X^T y.
   Com α=1: encolhe w1,w2 ligeiramente para 0 (ex. 3.0 -> ~2.9). Como X tem só
   2 colunas bem condicionadas (x e x^2 em 0-10, após split ainda ok), o efeito
   é pequeno: MSE_teste ≈ MSE_OLS + pequeno viés. Ridge nunca zera.

3) Lasso (L1):  J = (1/2n)||y-Xw||^2 + α||w||₁,  sem solução fechada;
   coordinate descent + soft-thresholding: w_j = sign(z_j)·max(|z_j|-α, 0).
   Com α=1 (FORTE para dados com w~2-3 e escala x até 100 em x^2!): o limiar α
   pode cortar parte real do sinal -> viés maior que o Ridge. Como a verdade é
   densa, a esparsidade do Lasso NÃO ajuda aqui — esperamos Lasso_MSE > Ridge_MSE.
   Lição: Lasso só vence quando a verdade é esparsa (alguns w=0); senão, Ridge
   (ou OLS) generaliza melhor. Ver practice_2 onde Lasso com α=1 colapsa.

4) Escala e α: x^2 chega a 100 enquanto x chega a 10. SEM padronizar (este
   ficheiro não usa StandardScaler!), a penalização α·(w1²+w2²) pune w2 com o
   mesmo peso embora x^2 tenha escala 10x maior. Na prática devia-se padronizar
   antes (como no additional_practice_2.py). Aqui funciona porque α=1 é pequeno
   face a ||y||²~O(10⁴), mas é metodologicamente imperfeito — bom ponto de discussão.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# Gerar os mesmos dados sintéticos (seed 42): parábola 3x^2+2x + ruído sigma=5.
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5

# Expandir para [x, x^2] (ver samples.py). fit_transform aqui não causa leakage
# relevante porque PolynomialFeatures não estima estatísticas (só aplica x²).
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Dividir 80 treino / 20 teste. O teste fica "escondido" até ao predict + MSE final.
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ridge α=1: penalização L2 fraca-média. Resolve forma fechada com αI na diagonal.
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)  # usa só o treino
ridge_predictions = ridge_model.predict(X_test)  # ŷ_teste = X_teste @ w_ridge + b

# Lasso α=1: penalização L1 (atenção: mesmo valor numérico ≠ mesma força que Ridge!
# As fórmulas têm normalizações diferentes: Ridge usa ||y-Xw||² + α||w||², Lasso
# usa (1/2n)||y-Xw||² + α||w||₁. α=1 no Lasso é proporcionalmente MUITO mais forte,
# daí esperarmos maior encolhimento / possível zeramento parcial aqui.)
lasso_model = Lasso(alpha=1)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

# Avaliar Ridge no teste: MSE = média(y_teste - ŷ)². Referência ~25 (ruído) + viés.
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("Ridge Regression MSE:", ridge_mse)

# Avaliar Lasso no teste: esperamos valor maior que o Ridge neste caso denso
# (a esparsidade não ajuda quando w_verdade=[2,3] ambos ativos).
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print("Lasso Regression MSE:", lasso_mse)
# Extensão sugerida: imprimir ridge_model.coef_ vs lasso_model.coef_ para ver o
# encolhimento diferencial (Lasso encolhe mais o w menor — efeito soft-threshold).
