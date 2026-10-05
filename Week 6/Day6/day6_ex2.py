# ============================================================
# DAY 6 EX 2 - REGRESSÃO LINEAR (PREVISÃO DE VALORES CONTÍNUOS)
# Dataset: California Housing (~20k bairros, 8 features -> preço mediano)
# Objetivo: prever o valor contínuo da casa (regressão, não classificação)
# ============================================================

# --- 1. IMPORTS ---
# sklearn.datasets.fetch_california_housing: baixa/carrega dataset de regressão real.
# train_test_split: divisão holdout treino/teste.
# LinearRegression: regressor linear por Mínimos Quadrados Ordinários (OLS).
# mean_absolute_error / mean_squared_error / r2_score: métricas de regressão.
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# CORREÇÃO: o original importava mean_squared_error 2x e nunca importava mean_absolute_error.
# Ciência da Computação: isso é um bug silencioso (o MAE era calculado como MSE).
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 2. CARREGAMENTO DOS DADOS ---
# fetch_california_housing() retorna Bunch com:
#   .data   -> X de shape (~20640, 8): MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
#   .target -> y = preço mediano em $100k (variável contínua, ex: 2.5 = $250k)
# Matemática: X ∈ R^(n×d), n≈20k, d=8. y ∈ R^n (regressão: y contínuo, não discreto como na classificação).
data = fetch_california_housing()
X, y = data.data, data.target

# --- 3. DIVISÃO TREINO / TESTE ---
# Mesmo princípio do ex1: 80% treino, 20% teste, semente fixa p/ reprodutibilidade.
# Importância extra em regressão: o teste mede erro de generalização em valores nunca vistos.
# Sem split, o R² no treino seria otimista (overfitting, especialmente com muitas features).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TREINO DA REGRESSÃO LINEAR ---
# Matemática da Regressão Linear (OLS):
#   1. Hipótese: ŷ = w^T x + b = w_1*x_1 + ... + w_d*x_d + b.
#      Geometricamente: um hiperplano em R^d que melhor atravessa a nuvem de pontos.
#   2. Custo (MSE no treino): J(w,b) = 1/m * Σ (y_i - ŷ_i)².
#      É convexo (paraboloide em dimensão alta) -> mínimo global único.
#   3. Solução fechada (Equação Normal): w* = (X^T X)^(-1) X^T y.
#      sklearn usa na prática SVD (Decomposição em Valores Singulares) por estabilidade
#      numérica — evita inverter X^T X diretamente, que pode ser singular/mal-condicionada.
#      Complexidade: O(n*d²) — eficiente aqui (20k x 8), mas caro se d >> milhares.
#   4. Diferença p/ Regressão Logística (ex1): lá passamos z por uma sigmoide e
#      minimizamos log-loss p/ classes; aqui minimizamos erro quadrático p/ valores reais.
model = LinearRegression()
model.fit(X_train, y_train)  # encontra w*, b* ótimos. Depois acessíveis via model.coef_ e model.intercept_

# --- 5. PREDIÇÃO ---
# Computação: para cada x_test, calcula ŷ = X_test @ w* + b* (produto matricial vetorizado, O(k*d)).
# Retorna y_pred ∈ R^k contínuo (ex: [1.2, 3.4, 0.8, ...]).
y_pred = model.predict(X_test)

# --- 6. AVALIAÇÃO: MÉTRICAS DE REGRESSÃO ---
# Matemática das 3 métricas:
#
#   a) MAE (Mean Absolute Error) = 1/n * Σ |y_i - ŷ_i|
#      - Penalidade linear: cada $1 de erro conta igual.
#      - Robusto a outliers. Unidade = mesma do y ($100k). Mediana minimiza o MAE.
#
#   b) MSE (Mean Squared Error) = 1/n * Σ (y_i - ŷ_i)²
#      - Penalidade quadrática: erros grandes pesam MUITO mais (ex: erro 10 pesa 100x erro 1).
#      - Diferenciável em todo ponto (por isso é usado como custo de treino). Média minimiza o MSE.
#      - Unidade = y² (difícil interpretar; por isso às vezes usa-se RMSE = √MSE, que volta à unidade original).
#
#   c) R² (Coeficiente de Determinação) = 1 - SS_res / SS_tot, onde:
#        SS_res = Σ (y_i - ŷ_i)²  (resíduo do modelo)
#        SS_tot = Σ (y_i - ȳ)²    (variância total, modelo baseline que sempre prevê a média)
#      - R² = 1   -> modelo perfeito. R² = 0 -> tão bom quanto prever sempre a média.
#      - R² < 0   -> pior que a média (modelo muito ruim / overfitting severo).
#      - Interpretação: fração da variância de y explicada por X.
mae = mean_absolute_error(y_test, y_pred)  # CORRIGIDO: antes chamava mean_squared_error aqui por engano
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- 7. EXIBIÇÃO ---
# :.2f formata com 2 casas decimais. Ex: MAE=0.53 significa erro médio de ~$53k.
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2- Score(R2): {r2:.2f}")
