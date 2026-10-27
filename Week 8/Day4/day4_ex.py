from sklearn.datasets import fetch_california_housing   
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# MATEMÁTICA / CIÊNCIA DA COMPUTAÇÃO — CARREGAMENTO DE DADOS
# O dataset California Housing contém 20.640 amostras e 8 features:
#   MedInc (renda média), HouseAge (idade da casa), AveRooms (quartos médios),
#   AveBedrms (quartos médios), Population (população), AveOccup (ocupação média),
#   Latitude, Longitude.
# O target (y) é o valor médio da casa em centenas de milhares de dólares.
#
# Ciência da computação: fetch_california_housing faz download/cache do dataset
# do repositório da scikit-learn. Retorna um Bunch (dicionário-like) com .data
# (matriz X: n_amostras × n_features) e .target (vetor y: n_amostras).
# ---------------------------------------------------------------------------
# Load dataset
california = fetch_california_housing()
X, y = california.data, california.target
feature_names = california.feature_names

# ---------------------------------------------------------------------------
# MATEMÁTICA — DIVISÃO TREINO/TESTE
# train_test_split particiona os dados em conjuntos de treino (80%) e teste (20%).
# random_state=42 garante reprodutibilidade (seed fixa do gerador de números
# pseudo-aleatórios), essencial para depuração e comparação justa de modelos.
#
# Ciência da computação: dividir os dados evita overfitting e vazamento de
# informação (data leakage). O teste é feito em dados nunca vistos pelo modelo,
# estimando o erro de generalização (erro em dados novos).
# ---------------------------------------------------------------------------
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# MATEMÁTICA — EXPLORAÇÃO DE DADOS
# Exibe os nomes das features e as primeiras 5 linhas (head) como DataFrame.
# Isso serve para inspeção visual (sanity check): valores ausentes, escalas,
# distribuição, tipos de dados.
#
# Ciência da computação: pandas DataFrame é uma estrutura tabular rotulada
# (linhas × colunas) otimizada para manipulação e análise de dados.
# ---------------------------------------------------------------------------
# Display dataset info
print("Feature Names:\n", feature_names)
print("\n Sample Data:\n", pd.DataFrame(X, columns=feature_names).head())

# ---------------------------------------------------------------------------
# MATEMÁTICA — REGRESSÃO LINEAR MÚLTIPLA (SEM REGULARIZAÇÃO)
# Modelo: y = X·w + b, onde:
#   y ∈ R^n  (vetor alvo),
#   X ∈ R^(n×p) (matriz de features, p = 8),
#   w ∈ R^p  (vetor de coeficientes/pesos),
#   b ∈ R    (intercepto/bias).
#
# O objetivo é minimizar o Ersto Quadrático Médio (MSE):
#   J(w, b) = (1/n) · Σᵢ (yᵢ - (xᵢ·w + b))²   =   (1/n) · ||y - Xw - b||²
#
# Solução de forma fechada (Normal Equation):
#   w = (XᵀX)⁻¹ Xᵀ y   (quando XᵀX é invertível)
#
# Ciência da computação: LinearRegression usa OLS (Ordinary Least Squares).
# Complexidade aproximada: O(p²n + p³) para ajuste (produto XᵀX e inversão);
# predição: O(p) por amostra.
# Sem regularização, coeficientes podem crescer livremente → risco de overfitting
# e instabilidade numérica (multicolinearidade).
# ---------------------------------------------------------------------------
# Train linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predição no conjunto de teste (dados não vistos durante o treino)
y_pred = lr_model.predict(X_test)

# MSE = (1/n_test) · Σᵢ (y_testᵢ - y_predᵢ)²
mse_lr = mean_squared_error(y_test, y_pred)

print(f"Linear Regression MSE (No Regularization): {mse_lr:.4f}")
print("Coefficients:\n", lr_model.coef_)
# lr_model.coef_ é o vetor w (peso de cada feature);
# lr_model.intercept_ é o bias b.

# ---------------------------------------------------------------------------
# MATEMÁTICA — REGRESSÃO RIDGE (L2)
# Função de custo com penalidade L2:
#   J_ridge(w, b) = (1/n) · ||y - Xw - b||²  +  α · ||w||²
#                            = MSE  +  α · Σⱼ wⱼ²
#
# α (alpha) controla a força da regularização:
#   α → 0: se aproxima da regressão linear simples (sem penalidade).
#   α → ∞: coeficientes tendem a zero (modelo prediz a média).
#
# Solução de forma fechada:
#   w = (XᵀX + α·I)⁻¹ Xᵀ y
# A adição de α·I (identidade) garante que a matriz seja invertível mesmo com
# multicolinearidade (problema de condicionamento), estabilizando a solução.
#
# Efetuo: encolhe coeficientes em direção a zero, mas raramente os anula
# exatamente. Reduz variância ao custo de um pequeno viés (bias-variance
# tradeoff).
#
# Ciência da computação: Ridge resolve um sistema linear regularizado.
# Complexidade similar à OLS, mas com matriz melhor condicionada (número de
# condição menor → solução numérica mais estável).
# ---------------------------------------------------------------------------
# Train Ridge regression model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Predição e avaliação Ridge (linhas duplicadas no original; mantidas conforme arquivo)
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"Ridge Regression MSE: {mse_ridge:.4f}")
print("Coefficient:\n", ridge_model.coef_)

# ---------------------------------------------------------------------------
# MATEMÁTICA — REGRESSÃO LASSO (L1)
# Função de custo com penalidade L1:
#   J_lasso(w, b) = (1/n) · ||y - Xw - b||²  +  α · ||w||₁
#                            = MSE  +  α · Σⱼ |wⱼ|
#
# Diferente da L2, a penalidade L1 não é diferenciável em wⱼ = 0, o que
# produz esparsidade: coeficientes de features irrelevantes são exatamente
# zerados. Isso executa seleção automática de features.
#
# Não há solução de forma fechada simples; é resolvido por otimização iterativa
# (coordinate descent, subgradiente, proximal gradient).
#
# Efetuo: produz modelos mais interpretáveis (menos features) e é útil quando
# muitas features são ruído.
#
# Ciência da computação: Lasso usa otimização convexa iterativa. Mais custoso
# computacionalmente que Ridge/OLS, mas resulta em modelo esparso (menor uso de
# memória no caso de muitos coeficientes zero).
# ---------------------------------------------------------------------------
# Train Lasso regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print(f"Lasso Regression MSE: {mse_lasso:.4f}")
print("Coefficient:\n", lasso_model.coef_)