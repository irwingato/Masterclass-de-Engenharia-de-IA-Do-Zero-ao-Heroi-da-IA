# =============================================================================
# MINI PROJETO DAY 7 — Parte 1: REGRESSÃO (California Housing)
# Objetivo: prever o valor médio das casas (MedHouseVal) a partir de 3 atributos.
# =============================================================================
# COMPUTAÇÃO + MATEMÁTICA RESUMIDA:
# - Modelo: Regressão Linear múltipla -> y_hat = w0 + w1*MedInc + w2*HouseAge + w3*AveRooms
#   Em forma matricial: y_hat = X @ w (com coluna de 1s embutida no intercepto).
# - Treino: o sklearn resolve mínimos quadrados, ou seja, minimiza a soma dos
#   resíduos ao quadrado:  min_w ||X_train @ w - y_train||².
#   Solução fechada (equação normal): w* = (X^T X)^-1 X^T y
#   (na prática o sklearn usa SVD, mais estável que inverter X^T X).
# - Avaliação: MSE = (1/n) * soma((y_test - y_pred)²). Penaliza erros grandes
#   ao quadrado; unidade = (unidade do alvo)². Aqui o alvo está em US$100k,
#   então MSE = 0.5 significa RMSE ~= sqrt(0.5) ~= 0.707 (~US$70.700).
# =============================================================================

# Task 1: EDA e Preprocessing
from sklearn.datasets import fetch_california_housing  # dataset clássico do sklearn (1990, Califórnia)
from sklearn.model_selection import train_test_split  # divisão treino/teste aleatória
from sklearn.linear_model import LinearRegression  # modelo linear de mínimos quadrados
from sklearn.metrics import mean_squared_error  # métrica MSE

# --- Load Dataset ---
# fetch_california_housing(as_frame=True) baixa (1ª vez) e devolve um Bunch com `.frame`
# (DataFrame pandas com 20640 linhas x 9 colunas: 8 features + 1 alvo).
# Colunas usadas aqui:
#   MedInc     = renda média do quarteirão (em US$10k) — feature mais preditiva
#   HouseAge   = idade média da casa (anos)
#   AveRooms   = nº médio de cômodos por domicílio
#   MedHouseVal= valor médio da casa (em US$100k) — este é o ALVO (y)
data = fetch_california_housing(as_frame=True)
df = data.frame  # atalho: DataFrame completo (features + alvo)

# --- Define features e target ---
# X: matriz (n_amostras, 3) — só 3 das 8 colunas originais, para simplificar o Day 7.
# y: vetor (n_amostras,) contínuo -> por isso é REGRESSÃO, não classificação.
X = df[['MedInc', 'HouseAge', 'AveRooms']]
y = df['MedHouseVal']

# # --- Inspect data (EDA) ---
# # df.info(): tipos, nº de linhas, memória e nulos por coluna.
# # df.describe(): count/mean/std/min/quartis/max — detecta escala e outliers
# # (ex.: MedInc ~ 0-15, HouseAge ~ 1-52: escalas diferentes, mas a Regressão
# #  Linear fechada não *exige* padronização; ela só ajuda na interpretação).
# # print(df.info())
# # print(df.describe())

# # --- Visualize relationships ---
# # sns.pairplot() desenha scatter plots 2 a 2 + histogramas na diagonal.
# # Serve para ver linearidade (MedInc x MedHouseVal é quase linear) e
# # colinearidade entre features. Custa O(k²) gráficos; com k=4 é ok.
# # sns.pairplot(df, vars=['MedInc', 'AveRooms', 'HouseAge', 'MedHouseVal'])
# # plt.show()

# # --- Check for missing values ---
# # df.isnull().sum(): conta NaNs por coluna. O California Housing não tem NaNs,
# # então este é um check de sanidade antes do treino (NaN quebraria o .fit()).
# # print("Missing Values: \n", df.isnull().sum())

# --- Split Dataset ---
# train_test_split embaralha e separa 80% treino / 20% teste.
# Computação: ~16512 treino / ~4128 teste. `random_state=42` fixa a semente do
# gerador pseudo-aleatório -> resultado reproduzível entre execuções.
# OBS: sem `stratify` (só faz sentido em classificação); em regressão o embaralho
# puro basta porque y é contínuo.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train linear regression ---
# LinearRegression().fit(X_train, y_train) estima w* pelos mínimos quadrados.
# Custo: O(n * p² + p³) com p=3 features -> instantâneo aqui.
# Atributos aprendidos depois do fit: model.coef_ (w1..w3) e model.intercept_ (w0).
model = LinearRegression()
model.fit(X_train, y_train)

# --- Make Predictions ---
# predict(X_test): para cada linha i, calcula y_hat_i = w0 + soma(wj * x_ij).
# Vetorizado: y_pred = X_test @ coef_ + intercept_ -> vetor de ~4128 valores.
y_pred = model.predict(X_test)

# --- Evaluate performance ---
# mean_squared_error(y_test, y_pred) = média de (y - y_hat)² no conjunto de TESTE
# (mede generalização, não decoreba do treino). Quanto menor, melhor.
# Referência típica com essas 3 features: MSE ~ 0.52-0.55. Com as 8 features: ~0.53?-> ~0.50.
mse = mean_squared_error(y_test, y_pred)
print("Linear Regression MSE:", mse)
