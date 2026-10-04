# =============================================================================
# Day 5 — Regressão Linear vs. Regressão Polinomial (Bike Sharing)
# -----------------------------------------------------------------------------
# OBJETIVO COMPUTACIONAL:
#   Prever a demanda diária de aluguel de bicicletas (`cnt`) a partir da
#   temperatura normalizada (`temp`) e comparar um modelo linear simples
#   com um modelo polinomial de grau 2.
#
# OBJETIVO MATEMÁTICO:
#   Modelo Linear:      y = b0 + b1*x + erro
#   Modelo Polinomial:  y = b0 + b1*x + b2*x² + erro
#   Ambos são ajustados por Mínimos Quadrados Ordinários (OLS) e avaliados
#   pelo Erro Quadrático Médio (MSE).
# =============================================================================

# --- 1. IMPORTS --------------------------------------------------------------
import pandas as pd  # Computação: biblioteca de DataFrames; aqui serve para ler o CSV e manipular colunas
from sklearn.preprocessing import PolynomialFeatures  # Computação: transforma X -> [X, X², ..., X^d]
from sklearn.linear_model import LinearRegression  # Computação: resolvedor OLS da regressão linear
from sklearn.model_selection import train_test_split  # Computação: divisão treino/teste aleatória e reprodutível
from sklearn.metrics import mean_squared_error  # Computação: calcula o MSE entre y verdadeiro e y previsto

# --- 2. CARGA DOS DADOS ------------------------------------------------------
# Computação: `read_csv` lê o arquivo texto, infere tipos e devolve um DataFrame
# (tabela 2D com linhas = dias, colunas = variáveis).
df = pd.read_csv("bike_sharing_daily.csv")

# Display dataset information
# Computação: `df.info()` mostraria nº de linhas, nº de não-nulos e dtype de cada
# coluna. Está comentado para não poluir a saída; descomente para auditar valores
# nulos/tipos antes de modelar.
# print("Dataset Info:")
# print(df.info())

# Preview the first few rows
# Computação: `df.head()` retorna as 5 primeiras linhas — inspeção visual rápida
# para conferir nomes de colunas (`temp`, `cnt`, `dteday`, ...) e escalas.
# print("\n Dataset Preview:")
# print(df.head())

# --- 3. ENGENHARIA DE ATRIBUTOS TEMPORAIS ------------------------------------
# Computação: `pd.to_datetime` converte a string "2011-01-01" em tipo datetime64,
# o que libera o acessor `.dt` para extrair partes da data de forma vetorizada
# (sem loop Python, em C por baixo dos panos -> O(n) rápido).
# Matemática: datas são variáveis cíclicas/ordinais; extrair dia-da-semana, mês
# e ano permite ao modelo capturar sazonalidade (ex.: fim de semana vs. dia útil).
df['dteday'] = pd.to_datetime(df['dteday'])

# Create new features
# .dt.dayofweek: segunda=0 ... domingo=6 (convenção do pandas).
# .dt.month: 1-12. .dt.year: 2011, 2012.
# NOTA: havia um typo no original (`daf_of_week`); corrigido para `day_of_week`.
df['day_of_week'] = df['dteday'].dt.dayofweek
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

# Display the new features
# Computação: seleciona só as 4 colunas para conferir se a extração funcionou.
# print("\n New Features:")
# print(df[['dteday', 'day_of_week', 'month', 'year']].head())

# --- 4. SELEÇÃO DE FEATURE (X) E ALVO (y) ------------------------------------
# Computação: `X` é DataFrame 2D (n, 1) — o sklearn exige 2D mesmo com 1 coluna.
# `y` é Series 1D (n,) com o total de aluguéis no dia (`casual + registered`).
# Matemática: estamos modelando E[cnt | temp] = f(temp). Só `temp` foi usada;
# `day_of_week/month/year` foram criadas mas não entram no modelo (poderiam!).
# `temp` já vem normalizada em [0,1] no dataset original (temp real / max).
X = df[['temp']]
y = df['cnt']

# --- 5. TRANSFORMAÇÃO POLINOMIAL ---------------------------------------------
# Matemática: regressão polinomial de grau 2 postula:
#   y_i = b0 + b1*x_i + b2*x_i² + e_i
# É "linear nos parâmetros" (b0,b1,b2), então o mesmo algoritmo OLS resolve,
# mas "não-linear na variável" — a parábola captura curvatura (ex.: demanda
# sobe com calor até certo ponto e depois cai no calor extremo).
# Computação: `PolynomialFeatures(degree=2, include_bias=False)` mapeia:
#   [x] -> [x, x²]
# `include_bias=False` evita adicionar coluna de 1s, pois `LinearRegression`
# já ajusta o intercepto b0 via `fit_intercept=True` (padrão).
poly = PolynomialFeatures(degree=2, include_bias=False)

# ATENÇÃO — vazamento de dados (data leakage), versão didática correta:
# O original fazia `fit_transform` no dataset inteiro ANTES do split e depois
# chamava `train_test_split` duas vezes. Com mesmo `random_state` até alinha,
# mas é frágil e conceitualmente errado. O correto é: dividir primeiro, fazer
# `fit` só no treino e `transform` no teste. Para expansão polinomial pura o
# `fit` não aprende estatística dos dados (só registra o grau), então o vazamento
# é inofensivo aqui — mas com StandardScaler, por exemplo, seria grave.
# Por isso comentamos o `fit_transform` global e fazemos o procedimento certo
# após o split (ver seção 6). Mantido aqui como referência visual:
# X_poly = poly.fit_transform(X)

# Display the tranformed feature
# Computação: embrulha a matriz numpy em DataFrame nomeado para inspeção:
# coluna 1 = temp, coluna 2 = temp².
# print("\n Original and Polynomial Features")
# print(pd.DataFrame(poly.fit_transform(X), columns=['temp', 'temp^2']).head())

# --- 6. DIVISÃO TREINO / TESTE -----------------------------------------------
# Matemática/ML: holdout 80/20 estima o erro de generalização. Treina-se em 80%
# e mede-se o MSE nos 20% nunca vistos — simula "dados futuros".
# Computação: `train_test_split(..., test_size=0.2, random_state=42)` embaralha
# com gerador pseudoaleatório de semente 42 -> reprodutível. Retorna
# X_train (80%), X_test (20%), y_train, y_test alinhados por índice.
# Complexidade: O(n) para embaralhar + fatiar.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Computação: agora sim expandimos a base. `fit_transform` no treino aprende/
# aplica [x, x²]; `transform` no teste só aplica a mesma regra, sem reaprender.
# Resultado: X_poly_train shape (n_train, 2), X_poly_test shape (n_test, 2).
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# --- 7. MODELO 1: REGRESSÃO LINEAR SIMPLES -----------------------------------
# Matemática (OLS): encontra b0, b1 que minimizam a soma dos quadrados:
#   min_{b0,b1} Σ_i (y_i - b0 - b1*x_i)²
# Solução fechada (equação normal): b = (X^T X)^{-1} X^T y, custo O(n·p² + p³)
# com p=1 feature -> baratíssimo. O sklearn usa SVD, mais estável numericamente.
# Computação: `fit` resolve os coeficientes; `predict` calcula b0 + b1*x no teste
# (produto matriz-vetor O(n·p)); `mean_squared_error` calcula:
#   MSE = (1/n_test) · Σ (y_true - y_pred)²
# MSE está em (aluguéis)² — penaliza quadraticamente erros grandes.
model_original = LinearRegression()
model_original.fit(X_train, y_train)  # aprende b0 (intercept_) e b1 (coef_)
y_pred_original = model_original.predict(X_test)  # aplica a reta aos 20% de teste
mse_original = mean_squared_error(y_test, y_pred_original)  # erro do modelo linear

# --- 8. MODELO 2: REGRESSÃO POLINOMIAL (GRAU 2) -------------------------------
# Matemática: mesmo OLS, agora com matriz de desenho [1, x, x²]:
#   min_{b0,b1,b2} Σ_i (y_i - b0 - b1*x_i - b2*x_i²)²
# A parábola tem 3 parâmetros (p=2 + intercepto) -> mais flexível, menor viés,
# porém maior variância (risco de overfitting se o grau fosse alto).
# Computação: idêntico ao anterior, só que `fit` recebe 2 colunas em vez de 1.
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)  # aprende b0, b1, b2
y_pred_poly = model_poly.predict(X_poly_test)  # avalia a parábola no teste
mse_poly = mean_squared_error(y_test, y_pred_poly)  # erro do modelo quadrático

# --- 9. COMPARAÇÃO -----------------------------------------------------------
# Matemática da decisão: menor MSE no teste = melhor generalização (neste split).
# Esperado: MSE_polinomial < MSE_linear se a relação temp×demanda for curva.
# Cuidado: comparar só 1 split é ruidoso; o ideal seria validação cruzada
# (k-fold) + inspeção de resíduos e R². Também notar: usar só `temp` ignora
# confundidores (estação, feriado, ano), então o MSE absoluto segue alto.
# Computação: f-string com `:.2f` formata com 2 casas decimais.
print(f"MSE original: {mse_original:.2f}")
print(f"MSE Polynomial: {mse_poly:.2f}")