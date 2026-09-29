# =============================================================================
# MINI PROJETO DAY 7 — Parte 2: CLASSIFICAÇÃO (Telco Customer Churn)
# Objetivo: prever se o cliente vai cancelar (churn = 1) ou não (churn = 0).
# =============================================================================
# COMPUTAÇÃO + MATEMÁTICA RESUMIDA:
# - É CLASSIFICAÇÃO BINÁRIA (y discreto 0/1), diferente da regressão do Projeto 1.
# - Pipeline: EDA -> codificar texto -> separar treino/teste -> padronizar ->
#   treinar 2 modelos -> comparar com precision/recall/F1 + matriz de confusão.
# - StandardScaler (z-score): z = (x - media_treino) / desvio_treino.
#   Deixa cada coluna com média 0 e variância 1. ESSENCIAL para k-NN e ajuda a
#   Regressão Logística a convergir (gradiente desce melhor em escala uniforme).
# - Regressão Logística: modela p(y=1|x) = sigmoide(z) = 1 / (1 + exp(-(w·x + b))).
#   Treino minimiza o log-loss: -[y*log(p) + (1-y)*log(1-p)], via L-BFGS por padrão.
#   `max_iter` = nº máx. de iterações do otimizador; se baixo demais, não converge.
# - k-NN (k=5): sem treino paramétrico — decora o treino. Para cada ponto de teste,
#   calcula a distância euclidiana d = sqrt(soma((x_i - x_j)²)) até todos os pontos
#   de treino, pega os 5 mais próximos e vota (maioria vira a predição).
#   Por isso o k-NN SOFRE com escala: feature em dólares (0-7000) dominaria feature
#   0/1 se não padronizar. Complexidade de predição: O(n_treino * d) por query.
# - Métricas (para a classe 1 = churn):
#     Precision = TP/(TP+FP) — de quem alertei como churn, quantos acertaram?
#     Recall    = TP/(TP+FN) — de quem realmente churnou, quantos capturei?
#     F1        = 2*P*R/(P+R) — média harmônica (pune extremo num só lado).
#   Matriz de confusão [[TN FP],[FN TP]] mostra os 4 quadrantes brutos.
# =============================================================================

# Task 1: EDA e Preprocessing
import pandas as pd  # DataFrames, leitura de CSV, get_dummies
import seaborn as sns  # countplot da distribuição do churn
import matplotlib.pyplot as plt  # título/layout/show do gráfico
from pathlib import Path  # caminho robusto: funciona de qualquer pasta (BASE_DIR / csv)
from sklearn.model_selection import train_test_split  # split estratificado
from sklearn.preprocessing import StandardScaler  # padronização z-score
from sklearn.linear_model import LogisticRegression  # classificador linear probabilístico
from sklearn.neighbors import KNeighborsClassifier  # classificador por vizinhança
from sklearn.metrics import classification_report, confusion_matrix  # métricas

# --- Load Telco Customer Churn Dataset ---
# BASE_DIR = pasta deste script; assim `python Day7/mini_....py` funciona mesmo se o
# terminal estiver em outra pasta (evita FileNotFoundError com caminho relativo puro).
# CSV real (500 linhas x 9 colunas): customer_id, gender, age, tenure,
# monthly_charges, total_charges, contract_type, payment_method, churn (0/1).
BASE_DIR = Path(__file__).resolve().parent
df_telco = pd.read_csv(BASE_DIR / 'Telco-Customer-Churn.csv')

# --- EDA rápida (antes estava no fim do arquivo e comentada; EDA vem ANTES do treino) ---
# df.info(): 500 non-null em tudo -> sem NaNs; dtypes: 4 int64, 2 float64, 3 str (object).
# df.describe(include="all"): para numéricas dá mean/std/quartis; para texto dá unique/top/freq
# (ex.: contract_type tem 3 valores, top=Monthly ~175; churn: mean ~0.31 -> 31% churnam, base desbalanceada).
# print(df_telco.info())
# print(df_telco.describe(include="all"))

# --- Visualize churn distribution ---
# sns.countplot(x='churn'): gráfico de barras 0 vs 1 — mostra o desbalanço (~69% vs 31%).
# Importante porque acurácia sozinha engana: chutar sempre "0" já dá 69%!
# sns.countplot(x='churn', data=df_telco)
# plt.title("Churn Distribution")
# plt.show()

# --- Handle missing values ---
# ATENÇÃO ao original comentado `df_telco.fillna(df_telco.mean())`: .mean() só funciona
# em colunas numéricas; com colunas de texto ele ignora/quebra conforme versão do pandas.
# Forma correta seria só nas numéricas: df_telco[numeric_cols].fillna(mediana...).
# Aqui nem é preciso: o CSV não tem nulos (isnull().sum() == 0 em tudo).
# df_telco.fillna(df_telco.mean(numeric_only=True), inplace=True)

# --- Encode categorical variables ---
# POR QUE NÃO LabelEncoder aqui? O original fazia:
#   le.fit_transform(gender/contract_type/payment_method)
# Isso cria ordem artificial (ex.: Monthly=0, Annual=1, Two-Year=2) e o modelo lê isso
# como "Two-Year é 2x Annual" — distância matemática que NÃO existe em categoria nominal.
# Para árvore até passa, mas para LogReg/k-NN (que usam produto escalar/distância) distorce.
# CORRETO: one-hot encoding — cada categoria vira uma coluna 0/1:
#   gender -> gender_Male; contract_type -> Annual/Two-Year (drop_first remove 1 p/ evitar
#   colinearidade perfeita); payment_method -> 3 colunas. Sem ordem inventada.
# 'churn' JÁ é 0/1 numérico -> não precisa codificar nada.
# 'customer_id' é identificador único (1..500) -> NÃO é feature (vazamento/ruído); remover.
df_model = df_telco.drop(columns=['customer_id'])  # remove ID antes de tudo
X = df_model.drop(columns=['churn'])  # matriz de features ainda com texto
y = df_model['churn'].astype(int)  # vetor alvo binário 0/1
X = pd.get_dummies(X, columns=['gender', 'contract_type', 'payment_method'], drop_first=True)
# Resultado: de 7 features (3 texto + 4 numéricas) -> ~10 colunas todas numéricas 0/1 + contínuas.

# --- Scale Features + Split dataset ---
# ORDEM CORRETA: split PRIMEIRO, scaler DEPOIS (fit só no treino).
# O original fazia fit_transform em TUDO e depois dividia -> data leakage: média/desvio
# do teste vazavam para o treino, inflando a métrica de forma otimista.
# `stratify=y` preserva os 31% de churn em treino e teste (sem isso, o teste poderia
# ficar com 20% ou 40% por azar do embaralho, enviesando precision/recall).
# `random_state=42`: reprodutibilidade do embaralho.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()  # instancia o z-score: guarda media_ e scale_ do treino
X_train = scaler.fit_transform(X_train)  # fit (aprende media/desvio) + transform no treino
X_test = scaler.transform(X_test)  # SÓ transform no teste (usa media/desvio do treino)

# --- Train logistic regression model ---
# LogisticRegression(max_iter=1000): otimizador L-BFGS tem até 1000 iterações para
# minimizar o log-loss. Original usava 200 — geralmente converge, mas 1000 dá margem
# sem custo relevante aqui (n=400 treino, d~10 -> treino em ms).
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)  # aprende w (coef_) e b (intercept_)

# --- Train k-NN model ---
# KNeighborsClassifier(n_neighbors=5): guarda X_train na memória (lazy learning).
# k=5 ímpar evita empate no voto binário; k pequeno = fronteira flexível (variância alta),
# k grande = fronteira lisa (viés alto). k=5 é o default sensato de partida.
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)  # na prática só memoriza os vetores + rótulos

# --- Evaluate models ---
# predict(): LogReg aplica sigmoide e limiar 0.5 (p>=0.5 -> 1); k-NN vota nos 5 vizinhos.
# Cada vetor tem 100 predições (20% de 500).
log_pred = log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

# classification_report: precision/recall/F1 por classe + macro avg (média simples das
# classes) + weighted avg (média ponderada pelo suporte). Suporte = nº real de cada
# classe no teste (~69 zeros, ~31 uns). Se o modelo nunca prevê "1", precision/recall
# da classe 1 zeram (UndefinedMetricWarning) — sinal de que ele colapsou para a maioria.
print("\n Logistic Regression Classification report:")
print(classification_report(y_test, log_pred))

print("\n k-NN Classification report:")
print(classification_report(y_test, knn_pred))

# --- Confusion Matrix ---
# confusion_matrix(y_test, y_pred) = [[TN FP],[FN TP]] (linhas=real, colunas=predito).
# Leitura: TN = acertou "não churn", FP = alarme falso, FN = churn perdido, TP = churn pego.
# O original só imprimia a da LogReg; aqui imprimimos as duas para comparar os trade-offs
# (LogReg costuma ter precision maior, k-NN recall diferente conforme a fronteira local).
print("Confusion Matrix (LogReg): \n", confusion_matrix(y_test, log_pred))
print("Confusion Matrix (k-NN): \n", confusion_matrix(y_test, knn_pred))
