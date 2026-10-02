# ============================================================================
# day2_ex.py - Efeito da Normalização no k-NN usando o dataset Iris
# Objetivo: comparar a acurácia do k-NN (k=5) em 3 cenários:
#   1) sem escala, 2) com Min-Max Scaling, 3) com Standardization (Z-score)
# ============================================================================
from sklearn.datasets import load_iris  # Função que carrega o dataset clássico Iris (150 flores, 4 medidas, 3 espécies)
from sklearn.model_selection import train_test_split  # Função para dividir dados em treino/teste de forma aleatória e reprodutível
from sklearn.neighbors import KNeighborsClassifier  # Implementação do classificador k-NN (baseado em distância)
from sklearn.metrics import accuracy_score  # Métrica: fração de acertos = corretos / total
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # Duas técnicas de reescala de features (ver matemática abaixo)
import pandas as pd  # Biblioteca para manipular tabelas (DataFrame)

# Load Iris dataset
# load_iris() retorna um dicionário-like com:
#   - data: matriz X de formato (150, 4) -> 150 amostras, 4 features:
#     [sepal length, sepal width, petal length, petal width] todas em cm
#   - target: vetor y de formato (150,) com rótulos 0, 1, 2
#   - target_names: ['setosa', 'versicolor', 'virginica']
#   - feature_names: nomes das 4 colunas
# Convertemos X para DataFrame do pandas para facilitar a análise (describe(), etc.)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display dataset Information
# X.describe() calcula estatísticas descritivas por coluna:
#   count = nº de amostras (150), mean = média μ = (1/n)Σxi,
#   std = desvio-padrão amostral σ = sqrt((1/(n-1))Σ(xi-μ)²),
#   min/max = extremos, 25%/50%/75% = quartis (50% = mediana).
# Serve para ver escalas diferentes: ex. sepal length ~ 4-8cm, petal width ~ 0-2.5cm.
# data.target_names mostra as 3 classes que o modelo deve prever.
print("Dataset Info:")
print(X.describe())
print("\n Target Classes:", data.target_names)

# Split the dataset
# train_test_split divide X e y em treino (80%) e teste (20%):
#   test_size=0.2 -> 30 amostras para teste, 120 para treino.
#   random_state=42 -> fixa a semente aleatória para o resultado ser reprodutível.
# Por que dividir? O modelo só aprende no treino e é avaliado no teste (dados nunca vistos),
# simulando o uso real e detectando overfitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train k-NN classifier
# k-NN (k-Nearest Neighbors) com n_neighbors=5:
#   1. NÃO tem fase de "aprender pesos" — apenas memoriza X_train (aprendizado preguiçoso/lazy).
#   2. Para cada ponto de teste q, calcula a distância Euclidiana a todos os pontos de treino x:
#        d(x, q) = sqrt( Σ_{j=1..4} (x_j - q_j)² )
#   3. Seleciona os k=5 vizinhos mais próximos e faz votação majoritária:
#        ŷ = moda(y dos 5 vizinhos)  (classe mais frequente vence).
#   k=5 (ímpar) evita empate em problema binário e suaviza ruído vs k=1.
#   Ponto crítico: como usa distância, features com escala maior dominam o cálculo -> por isso testamos scaling abaixo.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)  # Aqui "fit" só armazena os dados de treino na memória do objeto

# Predict and evaluate
# knn.predict(X_test) aplica a regra dos 5 vizinhos para cada uma das 30 amostras de teste.
# accuracy_score(y_test, y_pred) = Acurácia = (nº acertos) / (nº total) = (1/n)Σ 1[ŷ_i == y_i]
# Varia de 0 a 1 (ex. 0.97 = 97% de acerto). É adequada aqui pois as 3 classes são balanceadas (50 cada).
y_pred = knn.predict(X_test)
print("Accuracy Without Scaling:", accuracy_score(y_test, y_pred))

# Apply Min-Max Scaling
# MinMaxScaler transforma cada feature para o intervalo [0, 1] pela fórmula (aplicada por coluna):
#   X' = (X - X_min) / (X_max - X_min)
# Ex.: se petal length vai de 1.0 a 6.9, um valor 3.0 vira (3.0-1.0)/(6.9-1.0) ≈ 0.34.
# fit_transform(X) faz duas coisas: fit = calcula X_min e X_max do dataset; transform = aplica a fórmula.
# Efeito no k-NN: iguala a amplitude de todas as features, nenhuma domina a distância Euclidiana por ser "numericamente maior".
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# SPlit scaled data
# Divide os dados JÁ escalados com o mesmo test_size e random_state, garantindo comparação justa
# com o caso sem escala (os mesmos 30 índices caem no teste, só que agora com valores em [0,1]).
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train k-NN classifier on scaled data
# Mesmo modelo k=5, mas treinado/avaliado no espaço normalizado [0,1].
# A distância Euclidiana agora é calculada sobre X' em vez de X, mudando quem são os "5 vizinhos mais próximos".
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_scaled)

# Predict and evaluate
# Compara a acurácia com Min-Max vs sem escala para ver se equalizar as amplitudes ajudou.
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy with Min-Max Scaling:", accuracy_score(y_test_scaled, y_pred_scaled))

# Apply Standardization
# StandardScaler (Z-score / padronização) centraliza e reduz cada feature pela fórmula (por coluna):
#   z = (X - μ) / σ
# onde μ = média da coluna no treino e σ = desvio-padrão da coluna.
# Resultado: cada feature passa a ter média 0 e variância 1 (valores típicos ficam entre ~[-2, +2], sem limite rígido).
# Diferença vs Min-Max: Min-Max força [0,1] e é sensível a outliers (um outlier estica X_max);
# Z-score não tem limite fixo e lida melhor com distribuição ~gaussiana / outliers moderados.
# fit_transform calcula μ e σ de X e aplica a fórmula.
scaler = StandardScaler()
X_stand = scaler.fit_transform(X)

# SPlit standardized data
# Mesmo esquema 80/20 com random_state=42 para comparação justa entre os 3 cenários.
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_stand, y, test_size=0.2, random_state=42)

# Train k-NN classifier on standardized data
# Novamente k-NN com k=5, agora no espaço padronizado (média 0, desvio 1).
# Nota de boas práticas: o ideal seria fazer fit do scaler SÓ no X_train e depois transform em X_test
# (evita data leakage — vazar info do teste para o treino). Aqui o fit foi no X todo, o que é didático mas otimista.
knn_stand = KNeighborsClassifier(n_neighbors=5)
knn_stand.fit(X_train_std, y_train_std)

# Predict and evaluate
# Terceira acurácia: permite concluir qual pré-processamento é melhor para este dataset + k-NN.
# No Iris as escalas já são parecidas (tudo em cm), então a diferença costuma ser pequena,
# mas em datasets com escalas muito distintas (ex. idade vs salário) o scaling muda drasticamente o resultado.
y_pred_std = knn_stand.predict(X_test_std)
print("Accuracy with Standardization:", accuracy_score(y_test_std, y_pred_std))