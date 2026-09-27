# --- Imports ---
# load_iris: carrega o dataset Iris (Fisher, 1936).
#   Matemática: X tem shape (150, 4) -> 150 amostras, 4 features
#   (comprimento/largura de sépala e pétala). y tem shape (150,)
#   com 3 classes balanceadas {0: setosa, 1: versicolor, 2: virginica}, 50 cada.
from sklearn.datasets import load_iris
# cross_val_score: executa treino/avaliação repetidos e retorna 1 score por fold.
# KFold: gerador de partições para validação cruzada.
#   Matemática: divide D em k folds disjuntos D1..Dk, |Di| ≈ n/k.
#   Na iteração i: treina em D\Di e testa em Di.
from sklearn.model_selection import cross_val_score, KFold
# RandomForestClassifier: ensemble de árvores de decisão via bagging.
#   Matemática: treina B árvores, cada uma num bootstrap (amostragem com
#   reposição) de X e num subconjunto aleatório de features por split.
#   Cada split minimiza impureza de Gini: G = 1 - soma_k(p_k^2),
#   onde p_k = proporção da classe k no nó.
#   Predição final = voto majoritário: y_hat = moda{h_1(x),...,h_B(x)}.
from sklearn.ensemble import RandomForestClassifier

# --- Carga dos dados ---
# data.data = matriz X (150x4), data.target = vetor y (150,)
data = load_iris()
X, y = data.data, data.target

# --- Modelo ---
# random_state=42: fixa a aleatoriedade (bootstrap + sorteio de features)
# para reprodutibilidade. Não muda a matemática, só o resultado sorteado.
model = RandomForestClassifier(random_state=42)

# --- Validação cruzada K-Fold ---
# n_splits=5 -> k=5 folds de ~30 amostras cada (150/5).
# shuffle=True + random_state=42: embaralha antes de partir, de forma
# reprodutível. Sem shuffle, como o Iris vem ordenado por classe,
# os folds ficariam enviesados (ex: fold só com uma classe).
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# scoring="accuracy": para cada fold i calcula:
#   Acc_i = (1/|Di|) * soma 1[y_hat == y] = traço da matriz de confusão / total
# cross_val_score retorna vetor [Acc_1, ..., Acc_5].
cv_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

# --- Resultados ---
# cv_scores: estimativa da distribuição do desempenho.
# A média é o estimador do erro de generalização:
#   Acc_media = (1/k) * soma_i(Acc_i) ≈ E[acurácia em dados novos]
# Por que CV e não 1 único split? Um único split tem alta variância
# (depende do sorteio). A média dos k folds reduz essa variância e
# usa 100% dos dados para teste (cada ponto é testado 1x).
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
# Dica: cv_scores.std() mede a instabilidade: std alto = modelo sensível
# à partição dos dados (possível overfitting ou dados heterogêneos).
