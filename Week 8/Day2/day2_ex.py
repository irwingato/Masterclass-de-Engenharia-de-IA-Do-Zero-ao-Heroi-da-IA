from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# =============================================================================
# IMPORTS
# load_iris: carrega o conjunto de dados Iris (Fisher, 1936) - 150 amostras,
# 4 features (sépala/pétala: comprimento e largura) e 3 classes (setosa,
# versicolor, virginica). É um problema de classificação supervisionada
# multiclasse.
#
# train_test_split: divide os dados em treino e teste. RandomState controla
# a semente do gerador de números aleatórios (PRNG) para reprodutibilidade.
#
# GridSearchCV / RandomizedSearchCV: buscas exaustiva e aleatória sobre um
# espaço de hiperparâmetros com validação cruzada k-fold.
#
# RandomForestClassifier: ensemble de árvores de decisão (Bagging). Cada
# árvore é treinada em um bootstrap sample (amostragem com reposição) e em
# um subconjunto aleatório de features. A predição final é a votação por
# maioria (classe) ou média (regressão).
#
# accuracy_score: métrica de classificação. Acurácia = (VP + VN) / Total,
# onde VP = verdadeiros positivos, VN = verdadeiros negativos.
#
# numpy: suporte a arrays multidimensionais e álgebra linear.
# =============================================================================

# =============================================================================
# CARREGAMENTO DO DATASET
# data.data  -> matriz X de shape (n_samples, n_features) = (150, 4)
# data.target -> vetor y de shape (n_samples,) com rótulos inteiros {0,1,2}
# =============================================================================
# Load dataset
data = load_iris()
X, y = data.data, data.target

# =============================================================================
# DIVISÃO TREINO / TESTE (Hold-out)
# test_size=0.2  => 30 amostras para teste, 120 para treino (80/20).
# random_state=42 => semente fixa do PRNG: mesma divisão a cada execução
# (reprodutibilidade, essencial para experimentos científicos).
# Estratégia: avaliar desempenho em dados nunca vistos pelo modelo,
# estimando o erro de generalização E_gen = E[L(f(x), y)] sobre a distribuição
# real dos dados, aproximado pelo erro no conjunto de teste.
# =============================================================================
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dsiplay dataset info
print(f"Features: {data.feature_names}")
print(f"Classes: {data.target_names}")

# =============================================================================
# ESPAÇO DE HIPERPARÂMETROS (Grid Search)
# n_estimators: número de árvores na floresta. Mais árvores reduzem variância
# (Lei dos Grandes Números / média de estimadores), mas aumentam custo O(T)
# onde T = n_estimators. Retornos decrescentes após certo ponto.
# max_depth: profundidade máxima de cada árvore. Controla viés/variância:
#  - profunda => baixo viés, alta variância (risco de overfitting)
#  - rasa => alto viés, baixa variância (underfitting)
#  - None => cresce até folhas puras (máximo complexity)
# min_samples_split: número mínimo de amostras para dividir um nó. Evita
# divisões em ruído; aumenta o viés e reduz variância.
# Complexidade do Grid Search: O(|Grid| × cv × C_treino), onde |Grid| =
# 3 × 3 × 3 = 27 combinações, cv=5 => 135 treinos de RandomForest.
# =============================================================================
# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# =============================================================================
# GRID SEARCH COM VALIDAÇÃO CRUZADA K-FOLD (cv=5)
# O conjunto de treino é particionado em 5 folds (partições). Para cada
# combinação de hiperparâmetros:
#   - treina em 4 folds, valida no 1º fold (rotacionando)
#   - média das 5 acurácias = estimativa do desempenho daquela combinação
#   - escolhe-se a combinação com maior acurácia média
# Vantagem: reduz variância da estimativa comparado a um único hold-out.
# Desvantagem: custo computacional linear em |Grid| × k.
# n_jobs=-1 => paraleliza em todos os núcleos da CPU (data parallelism).
# scoring='accuracy' => função objetivo a maximizar.
# =============================================================================
# Initialize Grid Search
grid_search = GridSearchCV(
    estimator = RandomForestClassifier(random_state=42),
    param_grid = param_grid,
    cv = 5,
    scoring = 'accuracy',
    n_jobs = -1
)

# Perform Grid Search
grid_search.fit(X_train, y_train)

# =============================================================================
# AVALIAÇÃO DO MELHOR MODELO (Grid Search)
# best_estimator_ => instância do RandomForest com os melhores hiperparâmetros
# encontrados, já treinada em todo o conjunto de treino.
# predict(X_test) => para cada árvore, prediz classe; votação majoritária
# define a classe final.
# accuracy_score(y_test, y_pred) => fração de predições corretas no teste.
# .4f => formata com 4 casas decimais.
# =============================================================================
# Evaluate bvest model
best_grid_model = grid_search.best_estimator_
y_pred = best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred)

print(f"Best Hyperparameters (Grid Search): {grid_search.best_params_}")
print(f"Grid Search Accuracy: {accuracy_grid:.4f}")

# =============================================================================
# DISTRIBUIÇÃO DE HIPERPARÂMETROS (Random Search)
# Diferente do grid, amostra n_iter=20 combinações aleatoriamente das
# distribuições especificadas (np.arange gera intervalo uniforme discreto).
# Teoria (Bergstra & Bengio, 2012): busca aleatória é mais eficiente que
# grid quando poucos hiperparâmetros dominam o desempenho, porque explora
# o espaço de forma mais uniforme com o mesmo orçamento computacional.
# n_iter controla o trade-off precisão vs custo: O(n_iter × cv × C_treino).
# =============================================================================
# Define hyperparameter distribution
param_dist = {
    'n_estimators': np.arange(50, 200),
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10, 20]
}

# =============================================================================
# RANDOMIZED SEARCH COM VALIDAÇÃO CRUZADA
# Semelhante ao GridSearchCV, mas amostra n_iter pontos do espaço de
# busca em vez de testar todas as combinações. Útil para espaços grandes
# ou contínuos (aqui ainda é discreto, mas amostrado aleatoriamente).
# random_state=42 garante que as 20 amostras aleatórias sejam reproduzíveis.
# =============================================================================
# Initialize Random Search
random_search = RandomizedSearchCV(
    estimator = RandomForestClassifier(random_state=42),
    param_distributions = param_dist,
    n_iter=20,
    cv=5,
    scoring = 'accuracy',
    n_jobs = -1,
    random_state=42
)

# Perform Random Search
random_search.fit(X_train, y_train)

# =============================================================================
# AVALIAÇÃO DO MELHOR MODELO (Random Search)
# Mesma lógica do Grid Search: predição no conjunto de teste e cálculo da
# acurácia. Permite comparar qual estratégia de busca encontrou um modelo
# com melhor generalização nos dados de teste.
# =============================================================================
# Evaluate best model
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print(f"Best Hyperparameters (Random Search): {random_search.best_params_}")
print(f"Random Search Accuracy: {accuracy_random:.4f}")