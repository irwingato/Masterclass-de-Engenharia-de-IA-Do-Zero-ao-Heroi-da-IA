# =============================================================================
# IMPORTS
# =============================================================================
# load_breast_cancer: Dataset clássico de classificação binária (diagnóstico de câncer de mama)
#   - 569 amostras, 30 features numéricas (raio, textura, perímetro, área, etc.)
#   - Target: 0 = maligno, 1 = benigno
# train_test_split: Divide dados em treino/teste para avaliação honesta (evita data leakage)
# RandomForestClassifier: Ensemble de Decision Trees via bagging (Bootstrap Aggregating)
# accuracy_score, classification_report: Métricas de avaliação
# GridSearchCV: Busca exaustiva de hiperparâmetros com validação cruzada

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================
# Matematicamente: X ∈ ℝ^(n×d) onde n=569 amostras, d=30 features
#                  y ∈ {0,1}^n (classificação binária)
# Ciência da Computação: Estrutura de dados densa (numpy array) para computação vetorizada eficiente
data = load_breast_cancer()
X, y = data.data, data.target

# =============================================================================
# DIVISÃO TREINO/TESTE (Holdout Method)
# =============================================================================
# Matemática: Particiona (X,y) em D_train ∪ D_test = D, D_train ∩ D_test = ∅
#   - test_size=0.2 → 80% treino (455 amostras), 20% teste (114 amostras)
#   - random_state=42 → seed fixa para reprodutibilidade (determinismo algorítmico)
# Ciência da Computação: Amostragem estratificada preserva distribuição de classes
#   - Evita viés de seleção; garante que ambas classes apareçam em proporções similares
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# TREINAMENTO DO RANDOM FOREST
# =============================================================================
# MATEMÁTICA - Random Forest (Breiman, 2001):
#   - Ensemble de B decision trees: {h_1(x), h_2(x), ..., h_B(x)}
#   - Cada árvore treinada em bootstrap sample (amostragem com reposição, ~63.2% dados únicos)
#   - Feature bagging: em cada split, considera apenas m = max_features features aleatórias
#     * Classification: m = √d (padrão 'sqrt') ou log2(d) ('log2')
#   - Predição final: maioria (votação) para classificação
#       ŷ = mode({h_b(x)}_{b=1}^B)
#   - Reduz variância sem aumentar viés (diferente de boosting)
#
# CIÊNCIA DA COMPUTAÇÃO:
#   - Paralelizável: cada árvore independente → n_jobs=-1 usa todos os cores
#   - Complexidade treino: O(B * n * log(n) * m) 
#   - Complexidade inferência: O(B * depth) por amostra
#   - random_state=42: controla aleatoriedade (bootstrap + feature selection) para reprodutibilidade
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# =============================================================================
# PREDIÇÃO
# =============================================================================
# Matemática: Para cada x ∈ X_test, computar ŷ = mode({h_b(x)})
# Ciência da Computação: Batch prediction vetorizada (mais eficiente que loop)
y_pred = rf_model.predict(X_test)

# =============================================================================
# AVALIAÇÃO DE DESEMPENHO
# =============================================================================
# MÉTRICAS MATEMÁTICAS:
#   - Accuracy = (TP + TN) / (TP + TN + FP + FN) = acertos / total
#   - Precision = TP / (TP + FP) = de positivos previstos, quantos são reais
#   - Recall (Sensitivity) = TP / (TP + FN) = de positivos reais, quantos acertamos
#   - F1-Score = 2 * (Precision * Recall) / (Precision + Recall) = média harmônica
#   - Support = número de amostras reais de cada classe
#
# classification_report computa métricas por classe + médias macro/weighted
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy: ", accuracy)
print("\n Classification Report: \n", classification_report(y_test, y_pred))

# =============================================================================
# OTIMIZAÇÃO DE HIPERPARÂMETROS - GRID SEARCH COM VALIDAÇÃO CRUZADA
# =============================================================================
# ESPAÇO DE BUSCA (param_grid):
#   - n_estimators: nº de árvores no ensemble [50, 100, 200]
#     * Mais árvores → menor variância, mas retorna decrescente + custo computacional
#   - max_depth: profundidade máxima da árvore [None, 10, 20]
#     * None = cresce até folhas puras (risco overfitting)
#     * Limitado = regularização (bias-variance tradeoff)
#   - max_features: nº features consideradas por split ['sqrt', 'log2', None]
#     * 'sqrt' = √30 ≈ 5 features (padrão classificação)
#     * 'log2' = log2(30) ≈ 4 features (mais diversidade entre árvores)
#     * None = todas 30 features (equivale a bagging puro, menos diversidade)
#
# VALIDAÇÃO CRUZADA K-FOLD (cv=5):
#   - Particiona D_train em 5 folds de tamanho igual
#   - Para cada combinação de hiperparâmetros: treina em 4 folds, valida em 1
#   - Repete 5 vezes (cada fold como validação uma vez)
#   - Score final = média dos 5 scores de validação
#   - Matemática: CV estimate = (1/K) Σ_{k=1}^K Score(D_train \ Fold_k, Fold_k)
#
# GRID SEARCH:
#   - Busca exaustiva: |param_grid| = 3 × 3 × 3 = 27 combinações
#   - Total fits = 27 × 5 (cv folds) = 135 modelos treinados
#   - n_jobs=-1: paraleliza os 135 fits em todos cores CPU
#   - scoring='accuracy': métrica para seleção do melhor modelo
#
# CIÊNCIA DA COMPUTAÇÃO:
#   - Complexidade: O(|grid| × K × custo_treino)
#   - Tradeoff: exaustivo garante ótimo global no grid, mas custo exponencial em dimensões
#   - Alternativas: RandomizedSearchCV (amostragem aleatória), Bayesian Optimization
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# =============================================================================
# RESULTADOS DA OTIMIZAÇÃO
# =============================================================================
# best_params_: dicionário com combinação que maximizou CV score
# best_score_: média da accuracy nos 5 folds para best_params_
# best_estimator_: modelo já re-treinado em TODO D_train com best_params_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")