# Imports: bibliotecas de ML para dados, divisão, modelos ensemble, métricas
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================
# Dataset Wisconsin Breast Cancer: 569 amostras, 30 features numéricas
# Classes: 0 = maligno, 1 = benigno (target binário)
# Matematicamente: X ∈ ℝ^(569×30), y ∈ {0,1}^569
data = load_breast_cancer()
X, y = data.data, data.target

# Divisão treino/teste (80/20) com seed fixa para reprodutibilidade
# CS: Holdout method — evita data leakage, estima erro de generalização
# random_state garante mesma partição a cada execução (determinismo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Informações do dataset: nomes das 30 features (mean, se, worst de raio, textura, etc.)
print(f"Features: {data.feature_names}")
print(f"Classes: {data.target_names}")

# ============================================================
# GRADIENT BOOSTING CLASSIFIER (GBC)
# ============================================================
# Ensemble boosting: sequencial, cada árvore corrige resíduos da anterior
# Matemática: F_m(x) = F_{m-1}(x) + η * h_m(x)
#   - η = learning_rate (shrinkage): passo de atualização, previne overfitting
#   - h_m = weak learner (Decision Tree) ajustado ao pseudo-resíduo r_i = -∂L/∂F
#   - Loss L = log-loss (binário): -[y log(p) + (1-y)log(1-p)]
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Inferência: soma ponderada das árvores → probabilidade → classe via threshold 0.5
y_pred_gb = gb_model.predict(X_test)

# Métricas: accuracy = (TP+TN)/(TP+TN+FP+FN)
# Classification report: precision, recall, f1-score por classe + macro/weighted avg
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb}")
print("\nClassification Report: \n", classification_report(y_test, y_pred_gb))

# ============================================================
# BUSCA EM GRADE (GRID SEARCH) PARA HIPERPARÂMETROS
# ============================================================
# Hiperparâmetros do GBC:
# - learning_rate (η): controle do passo de gradiente (menor = mais árvores necessárias)
# - n_estimators (M): número de árvores/iterações de boosting
# - max_depth: profundidade máx de cada árvore fraca (controla complexidade/overfitting)
# CORREÇÃO: 'learning_rate' (um 'n'), não 'learnning_rate'
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

# GridSearchCV: validação cruzada k-fold (k=5) exaustiva sobre o grid
# CS: Busca em espaço discreto — O(|grid| × k × treino)
# scoring='accuracy': otimiza acurácia média nos folds de validação
# n_jobs=-1: paraleliza em todos os cores (CPU-bound)
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# best_params_: combinação que maximizou accuracy média nos 5 folds
# best_score_: accuracy de validação cruzada (não é test accuracy!)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================
# Ensemble bagging: árvores independentes, treinadas em bootstrap samples
# Matemática: média de preditores — reduz variância sem aumentar viés
#   - Cada árvore vê subset aleatório de features (feature bagging) → decorrelação
#   - Predição final: votação majoritária (classificação) ou média (regressão)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:}")