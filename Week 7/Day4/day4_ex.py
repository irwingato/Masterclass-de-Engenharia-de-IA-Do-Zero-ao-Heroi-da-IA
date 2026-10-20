import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================
# Dataset Breast Cancer Wisconsin (Diagnóstico)
# - 569 amostras, 30 features numéricas (médias, erros, piores valores de núcleos celulares)
# - Target binário: 0 = maligno, 1 = benigno
# - Problema de classificação binária supervisionada
data = load_breast_cancer()
X, y = data.data, data.target  # X: matriz (n_samples, n_features), y: vetor (n_samples,)

# =============================================================================
# DIVISÃO TREINO/TESTE (Holdout Method)
# =============================================================================
# train_test_split implementa amostragem estratificada aleatória
# - test_size=0.2: 20% para teste (114 amostras), 80% para treino (455 amostras)
# - random_state=42: seed fixa para reprodutibilidade (determinismo)
# - Estratificação mantém proporção de classes em ambos conjuntos
# Matematicamente: particiona D = D_train ∪ D_test, D_train ∩ D_test = ∅
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exibição de metadados do dataset
print(f"Features: {data.feature_names}")
print(f"Classes: {data.target_names}")

# =============================================================================
# XGBOOST NATIVO (API de baixo nível) - DMatrix
# =============================================================================
# DMatrix: estrutura de dados otimizada do XGBoost (columnar, sparse-aware)
# - Armazena features + labels + metadados (pesos, base_margin, etc.)
# - Permite computação eficiente de histogramas para splits
# - Thread-safe para treinamento paralelo
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# =============================================================================
# HIPERPARÂMETROS XGBOOST (Teoria do Boosting)
# =============================================================================
# XGBoost = Extreme Gradient Boosting (Chen & Guestrin, 2016)
# Minimiza: L(θ) = Σ l(y_i, ŷ_i) + Σ Ω(f_k)
#   - l: loss function (logloss para classificação binária)
#   - Ω: regularização (complexidade da árvore: γT + ½λ||w||²)
params = {
    'objective': 'binary:logistic',  # Loss: logloss = -[y log(p) + (1-y) log(1-p)]
    'eval_metric': 'logloss',        # Métrica de validação (monitora overfitting)
    'max_depth': 3,                  # Profundidade máxima da árvore (regularização estrutural)
    'eta': 0.1                       # Learning rate (shrinkage): η ∈ (0,1], passo do gradiente
}

# Treinamento: Gradient Boosting iterativo
# Para t = 1 to T (num_boost_round=100):
#   1. Calcula gradientes g_i = ∂l/∂ŷ, hessianas h_i = ∂²l/∂ŷ²
#   2. Ajusta árvore f_t nos pseudo-resíduos (g_i, h_i)
#   3. Atualiza: ŷ_i^(t) = ŷ_i^(t-1) + η · f_t(x_i)
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# =============================================================================
# PREDIÇÃO E AVALIAÇÃO
# =============================================================================
# predict() retorna probabilidades p(y=1|x) ∈ [0,1]
# Threshold 0.5: decisão bayesiana ótima para 0-1 loss com priors iguais
y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)

# Métricas de avaliação
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2PR/(P+R)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =============================================================================
# BUSCA DE HIPERPARÂMETROS (Grid Search + Cross-Validation)
# =============================================================================
# Grid Search: busca exaustiva no espaço cartesiano de hiperparâmetros
# Cross-Validation (k-fold): estimação de erro de generalização
#   - Particiona D_train em k folds, treina k vezes, média dos scores
#   - Reduz variância da estimativa vs holdout único
# Complexidade: |param_grid| × k treinos = 3×3×3×2×2 × 5 = 540 treinos
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],      # η: passo do gradiente (trade-off viés-variância)
    'n_estimators': [50, 100, 200],         # T: nº de árvores (early stopping implícito via CV)
    'max_depth': [3, 5, 7],                 # Complexidade da árvore (overfitting se alto)
    'subsample': [0.8, 1.0],                # Row sampling (stochastic gradient boosting)
    'colsample_bytree': [0.8, 1.0]          # Feature sampling (reduz correlação entre árvores)
}

# XGBClassifier: wrapper scikit-learn compatível com GridSearchCV
xgb_clf = XGBClassifier(eval_metric='logloss', random_state=42)

# GridSearchCV com paralelização (n_jobs=-1 = todos os cores)
# Refaz treino completo para cada combinação (não incremental)
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=5,                    # 5-fold CV
    scoring='accuracy',      # Função objetivo: acurácia
    n_jobs=-1                # Paralelismo máximo
)
grid_search.fit(X_train, y_train)

# Resultados da busca
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

# =============================================================================
# GRADIENT BOOSTING CLASSIFIER (scikit-learn implementation)
# =============================================================================
# Implementação clássica de Friedman (2001) - Gradient Boosting Machines
# Similar ao XGBoost mas sem:
#   - Regularização L1/L2 nos pesos das folhas
#   - Aproximação de histogramas (usa splits exatos)
#   - Paralelismo nativo no nível de árvore
#   - Tratamento nativo de missing values
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Avaliação comparativa
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb}")
print("\nClassification Report: \n", classification_report(y_test, y_pred_gb))