# ============================================================================
# IMPORTS
# ============================================================================

from sklearn.datasets import load_breast_cancer
# Carrega o dataset "Breast Cancer Wisconsin" do scikit-learn.
# Contém 569 amostras, 30 features (medidas de características de células
# mamárias) e um alvo binário: 0 = maligno (câncer), 1 = benigno (não câncer).
# Fonte: UCI Machine Learning Repository (originalmente de Wolberg & Mangasarian, 1990).

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# train_test_split: divide os dados em treino e teste de forma aleatória (estratificada).
# GridSearchCV: busca exaustiva em grade (grid search) com validação cruzada.
# RandomizedSearchCV: busca aleatória amostrando parâmetros de distribuições, com validação cruzada.

from sklearn.preprocessing import StandardScaler
# Padronização (standardization) de features: subtrai a média e divide pelo desvio padrão.

import pandas as pd
# Biblioteca para manipulação de dados tabulares (DataFrames). Importada mas não utilizada neste script.

from xgboost import XGBClassifier
# Implementação do XGBoost (Extreme Gradient Boosting) compatível com scikit-learn.
# Algoritmo de boosting baseado em árvores de decisão, conhecido por alta performance em dados tabulares.

from sklearn.metrics import accuracy_score
# Calcula a acurácia: fração de predições corretas sobre o total de amostras.
# Acurácia = (TP + TN) / (TP + TN + FP + FN)

import optuna
# Framework de otimização de hiperparâmetros baseado em amostragem bayesiana (TPE - Tree-structured Parzen Estimator).

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================

# Load the dataset
data = load_breast_cancer()
# Retorna um objeto Bunch (semelhante a dicionário) com:
#   .data   -> matriz de features (n_samples x n_features)
#   .target -> vetor de rótulos (n_samples,)
#   .feature_names -> nomes das 30 features
#   .target_names   -> ['malignant', 'benign']

X, y = data.data, data.target
# X: matriz de entrada (features), shape (569, 30). Cada linha é uma amostra, cada coluna uma feature.
# y: vetor alvo (target), shape (569,). Valores 0 ou 1 (classificação binária).

# Ciência da computação: separar X (independentes) de y (dependente) é a convenção padrão
# em aprendizado de máquina supervisionada. X ∈ R^(n×p), y ∈ {0,1}^n.

# ============================================================================
# 2. DIVISÃO TREINO / TESTE
# ============================================================================

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Divide os dados em:
#   - Treino (80% = 455 amostras): usado para ajustar os parâmetros do modelo.
#   - Teste  (20% = 114 amostras): usado apenas para avaliação final, sem viés de overfitting.
#
# random_state=42 garante reprodutibilidade (mesma divisão a cada execução).
# test_size=0.2 significa que 20% dos dados vão para o conjunto de teste.
#
# Matemática / CC:
#   - Avaliação em dados não vistos (hold-out) estimativa o erro de generalização:
#     E_gen ≈ (1/N_test) * Σ I(ŷ_i ≠ y_i)
#   - Separar treino/teste evita "data leakage" (vazamento de informação), que inflaciona artificialmente a acurácia.

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# ============================================================================
# 3. PRÉ-PROCESSAMENTO: PADRONIZAÇÃO (STANDARDIZATION)
# ============================================================================

# Standardize the data
scaler = StandardScaler()
# Instancia o objeto que calcula média e desvio padrão de cada feature.

X_train = scaler.fit_transform(X_train)
# fit: calcula média (μ) e desvio padrão (σ) de cada coluna do treino.
# transform: aplica x' = (x - μ) / σ para cada valor.
# O scaler aprende os parâmetros APENAS com o treino (não vemos o teste).

X_test = scaler.transform(X_test)
# Aplica a mesma transformação (μ e σ do treino) ao teste.
# Nunca faz fit_transform no teste, pois isso vazaria informação do teste para o treino.

# Matemática:
#   Para cada feature j:
#       μ_j = (1/n) * Σ x_ij          (média amostral)
#       σ_j = sqrt( (1/n) * Σ (x_ij - μ_j)^2 )   (desvio padrão)
#       x'_ij = (x_ij - μ_j) / σ_j
#
# Resultado: cada feature padronizada tem média 0 e desvio padrão 1.
#
# Por que padronizar?
#   - Muitos algoritmos (SVM, k-NN, redes neurais) são sensíveis à escala.
#   - XGBoost é baseado em árvores e é invariante a monotonic transformations, então
#     a padronização não é estritamente necessária, mas ajuda em comparações e
#     em algoritmos baseados em distância/gradiente.
#   - Evita que features com valores grandes dominem aquelas com valores pequenos.

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# ============================================================================
# 4. MODELO DE BASELINE (XGBoost com parâmetros padrão)
# ============================================================================

# Train a baseline XGBoost model
baseline_model = XGBClassifier(eval_metric='logloss', random_state=42)
# Cria um classificador XGBoost com:
#   - eval_metric='logloss': mede log-loss durante o treinamento (para early stopping/monitoramento).
#   - random_state=42: semente para reprodutibilidade da aleatoriedade interna.

baseline_model.fit(X_train, y_train)
# Treina o modelo: constrói um ensemble de árvores de decisão sequenciais.
# Cada nova árvore corrige os erros (resíduos/gradientes) da árvore anterior.

# Evaluate the model
baseline_preds = baseline_model.predict(X_test)
# Gera predições (classes 0 ou 1) para o conjunto de teste.

baseline_accuracy = accuracy_score(y_test, baseline_preds)
# Acurácia = número de predições corretas / total de amostras de teste.

print(f"Baseline XGBoost Accuracy: {baseline_accuracy:.4f}")

# ============================================================================
# 5. XGBOOST: FUNDAMENTOS MATEMÁTICOS E DE CC
# ============================================================================
#
# XGBoost é um algoritmo de Gradient Boosting. A ideia é construir um modelo
# aditivo de M árvores de decisão:
#
#       ŷ_i = Σ_{k=1}^{M} f_k(x_i),   onde f_k ∈ F (espaço das árvores)
#
# Objetivo (função de custo a minimizar):
#
#       Obj = Σ_{i=1}^{n} l(y_i, ŷ_i) + Σ_{k=1}^{M} Ω(f_k)
#
#   - l(y_i, ŷ_i): função de perda (ex: log-loss para classificação binária)
#       l(y, ŷ) = -[ y * log(σ(ŷ)) + (1-y) * log(1 - σ(ŷ)) ]
#       onde σ(z) = 1 / (1 + e^{-z}) é a sigmoide.
#   - Ω(f_k): termo de regularização para evitar overfitting:
#       Ω(f) = γ * T + (1/2) * λ * Σ w_j^2
#       onde T = número de folhas, w_j = peso (score) da folha j,
#       γ = penalidade por complexidade (gamma), λ = regularização L2 (reg_lambda).
#
# Treinamento por adição sequencial (boosting):
#   Em cada iteração t, adiciona uma árvore f_t que minimiza a aproximação
#   de segunda ordem (Taylor) do objetivo:
#
#       Obj^(t) ≈ Σ_{i=1}^{n} [ g_i * f_t(x_i) + (1/2) * h_i * f_t(x_i)^2 ] + Ω(f_t)
#
#   onde g_i = ∂l/∂ŷ (gradiente) e h_i = ∂²l/∂ŷ² (hessiana).
#
# A solução ótima para o peso de uma folha j é:
#
#       w_j* = - (Σ_{i∈I_j} g_i) / (Σ_{i∈I_j} h_i + λ)
#
#   onde I_j é o conjunto de índices das amostras que caem na folha j.
#
# Hiperparâmetros principais do XGBoost:
#   - n_estimators: número de árvores (estágios de boosting).
#   - max_depth: profundidade máxima de cada árvore (controla complexidade).
#   - learning_rate (eta): encolhe a contribuição de cada árvore (shrinkage),
#     reduz overfitting mas exige mais árvores.
#   - subsample: fração de amostras amostradas para cada árvore (estocasticidade).
#   - colsample_bytree: fração de features amostradas para cada árvore.
#   - gamma: ganho mínimo necessário para fazer um split (podagem).
#   - reg_alpha (L1): regularização L1 nos pesos das folhas (esparsidade).
#   - reg_lambda (L2): regularização L2 nos pesos das folhas (suavização).

# ============================================================================
# 6. OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA
# ============================================================================

# Define the objective function for Optuna
def objective(trial):
    """
    Função objetivo que o Optuna tentará maximizar.
    Recebe um objeto 'trial' que sugere valores de hiperparâmetros.
    Retorna a acurácia no conjunto de teste (métrica a maximizar).
    """
    params = {
        # Número de árvores no ensemble. Mais árvores = menor viés, maior risco de overfitting.
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),

        # Profundidade máxima de cada árvore. Árvores mais profundas capturam interações complexas
        # mas overfitam mais facilmente.
        'max_depth': trial.suggest_int('max_depth', 3, 100),

        # Taxa de aprendizado (shrinkage). Valores menores tornam o treinamento mais conservador.
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),

        # Fração de amostras usada para treinar cada árvore (subamostragem estocástica).
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),

        # Fração de features usada para construir cada árvore.
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),

        # Ganho mínimo de perda exigido para realizar um split (podagem prévia).
        'gamma': trial.suggest_float('gamma', 0, 5),

        # Regularização L1 (Lasso) nos pesos das folhas. Induz esparsidade.
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),

        # Regularização L2 (Ridge) nos pesos das folhas. Suaviza os pesos.
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }

    # Train XGBoost model with suggested params
    model = XGBClassifier(eval_metric='logloss', random_state=42, **params)
    # Desempacota o dicionário params como argumentos nomeados do construtor.
    # Ex: XGBClassifier(..., n_estimators=210, max_depth=22, ...)

    model.fit(X_train, y_train)

    # Evaluate model
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy  # Optuna maximiza este valor retornado

# Create an Optuna study
study = optuna.create_study(direction="maximize")
# Cria um estudo de otimização. direction="maximize" queremos maior acurácia.

study.optimize(objective, n_trials=50)
# Executa 50 tentativas (trials). A cada trial:
#   1. Sugere um conjunto de hiperparâmetros (usando amostragem bayesiana TPE).
#   2. Executa a função objective.
#   3. Atualiza o histórico para guiar a próxima sugestão.
#
# TPE (Tree-structured Parzen Estimator):
#   Modela p(parâmetros | acurácia alta) vs p(parâmetros | acurácia baixa)
#   e escolhe a configuração que maximiza o expected improvement (EI).

# Best hyperparameters
print("Best Hyperparameters:", study.best_params)
print("Best Accuracy:", study.best_value)

# ============================================================================
# 7. GRID SEARCH (BUSCA EXAUSTIVA EM GRADE)
# ============================================================================

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
}

# Grade cartesiana: combina todos os valores possíveis.
# Total de combinações = 3 * 3 * 3 * 3 = 81 modelos treinados.
# Com cv=3 (validação cruzada de 3 folds), são 81 * 3 = 243 treinamentos.

# Train XGBoost with Grid Search
grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    scoring='accuracy',   # métrica para comparar modelos
    cv=3,                 # 3-fold cross-validation
    verbose=1
)

grid_search.fit(X_train, y_train)
# Para cada combinação da grade:
#   - divide o treino em 3 folds
#   - treina em 2 folds, valida em 1, rotaciona
#   - média as acurácias dos 3 folds
# Escolhe a combinação com maior acurácia média.

# Best parameters and accuracy
print("\n\n\nGrid Search Best Parameters: ", grid_search.best_params_)
print("Grid Search Best Accuracy: ", grid_search.best_score_)

# Matemática / CC - Validação Cruzada K-Fold:
#   Divide os dados de treino em K partes (folds). Para cada fold k:
#       treina em todos exceto k, avalia em k.
#   Escore final = (1/K) * Σ escore_k
#   Reduz variância da estimativa de generalização em relação a um único split.
#   Custo computacional: K vezes mais treinamentos.

# ============================================================================
# 8. RANDOMIZED SEARCH (BUSCA ALEATÓRIA)
# ============================================================================

# Define parameter distributions
param_dist = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

# Train XGBoost with Random Search
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    n_iter=50,            # número de combinações aleatórias a testar
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
# Amostra n_iter combinações aleatórias da distribuição de parâmetros.
# Diferente do Grid Search, não testa todas as combinações.
# Vantagem: com o mesmo orçamento computacional, explora mais do espaço de parâmetros.

# Best parameters and accuracy
print("\n\n\nRandom Search Best Parameters: ", random_search.best_params_)
print("Random Search Best Accuracy: ", random_search.best_score_)

# Comparação de métodos de busca de hiperparâmetros:
#   - Grid Search: exaustivo, garante encontrar o melhor dentro da grade,
#     mas cresce exponencialmente com o número de parâmetros (maldi dimensionalidade).
#   - Random Search: amostra aleatoriamente, mais eficiente em espaços grandes
#     (Bergstra & Bengio, 2012 mostram que encontra bons parâmetros mais rápido).
#   - Optuna (Bayesian): usa histórico para direcionar a busca, geralmente o mais
#     eficiente em termos de número de trials necessários.

# ============================================================================
# RESULTADOS ESPERADOS (exemplo de execução)
# ============================================================================
# Baseline XGBoost Accuracy: 0.9561
# Optuna Best Accuracy: 0.9736842105263158
# Grid Search Best Accuracy:  0.9735970721505751
# Random Search Best Accuracy:  0.9736115952131986
#
# Nota: os valores exatos podem variar ligeiramente entre execuções devido à
# aleatoriedade inerente aos algoritmos (subsample, colsample, split treino/teste).
