import pandas as pd  # Manipulação de dados tabulares (DataFrame)
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Codificação categórica e padronização numérica
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score  # Divisão, busca aleatória e validação cruzada
from sklearn.ensemble import RandomForestClassifier  # Floresta aleatória (conjunto de árvores de decisão)
from sklearn.metrics import accuracy_score, classification_report  # Métricas de avaliação (acurácia, precisão, recall, F1)
import numpy as np  # Operações numéricas e geração de espaços de parâmetros

# --- CARREGAMENTO E ANÁLISE EXPLORATÓRIA DE DADOS ---
# Load dataset: lê o CSV em um DataFrame pandas (estrutura tabular de linhas/colunas)
df = pd.read_csv("Telco-Customer-Churn.csv")

# Display dataset info: mostra tipos, não-nulos e uso de memória de cada coluna
print("Dataset Info: \n")
print(df.info())
# Class Distribution: contagem de classes do alvo (Churn = 0/1 após encoding) — desbalanceamento afeta métricas
print("\n Class Distribution: \n")
print(df['Churn'].value_counts())
print("\n Sample Data: \n", df.head())  # primeiras 5 linhas para inspeção visual

# --- TRATAMENTO DE VALORES AUSENTES ---
# Handle missing values: 'TotalCharges' tem espaços em branco → converte para numérico, inváveis viram NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Imputação por mediana: substitui NaN pelo valor médio (50º percentil), robusto a outliers
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

# --- CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS ---
label_encoder = LabelEncoder()  # mapeia cada categoria única para um inteiro (0, 1, 2, ...)
for column in df.select_dtypes(include=['object']).columns:  # colunas do tipo string/object
    if column != 'Churn':  # não codifica o alvo ainda (mantém separado para clareza)
        # fit_transform aprende o mapeamento e converte: categorias → inteiros (ordinal encoding)
        df[column] = label_encoder.fit_transform(df[column])

# Encode target variable: Churn (Yes/No) → 1/0 (binário), necessário para classificação supervisionada
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# --- PADRONIZAÇÃO DE FEATURES NUMÉRICAS ---
scaler = StandardScaler()  # z-score normalization
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
# Matemática: z = (x - μ) / σ, onde μ = média e σ = desvio padrão. Resultado: média 0, desvio padrão 1.
# Importante: ajusta (fit) só nos dados de treino conceptualmente; aqui usa todo df antes do split (simplificação didática).
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# --- SEPARAÇÃO FEATURES / ALVO ---
X = df.drop(columns=['Churn'])  # matriz de features (entrada), shape (n_amostras, n_features)
y = df['Churn']                 # vetor alvo (saída), shape (n_amostras,)

# --- DIVISÃO TREINO / TESTE (Holdout Method) ---
# test_size=0.2 → 20% para teste, 80% para treino. random_state=42 garante reprodutibilidade.
# Ciência da computação: avaliação em dados não vistos evita overfitting (memorização do treino).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TREINAMENTO DO MODELO INICIAL (Random Forest) ---
# RandomForestClassifier: ensemble de árvores de decisão por bagging (Bootstrap Aggregating).
# n_estimators=50 → número de árvores; random_state=42 → semente para aleatoriedade controlada.
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
# fit: constrói 50 árvores, cada uma em um subconjunto bootstrap (amostragem com reposição)
# e em um subconjunto aleatório de features; divisão por critério de impureza (Gini ou Entropia).
rf_model.fit(X_train, y_train)

# --- AVALIAÇÃO DO MODELO INICIAL ---
# predict: cada árvore vota na classe; a classe majoritária vence (votação por maioria).
y_pred = rf_model.predict(X_test)
# accuracy = (TP + TN) / (TP + TN + FP + FN) — proporção de acertos totais (pode ser enganosa em classes desbalanceadas)
accuracy_initial = accuracy_score(y_test, y_pred)

print(f"Initial Model Accuracy: {accuracy_initial:.4f}")
# classification_report: precisão (TP/(TP+FP)), recall (TP/(TP+FN)), F1-score (média harmônica) e suporte por classe
print("\n Classification Report: \n", classification_report(y_test, y_pred))

# --- BUSCA ALEATÓRIA DE HIPERPARÂMETROS (RandomizedSearchCV) ---
param_grid = {
    'n_estimators': np.arange(50, 200, 10),   # amostra inteiros de 50 a 190 de 10 em 10
    'max_depth': [None, 5, 10, 15],           # profundidade máxima das árvores (None = ilimitada)
    'min_samples_split': [2, 5, 10, 20],      # mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4]             # mínimo de amostras em uma folha
}

# RandomizedSearchCV: amostra n_iter combinações aleatórias do grid (mais eficiente que Grid Search exaustivo)
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,                                # número de combinações testadas
    cv=5,                                     # validação cruzada k-fold (k=5): divide treino em 5 partes,
                                              # treina em 4 e valida em 1, rotacionando; média as acurácias.
    scoring='accuracy',                       # métrica otimizada
    n_jobs=-1,                                # usa todos os núcleos de CPU disponíveis (paralelismo)
    random_state=42
)

# Perform Randomized Search: executa n_iter × cv = 100 treinos e guarda o melhor
random_search.fit(X_train, y_train)

# Get best parameters: retorna o conjunto de hiperparâmetros com melhor média de validação cruzada
best_params = random_search.best_params_
print(f"Best Parameters (Randomized SearchCV): {best_params}")

# best_estimator_: modelo já treinado com os melhores parâmetros em todo o conjunto de treino
best_model = random_search.best_estimator_

# --- AVALIAÇÃO DO MODELO AJUSTADO ---
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Model Accuracy: {accuracy_tuned:.4f}")
print("\n Classification Report (Tuned Model): \n", classification_report(y_test, y_pred_tuned))

# --- VALIDAÇÃO CRUZADA FINAL (estimativa mais robusta do erro de generalização) ---
# cross_val_score: roda k-fold nos dados X, y inteiros e retorna acurácia de cada fold
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
# Média das acurácias dos 5 folds: estimativa do desempenho em dados não vistos (generalização)
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")