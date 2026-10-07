# ============================================================
# DAY 7 PROJECT: Classificação de Sobreviventes do Titanic
# Objetivo: prever `Survived` (0 = morreu, 1 = sobreviveu)
# Pipeline clássico de ML supervisionado: carga -> limpeza ->
# pré-processamento -> validação cruzada -> tuning (GridSearch)
# ============================================================

# --- 1. IMPORTS ---
import pandas as pd  # Manipulação tabular (DataFrame). Base: NumPy arrays + índices.
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Normalização numérica + codificação categórica
from sklearn.compose import ColumnTransformer  # Aplica transformações diferentes por coluna em paralelo
from sklearn.model_selection import cross_val_score  # Validação cruzada (estimativa robusta de generalização)
from sklearn.linear_model import LogisticRegression  # Classificador linear probabilístico
from sklearn.ensemble import RandomForestClassifier  # Ensemble de árvores (bagging)
from sklearn.model_selection import GridSearchCV  # Busca exaustiva de hiperparâmetros com CV interna

# --- 2. CARGA DOS DADOS ---
# Loading Titanic dataset
# pd.read_csv baixa o CSV do GitHub e monta um DataFrame (tabela 2D).
# Computação: O(n*m) para parsear n linhas x m colunas; dados ficam em RAM.
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# --- 3. SELEÇÃO DE FEATURES ---
# Select relevant features
# Mantém só 5 preditoras + alvo para reduzir dimensionalidade, ruído e overfitting.
# Pclass (classe socioeconômica 1/2/3), Sex, Age, Fare (tarifa), Embarked (porto C/Q/S).
# Decisão de engenharia: features com alta correlação histórica com sobrevivência
# ("women and children first", classe -> acesso aos botes).
df = df[['Pclass','Sex','Age','Fare','Embarked','Survived']]

# --- 4. TRATAMENTO DE VALORES AUSENTES ---
# Handle missing values
#df.method({col: value}, inplace=True)
# Matemática: imputação por mediana (Age) é robusta a outliers (ex: tarifas/bebês extremos),
# pois mediana minimiza erro L1: argmin_m sum|x_i - m|.
# Moda (Embarked) = valor de máxima frequência, i.e., estimador MAP sob prior uniforme categórica.
# Computação: inplace=True evita cópia O(n) extra do DataFrame.
df.fillna({'Age':df['Age'].median()}, inplace=True)
df.fillna({'Embarked':df['Embarked'].mode()[0]}, inplace=True)

# --- 5. SEPARAÇÃO X (features) / y (alvo) ---
# Define features and target
# X: matriz n x 5. y: vetor n (rótulos binários).
# Formalmente buscamos função f: X -> {0,1} que minimize risco esperado E[L(f(x), y)].
X = df.drop(columns=['Survived'])
y = df['Survived']

# --- 6. PRÉ-PROCESSAMENTO: SCALING + ENCODING ---
# Apply feature scaling and encoding
# ColumnTransformer = grafo de transformações paralelas que concatena saídas.
# a) StandardScaler em ['Age','Fare']: z = (x - mu) / sigma, mu=media, sigma=desvio-padrão.
#    Por quê? Regressão Logística usa gradiente descendente + regularização L2:
#    sem escala, Fare (~0-500) domina Age (~0-80) no produto w·x e na norma ||w||^2.
#    Após scaler: media 0, variância 1 -> otimização convexa converge rápido, contornos isotrópicos.
# b) OneHotEncoder em ['Pclass','Sex','Embarked']: transforma categoria em vetor binário.
#    Ex: Sex={male,female} -> [1,0] / [0,1]; evita impor ordem falsa (ex: C=0,Q=1,S=2 implicaria Q=(C+S)/2).
#    Matemática: embedding em base canônica de R^k, distância Hamming/euclidiana justa entre categorias.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Fare']),
        ('cat', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked'])
    ]
)

# fit_transform = 2 passos: fit aprende parâmetros (mu, sigma, vocabulário de categorias)
# e transform aplica. Resultado: matriz esparsa/densa pronta para álgebra linear.
# CUIDADO computacional: fit aqui usa dados inteiros -> pequeno vazamento de dados (leakage).
# Ideal: fit só no train dentro de Pipeline + cross_val_score(Pipeline...).
X_preprocessed = preprocessor.fit_transform(X)

# --- 7. BASELINE 1: REGRESSÃO LOGÍSTICA ---
# Train and evaluate Logistic Regression
# Modelo: P(y=1|x) = sigmoid(w·x + b) = 1/(1+exp(-(w·x+b))).
# Treino: minimiza log-loss (entropia cruzada) + L2: J(w) = -sum[y log p + (1-y)log(1-p)] + λ||w||².
# Otimização convexa (lbfgs por padrão) -> ótimo global único. Fronteira de decisão LINEAR.
# cross_val_score(cv=5): divide em 5 folds estratificados? (aqui KFold padrão), treina em 4/5 e testa em 1/5,
# roda 5x e retorna 5 acurácias. accuracy = (TP+TN)/(total). Média reduz variância da estimativa.
log_model = LogisticRegression()
log_scores = cross_val_score(log_model, X_preprocessed, y, cv=5, scoring='accuracy')
print(f"Logistic Regression Accuracy: {log_scores.mean():.2f}")

# --- 8. BASELINE 2: RANDOM FOREST ---
# Train and evaluate Random Forest
# Ensemble de T árvores CART via bagging: cada árvore treina em bootstrap (amostra com reposição)
# + sorteia subset de features em cada split. Predição = voto majoritário.
# Matemática do split: minimiza impureza Gini G = 1 - sum_k p_k² ou entropia H = -sum_k p_k log p_k.
# Ganho de informação = G_pai - média_ponderada(G_filhos). Reduz variância por E[ensemble] ≈ 1/T * var_individual
# se árvores descorrelacionadas. Captura relações NÃO-LINEARES e interações (ex: Sex=female & Pclass=1).
# random_state=42: fixa RNG (bootstrap + feature sampling) para reprodutibilidade.
rf_model = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf_model, X_preprocessed, y, cv=5, scoring='accuracy')
print(f"Random Forest Accuracy: {rf_scores.mean():.2f}")

# --- 9. GRADE DE HIPERPARÂMETROS ---
# Define hyperparameter grid
# Hiperparâmetro ≠ parâmetro: não aprendido por gradiente, controla capacidade/complexidade.
# n_estimators (T): nº árvores. Mais árvores = menor variância, custo O(T * n log n * m).
# max_depth: profundidade máxima. None = expande até puro (alta variância/overfit);
#   10/20 = regularização por truncamento (viés ↑, variância ↓).
# min_samples_split: mínimo de amostras para dividir um nó. 2 = agressivo, 5/10 = mais conservador/suave.
# Grid total: 3*3*3 = 27 combinações.
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# --- 10. GRID SEARCH COM VALIDAÇÃO CRUZADA ---
# Perform Grid Search
# GridSearchCV = busca exaustiva: para cada uma das 27 configs, roda CV de 5 folds (27*5=135 treinos)
# e escolhe a de maior accuracy média. É otimização de caixa-preta (sem gradiente).
# cv=5: estimativa honesta; scoring='accuracy': métrica alvo (ok aqui pois classes ~38% vs 62%, quase balanceado;
#   se desbalanceado, preferir f1/roc_auc).
# n_jobs=-1: paralelismo em todos os núcleos CPU (cada fit num worker) -> speedup ~n_cores.
# Complexidade: O(135 * custo_RF). É caro, mas embaraçosamente paralelo.
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_preprocessed, y)

# --- 11. RESULTADO FINAL ---
# Display best hyperparameters and score
# best_params_: argmax da accuracy média no grid. best_score_: essa accuracy média (não acurácia em holdout!).
# best_estimator_ (não impresso) já está re-treinado no dataset inteiro com os melhores params.
# Próximo passo ideal: avaliar em holdout separado ou nested-CV para estimativa sem viés otimista.
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_:.2f}")