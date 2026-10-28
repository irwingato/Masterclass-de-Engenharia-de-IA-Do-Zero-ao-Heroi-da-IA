# =============================================================================
# IMPORTS E BIBLIOTECAS (Engenharia de Software / Ciência da Computação)
# =============================================================================
import pandas as pd  # Estruturas de dados tabulares (DataFrame) e I/O de CSV.
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold  # Utilitários de validação cruzada e divisão de dados.
from sklearn.ensemble import RandomForestClassifier  # Floresta aleatória: ensemble de árvores de decisão.

# -----------------------------------------------------------------------------
# CARREGAMENTO DO DATASET (Engenharia de Dados / Estatística Descritiva)
# -----------------------------------------------------------------------------
# URL pública do dataset de fraude de cartão de crédito (TensorFlow / Kaggle).
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
# pd.read_csv faz o download e parseia o CSV em um DataFrame (tabela de m linhas x n colunas).
df = pd.read_csv(url)

# [MODIFICADO vs ORIGINAL] Usa apenas uma amostra de 10% dos dados para acelerar.
# O dataset original tem ~285 mil linhas; com amostragem fica cerca de 28 mil.
# Isso reduz drasticamente o tempo de treino do Random Forest (que escala mal com n amostras).
# Para uso produção, remova o .sample() e use o dataset completo.
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

# [CORREÇÃO] df.info é um método: deve ser chamado com parênteses df.info() para imprimir o conteúdo.
# No original, print(df.info) imprimia apenas a representação do objeto método (<bound method...>).
print("Dataset Info:\n")
df.info()
# Distribuição de classes da variável alvo (Class). Estatística descritiva de frequência absoluta.
print("\n Class Distribution:\n")
print(df['Class'].value_counts())

# -----------------------------------------------------------------------------
# DEFINIÇÃO DE FEATURES (X) E ALVO (y) (Aprendizado de Máquina Supervisionado)
# -----------------------------------------------------------------------------
# X = matriz de features (preditores): todas as colunas exceto o rótulo alvo 'Class'.
# y  = vetor target (resposta): a coluna binária 'Class' (0 = legítimo, 1 = fraude).
X = df.drop(columns=['Class'])
y = df['Class']

# -----------------------------------------------------------------------------
# DIVISÃO TREINO / TESTE (Teoria da Aprendizagem Estatística)
# -----------------------------------------------------------------------------
# train_test_split particiona os dados em conjuntos de treino (80%) e teste (20%).
# test_size=0.2  -> 20% dos dados reservados para avaliação final (hold-out).
# random_state=42 -> semente do gerador de números aleatórios (reprodutibilidade determinística).
# Motivo: avaliar o modelo em dados nunca vistos durante o treino (generalização),
# evitando overfitting (memorização do treino) e estimando erro de generalização.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------------------------
# K-FOLD CROSS-VALIDATION (Estimativa de Risco / Validação Cruzada)
# -----------------------------------------------------------------------------
# KFold divide os dados de treino em K=5 "folds" (partições) de tamanho igual.
# n_splits=5      -> número de partições.
# shuffle=True    -> embaralha antes de dividir (evita viés de ordenação dos dados).
# random_state=42 -> reprodutibilidade.
# Em cada iteração, um fold é usado como validação e os outros K-1 como treino;
# o modelo é treinado K vezes e a métrica é média dos K resultados.
# Matematicamente:  CV_score = (1/K) * sum_{i=1}^{K} score(modelo_treino_em_(D \ fold_i), fold_i)
# Isso reduz variância da estimativa de erro comparado a um único split treino/val.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# [MODIFICADO vs ORIGINAL] Reduz n_estimators de 100 (padrão do sklearn) para 50.
# Random Forest treina n árvores independentemente; reduzir de 100 para 50 corta cerca
# de metade o tempo de treino. custo: pequeno aumento de variância da previsão.
# random_state=42 garante que a aleatoriedade (bootstrapping, escolha de features) seja fixa.
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
# cross_val_score executa o K-Fold acima automaticamente:
#   - treina rf_model nos K-1 folds
#   - avalia no fold restante com scoring='accuracy'
#   - retorna array com os K scores de acurácia.
scores_kfold = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')

# scores_kfold.mean() -> média aritmética dos K acurácias: estimativa do erro de generalização.
print(f"K-fold cross validation accuracy: {scores_kfold.mean()}")
print(f"Mean Accuracy (K-Fold): {scores_kfold.mean():.2f}")

# -----------------------------------------------------------------------------
# STRATIFIED K-FOLD (Validação Cruzada Estratificada)
# -----------------------------------------------------------------------------
# StratifiedKFold é como KFold, MAS preserva a proporção de classes em cada fold.
# Crítico aqui porque o dataset é DESBALANCEADO (poucas fraudes, muitos legítimos).
# Sem estratificação, um fold poderia ficar sem exemplos da classe minoritária (fraude),
# tornando a acurácia enganosa (modelo que chuta "legítimo" sempre tem alta acurácia).
# n_splits=5, shuffle=True, random_state=42: mesma lógica do KFold simples.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Repete a validação cruzada, agora com folds estratificados.
scores_stratified = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')

print(f"Stratified K-fold cross validation scores: {scores_stratified}")
print(f"Mean Accuracy (Stratified K-Fold): {scores_stratified.mean():.2f}")

# -----------------------------------------------------------------------------
# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (Hold-out) — aproveita X_test/y_test
# -----------------------------------------------------------------------------
# [MODIFICADO vs ORIGINAL] O original criava X_test, y_test mas nunca os usava.
# Agora treina o modelo nos dados de treino inteiros e avalia no hold-out (dados nunca vistos).
rf_model.fit(X_train, y_train)
test_acc = rf_model.score(X_test, y_test)
print(f"\nAcurácia no conjunto de teste (hold-out): {test_acc:.2f}")

# -----------------------------------------------------------------------------
# NOTA MATEMÁTICA / CIÊNCIA DA COMPUTAÇÃO SOBRE ACURÁCIA EM DADOS DESBALANCEADOS
# -----------------------------------------------------------------------------
# Acurácia = (TP + TN) / (TP + TN + FP + FN). Em datasets muito desbalanceados
# (ex: 0,17% de fraude), um modelo que prevê sempre "legítimo" atinge ~99,8% de acurácia
# sem detectar nenhuma fraude. Portanto, acurácia sozinha é uma métrica enganosa aqui.
# Métricas mais adequadas para desbalanceamento: Precisão, Recall, F1-Score, AUC-ROC,
# e matriz de confusão. Random Forest lida com isso parcialmente via amostragem
# (class_weight, subsampling), mas a avaliação deve focar na classe minoritária.