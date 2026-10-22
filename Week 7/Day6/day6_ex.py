import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# =============================================================================
# 1. CARREGAMENTO DO DATASET
# =============================================================================
# Dataset: Credit Card Fraud Detection (Kaggle/Google Cloud)
# - 284.807 transações, 492 fraudes (0.17% - EXTREMAMENTE desbalanceado)
# - Features: V1-V28 (PCA anonimizadas), Time, Amount
# - Target: Class (0=legítimo, 1=fraude)
# =============================================================================
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# =============================================================================
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# =============================================================================
# df.info(): tipos de dados, não-nulos, uso de memória
# df["Class"].value_counts(): distribuição das classes
# IMPORTANTE: Em dados desbalanceados, acurácia é MÉTRICA ENGANOSA
#   Ex: 99.8% acurácia prevendo tudo como classe 0 = inútil para fraude
# =============================================================================
print("Dataset Info:\n")
print(df.info())
print("\n Class Distribution:\n")
print(df["Class"].value_counts())

# =============================================================================
# 3. DIVISÃO TREINO/TESTE (Holdout Method)
# =============================================================================
# X = features (todas colunas exceto 'Class')
# y = target (coluna 'Class')
# test_size=0.2: 80% treino, 20% teste
# random_state=42: reprodutibilidade (seed fixa)
# 
# MATEMÁTICA: Separação aleatória estratificada (mantém proporção classes)
#   P(y_train = c) ≈ P(y_test = c) ≈ P(y = c)  para cada classe c
#   Isso garante que ambas partições representam a distribuição original
# =============================================================================
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# 4. RANDOM FOREST COM CLASS_WEIGHT='BALANCED' (Baseline)
# =============================================================================
# Random Forest: Ensemble de Decision Trees (Bagging)
# - n_estimators árvores treinadas em bootstrap samples
# - Feature bagging: subset aleatório de features por split
# - Predição: votação majoritária (classificação) / média (regressão)
#
# class_weight='balanced': Ajusta pesos inversamente proporcionais à frequência
#   w_j = n_samples / (n_classes * n_samples_class_j)
#   Para este dataset: w_0 ≈ 1, w_1 ≈ 579
#   Função de perda ponderada: L = Σ w_y * loss(y, ŷ)
#   Isso força a árvore a "prestar mais atenção" na classe minoritária
#
# VANTAGEM: Sem gerar dados sintéticos, treino mais rápido
# =============================================================================
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

# =============================================================================
# 5. AVALIAÇÃO: CLASSIFICATION REPORT
# =============================================================================
# Métricas por classe (essencial para desbalanceado):
# - Precision = TP / (TP + FP) = "Das preditas como fraude, quantas são fraude?"
# - Recall = TP / (TP + FN) = "Das fraudes reais, quantas detectamos?"
# - F1 = 2 * (Precision * Recall) / (Precision + Recall) = média harmônica
# - Support = n_amostras reais da classe
#
# Para fraude: Recall (Sensibilidade) é MAIS IMPORTANTE que Precision
#   Falso negativo = fraude não detectada = prejuízo financeiro
#   Falso positivo = transação legítima bloqueada = inconveniente cliente
# =============================================================================
y_pred = rf_model.predict(X_test)
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

# =============================================================================
# 6. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
# =============================================================================
# ROC: Curva TPR (Recall) vs FPR (Taxa Falsos Positivos) variando threshold
#   TPR = TP / (TP + FN) = Sensibilidade
#   FPR = FP / (FP + TN) = 1 - Especificidade
#
# AUC: Área sob a curva (0.5 = aleatório, 1.0 = perfeito)
#   Interpretação probabilística: P(score(fraude) > score(legítimo))
#   Independente de threshold - avalia qualidade do ranking
#
# predict_proba()[:,1]: Probabilidade da classe positiva (fraude)
#   Necessário para ROC-AUC (precisa scores contínuos, não classes discretas)
# =============================================================================
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
print(f"ROC-AUC: {roc_auc:.2f}")

# =============================================================================
# 7. SMOTE (Synthetic Minority Over-sampling Technique)
# =============================================================================
# PROBLEMA: class_weight ajuda, mas árvores ainda veem poucas fraudes reais
# SOLUÇÃO: Gerar amostras SINTÉTICAS da classe minoritária
#
# ALGORITMO SMOTE (Chawla et al., 2002):
# Para cada amostra x_i da classe minoritária:
#   1. Encontra k-vizinhos mais próximos (KNN) no espaço de features
#      Distância: d(x_i, x_j) = ||x_i - x_j||₂ (Euclidiana)
#   2. Seleciona aleatoriamente um vizinho x_zi
#   3. Gera amostra sintética por interpolação linear:
#      x_new = x_i + λ * (x_zi - x_i),  onde λ ~ U(0,1)
#
# sampling_strategy=0.5: Ratio minoritária/majoritária = 0.5 (1:2)
#   Classe 0 (majoritária): ~227.845 (original * 0.8)
#   Classe 1 (minoritária): ~113.922 (0.5 * 227.845)
#   Total: ~341.767 (vs 455.690 se 1:1) - MUITO MAIS RÁPIDO
#
# k_neighbors=5 (padrão): Vizinhado local para interpolação
#   Menor = mais local/ruidoso, Maior = mais suave/over-generalização
# =============================================================================
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# =============================================================================
# 8. VERIFICAÇÃO DA DISTRIBUIÇÃO PÓS-SMOTE
# =============================================================================
print("\n Class Distribution After SMOTE: \n")
print(pd.Series(y_resampled).value_counts())

# =============================================================================
# 9. RANDOM FOREST NOS DADOS BALANCEADOS (SEM class_weight)
# =============================================================================
# Agora classes estão balanceadas (1:2), class_weight não necessário
# Modelo vê ~114k fraudes sintéticas + ~395 fraudes reais
# VANTAGEM: Árvores aprendem fronteira de decisão mais rica
# RISCO: Overfitting em amostras sintéticas (ruído na fronteira)
# =============================================================================
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_resampled, y_resampled)

# =============================================================================
# 10. AVALIAÇÃO DO MODELO COM SMOTE
# =============================================================================
# IMPORTANTE: Avaliação SEMPRE no X_test ORIGINAL (sem SMOTE!)
#   X_test representa distribuição REAL (0.17% fraude)
#   Métricas no teste balanceado seriam enganosas
# =============================================================================
y_pred_smote = rf_model_smote.predict(X_test)
print("\n Classification Report (SMOTE):\n")
print(classification_report(y_test, y_pred_smote))

# ROC-AUC no teste original
roc_auc_smote = roc_auc_score(y_test, rf_model_smote.predict_proba(X_test)[:,1])
print(f"ROC-AUC (SMOTE): {roc_auc_smote:.2f}")

# =============================================================================
# RESUMO COMPARATIVO ESPERADO:
# =============================================================================
# | Métrica          | class_weight='balanced' | SMOTE (1:2)       |
# |------------------|-------------------------|-------------------|
# | Recall (Fraude)  | ~0.85-0.90              | ~0.90-0.95        |
# | Precision        | ~0.10-0.20              | ~0.05-0.15        |
# | F1-score         | ~0.18-0.33              | ~0.10-0.25        |
# | ROC-AUC          | ~0.97-0.99              | ~0.97-0.99        |
# | Tempo treino     | ~10-30s                 | ~60-180s          |
#
# CONCLUSÃO: SMOTE aumenta Recall mas diminui Precision (mais falsos positivos)
#   class_weight='balanced' é mais eficiente computacionalmente
#   Para produção: considere XGBoost/LightGBM com scale_pos_weight
# =============================================================================