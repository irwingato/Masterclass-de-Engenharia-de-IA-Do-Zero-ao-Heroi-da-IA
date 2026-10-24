# =============================================================================
# IMPORTS - Biblioteca e módulos utilizados
# =============================================================================

from sklearn.datasets import load_breast_cancer  # Carrega o dataset "Wisconsin Breast Cancer" (classificação binária: câncer maligno vs benigno) direto do scikit-learn.
from sklearn.model_selection import train_test_split  # Função que particiona dados em treino e teste de forma aleatória, controlada e reprodutível.
import pandas as pd  # Biblioteca para manipulação de dados tabulares (DataFrames) — importada aqui, embora não seja usada no restante do código.
from sklearn.ensemble import RandomForestClassifier  # Implementa Random Forest: um ensemble (conjunto) de árvores de decisão treinadas via bagging.
from sklearn.metrics import accuracy_score, classification_report  # Métricas de avaliação: acurácia global e relatório com precisão, recall e F1-score por classe.

# =============================================================================
# CARREGAMENTO DO DATASET
# =============================================================================

# Load dataset
data = load_breast_cancer()  # Retorna um objeto Bunch (dicionário-like) com .data (matriz de features), .target (vetor de rótulos), .feature_names, .target_names etc.

X, y = data.data, data.target  # X = matriz de características (n_samples × n_features), y = vetor alvo (n_samples,) com classes 0 (maligno) e 1 (benigno).

# =============================================================================
# DIVISÃO TREINO / TESTE
# =============================================================================

# Split dataset
# train_test_split embaralha os índices e divide em 4 blocos:
#   X_train: features para treino do modelo
#   X_test : features para avaliação (nunca vistas durante o treino)
#   y_train: rótulos verdadeiros do treino
#   y_test : rótulos verdadeiros do teste (ground truth para avaliação)
# test_size=0.2 reserva 20% dos dados para teste e 80% para treino.
# random_state=42 fixa a semente do gerador de números aleatórios (RNG) para que a divisão seja idêntica em toda execução (reprodutibilidade).
# Ciência da computação: esta separação evita "vazamento de dados" (data leakage) e superestimação do desempenho (overfitting de avaliação).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# INFORMAÇÕES DO DATASET
# =============================================================================

# Display dataset info
print(f"Feature Names: {data.feature_names}")  # Nomes das 30 características (medidas de células: raio, textura, perímetro, área, etc.)
print(f"Class Names: {data.target_names}")      # Nomes das classes: ['malignant' 'benign']

# =============================================================================
# TREINAMENTO — RANDOM FOREST COM HIPERPARÂMETROS PADRÃO
# =============================================================================

# Train Random Forest with default hyperparameters
# RandomForestClassifier é um ensemble de árvores de decisão (CART) treinadas pelo método de "bagging" (Bootstrap AGGregatING):
#   1. Para cada árvore, amostra com reposição um subconjunto de treino (bootstrap sample).
#   2. Em cada nó, escolhe a melhor divisão entre um subconjunto aleatório de features (mtry).
#   3. A predição final é a votação majoritária (classificação) ou média (regressão) das árvores.
# Matemática: dado um input x, cada árvore t produz uma predição h_t(x). O ensemble retorna:
#     H(x) = mode{ h_1(x), h_2(x), ..., h_T(x) }   (votação por maioria)
# Isso reduz a variância do modelo sem aumentar significativamente o viés (trade-off viés-variância).
rf_default = RandomForestClassifier(random_state=42)  # Instancia o modelo com hiperparâmetros padrão (n_estimators=100, max_depth=None, etc.).
rf_default.fit(X_train, y_train)  # Ajuste (treinamento): constrói as árvores aprendendo os pontos de corte ótimos em cada nó (critério: impureza de Gini ou entropia).

# =============================================================================
# PREDITO E AVALIAÇÃO — MODELO PADRÃO
# =============================================================================

# Predict and evaluate
y_pred_default = rf_default.predict(X_test)  # Gera predições (rótulos 0 ou 1) para as amostras de teste usando o modelo treinado.

# accuracy_score calcula a acurácia: fração de predições corretas.
# Matemática: Acurácia = (TP + TN) / (TP + TN + FP + FN)
#   TP = True Positives (benigno previsto como benigno)
#   TN = True Negatives (maligno previsto como maligno)
#   FP = False Positives (maligno previsto como benigno — erro tipo I)
#   FN = False Negatives (benigno previsto como maligno — erro tipo II)
# Ciência da computação: acurácia sozinha pode ser enganosa em datasets desbalanceados; por isso usamos também o classification_report.
accuracy_default = accuracy_score(y_test, y_pred_default)

print(f"Default Model Accuracy: {accuracy_default:.4f}")  # Imprime acurácia com 4 casas decimais.
print("\nClassification Report:\n", classification_report(y_test, y_pred_default))
# classification_report imprime, por classe:
#   - precision = TP / (TP + FP)  (quando o modelo diz "benigno", quão spesso está certo?)
#   - recall    = TP / (TP + FN)  (de todos os benignos reais, quantos o modelo encontrou?)
#   - f1-score  = 2 · (precision · recall) / (precision + recall)  (média harmônica de precisão e recall)
#   - support   = número de amostras reais de cada classe no teste

# =============================================================================
# TREINAMENTO — RANDOM FOREST COM HIPERPARÂMETROS AJUSTADOS
# =============================================================================

# Train Random Forest with custom hyperparameters
rf_tuned = RandomForestClassifier(
    n_estimators=400,   # Número de árvores na floresta. Mais árvores → menor variância, mas maior custo de computação e memória. (Padrão: 100)
    max_depth=5,        # Profundidade máxima de cada árvore. Limita o crescimento para evitar overfitting (árvore muito específica ao treino). (Padrão: None = cresce até folhas puras)
    random_state=42     # Semente para reprodutibilidade da amostragem bootstrap e divisão de features.
)
rf_tuned.fit(X_train, y_train)  # Treina a floresta com os parâmetros acima.

# =============================================================================
# PREDITO E AVALIAÇÃO — MODELO AJUSTADO
# =============================================================================

# Predict and evaluate
y_pred_tuned = rf_tuned.predict(X_test)  # Predições do modelo ajustado no conjunto de teste.
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)  # Acurácia do modelo ajustado.

print(f"Tuned Model Accuracy: {accuracy_tuned:.4f}")  # Imprime acurácia do modelo ajustado.
print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned))  # Relatório detalhado do modelo ajustado.

# =============================================================================
# RESUMO CONCEITUAL (Matemática e Ciência da Computação)
# =============================================================================
# - Supervised Learning (aprendizado supervisionado): o modelo aprende mapeamento X → y usando rótulos conhecidos.
# - Classification (classificação binária): saída é uma classe discreta (0 ou 1), não um valor contínuo.
# - Train/Test split: estima o erro de generalização (desempenho em dados não vistos).
# - Ensemble/Bagging: combina múltiplos modelos fracos (árvores) para criar um modelo forte e robusto.
# - Hiperparâmetros vs parâmetros: hiperparâmetros (n_estimators, max_depth) são definidos antes do treino; parâmetros (pontos de corte das árvores) são aprendidos durante o fit().
# - Overfitting: modelo decorre o treino e falha em dados novos. max_depth e n_estimators ajudam a controlá-lo.
# - Métricas: acurácia global vs precisão/recall/F1 por classe para avaliação mais completa.