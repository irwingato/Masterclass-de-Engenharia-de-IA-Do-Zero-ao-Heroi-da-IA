# ==============================================================================
# day6_ex.py - Hiperparâmetros, Busca em Grade (GridSearch) e Busca Aleatória
# Dataset: Iris (Fisher) | Modelos: Gradient Boosting Classifier + SVC (SVM)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORTAÇÕES DE BIBLIOTECAS
# ------------------------------------------------------------------------------
# scikit-learn (sklearn): biblioteca padrão de Machine Learning em Python.
# Fornece algoritmos, utilitários de pré-processamento, validação e métricas.

from sklearn.datasets import load_iris  # Carrega o dataset clássico "Iris" (flores de íris).
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# train_test_split: divide os dados em treino e teste (avaliação geralização).
# GridSearchCV: busca exaustiva em grade de hiperparâmetros com validação cruzada.
# RandomizedSearchCV: busca aleatória amostrando n_iter combinações (mais rápido que grid exaustiva).
from sklearn.ensemble import GradientBoostingClassifier
# GradientBoostingClassifier: ensemble de árvores de decisão sequenciais (boosting),
#  que corrige erros de árvores anteriores via gradiente descendente no espaço das previsões.
from sklearn.metrics import accuracy_score, classification_report
# accuracy_score: fração de previsões corretas = (TP + TN) / total.
# classification_report: precisão, recall, f1-score por classe e suporte.
from sklearn.svm import SVC
# SVC: Support Vector Classification (máquina de vetores de suporte para classificação).
# Busca o hiperplano que maximiza a margem entre classes no espaço de características.
import numpy as np
# NumPy: suporte a arrays/multidimensionais e operações matemáticas vetorizadas.

# ------------------------------------------------------------------------------
# 2. CARREGAMENTO DO DATASET IRIS
# ------------------------------------------------------------------------------
# O dataset Iris contém 150 amostras de 3 espécies de íris (Setosa, Versicolor, Virginica),
# cada uma com 4 características (sepal length, sepal width, petal length, petal width).
# data  -> matriz X de shape (150, 4): as 4 features (variáveis independentes/preditoras).
# target -> vetor y  de shape (150,): rótulo da classe (0, 1 ou 2) (variável dependente/alvo).

data = load_iris()
X, y = data.data, data.target

# ------------------------------------------------------------------------------
# 3. DIVISÃO TREINO / TESTE (Hold-out / Split)
# ------------------------------------------------------------------------------
# Objetivo: avaliar capacidade de generalização do modelo em dados não vistos.
# test_size=0.2  -> 20% dos dados reservados para teste (30 amostras), 80% para treino (120).
# random_state=42 -> semente do gerador de números aleatórios: garante REPRODUTIBILIDADE
#  (mesma divisão a cada execução). Essencial em ciência de dados para comparar resultados.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset Loaded and Split Successfully")

# ------------------------------------------------------------------------------
# 4. DEFINIÇÃO DA GRADE DE HIPERPARÂMETROS (Grid Search)
# ------------------------------------------------------------------------------
# Hiperparâmetros são configurações do modelo definidas ANTES do treino (ao contrário
# dos parâmetros internos, que são aprendidos durante o fit, ex: pesos das árvores).
# param_grid: dicionário onde cada chave é um hiperparâmetro e o valor é uma lista
# de candidatos a serem testados exaustivamente (produto cartesiano das opções).

param_grid = {
    'n_estimators': [50, 100, 150],
    # n_estimators: número de árvores de decisão (estágios de boosting) na sequência.
    # Mais árvores = modelo mais complexo, maior risco de overfitting e maior custo de treino.
    # Matemática: previsão final = soma ponderada das saídas de cada árvore fraca.

    'learning_rate': [0.01, 0.1, 0.2],
    # learning_rate (taxa de aprendizado / shrinkage): fator de contração que escala a
    # contribuição de cada nova árvore. Valores pequenos exigem mais árvores, mas
    # geralmente generalizam melhor (convergência mais estável no espaço funcional).
    # Atualização: F_m(x) = F_{m-1}(x) + eta * h_m(x), onde eta = learning_rate.

    'max_depth': [3, 5, 7]
    # max_depth: profundidade máxima de cada árvore de decisão.
    # Controla a complexidade local: profundidade alta captura mais padrões (overfitting),
    # profundidade baixa pode underfitting. A árvore particiona o espaço de features
    # recursivamente até atingir essa profundidade ou critério de parada.
}

# ------------------------------------------------------------------------------
# 5. INICIALIZAÇÃO DO GRIDSEARCHCV (Busca em Grade Exaustiva + CV)
# ------------------------------------------------------------------------------
# GridSearchCV testa TODAS as combinações possíveis da grade:
# 3 (n_estimators) × 3 (learning_rate) × 3 (max_depth) = 27 combinações.
# Para cada combinação, executa validação cruzada (cv=5).

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    # estimator: modelo base a ser otimizado (aqui, Gradient Boosting).
    # random_state=42 garante que a construção das árvores seja determinística.

    param_grid=param_grid,
    # grade de candidatos a serem explorados exaustivamente.

    scoring='accuracy',
    # métrica de avaliação para escolher a melhor combinação: acurácia
    # = (previsões corretas) / (total de previsões).

    cv=5,
    # cv = 5-fold cross-validation: divide o treino em 5 dobras (folds). Para cada
    # combinação de hiperparâmetros, treina em 4 folds e valida em 1, rotacionando
    # o fold de validação 5 vezes. A pontuação final é a MÉDIA das 5 validações.
    # Reduz viés da divisão aleatória e estimativa mais robusta do desempenho.

    n_jobs=-1
    # n_jobs=-1: utiliza TODOS os núcleos de CPU disponíveis para paralelizar
    # o treino das 27 combinações × 5 folds = 136 treinamentos. Acelera drasticamente.
)

# ------------------------------------------------------------------------------
# 6. EXECUÇÃO DA BUSCA (fit)
# ------------------------------------------------------------------------------
# .fit(X_train, y_train) executa o loop: para cada combinação da grade × cada fold,
# treina o GradientBoosting no treino do fold e avalia acurácia na validação do fold.
# Ao final, guarda a combinação com maior acurácia média de CV.

grid_search.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 7. EXTRAÇÃO DOS MELHORES HIPERPARÂMETROS E PONTUAÇÃO
# ------------------------------------------------------------------------------
best_params_grid = grid_search.best_params_
# Dicionário com a combinação de hiperparâmetros que obteve a maior acurácia média em CV.

best_score_grid = grid_search.best_score_
# Acurácia média de validação cruzada da melhor combinação (estimativa de desempenho).

print(f"Best Parameters (GridSearchCV): {best_params_grid}")
print(f"Best Cross-Validation Accuracy (GridSearchCV): {best_score_grid:.4f}")

# ------------------------------------------------------------------------------
# 8. RETREINO COM O MELHOR MODELO E AVALIAÇÃO NO TESTE
# ------------------------------------------------------------------------------
best_grid_model = grid_search.best_estimator_
# Instância do GradientBoosting já treinada com os melhores hiperparâmetros
# sobre TODO o conjunto de treino (X_train, y_train).

y_pred_grid = best_grid_model.predict(X_test)
# Gera previsões (rótulos de classe) para as 30 amostras de teste.

accuracy_grid = accuracy_score(y_test, y_pred_grid)
# Acurácia real em dados não vistos (teste): indica capacidade de generalização.
# Diferença entre best_score_grid (CV) e accuracy_grid (teste) revela overfitting.

print(f"Test Accuracy (GridSearchCV): {accuracy_grid:.4f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred_grid))
# classification_report mostra, por classe:
#   - precision = TP / (TP + FP)  (quando diz que é a classe, quão spesso acerta)
#   - recall    = TP / (TP + FN)  (quão spesso acha instâncias da classe)
#   - f1-score  = 2 * (precision*recall)/(precision+recall) (média harmônica)
#   - support   = número de amostras reais de cada classe no teste.

# ------------------------------------------------------------------------------
# 9. DEFINIÇÃO DA DISTRIBUIÇÃO DE HIPERPARÂMETROS (Randomized Search)
# ------------------------------------------------------------------------------
# Diferente do GridSearch (que testa TODAS combinações), o RandomizedSearchCV
# amostra aleatoriamente n_iter combinações da distribuição. É mais eficiente
# quando o espaço de busca é grande, pois explora menos combinações com custo menor.

param_dist = {
    'C': np.logspace(-3, 3, 10),
    # C: parâmetro de regularização do SVM. Inverso da força da regularização.
    # C pequeno = margem larga, mais tolerância a erros (menos overfitting).
    # C grande = margem estreita, tenta classificar tudo corretamente (risco overfitting).
    # np.logspace(-3, 3, 10): 10 valores espaçados logarithmicamente de 10^-3 a 10^3
    # (0.001, ..., 1000), adequado porque C age em escala multiplicativa/logarítmica.

    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    # kernel: função que mapeia features para um espaço de maior dimensão (trick do kernel).
    # 'linear': hiperplano linear no espaço original.
    # 'rbf' (Gaussian): exp(-gamma * ||x - x'||^2), mapeia para espaço infinito-dimensional,
    #   captura fronteiras não lineares complexas.
    # 'poly': (gamma * <x,x'> + coef0)^degree, fronteira polinomial.
    # 'sigmoid': tanh(gamma * <x,x'> + coef0), semelhante a rede neural com 1 camada.

    'gamma': ['scale', 'auto']
    # gamma: coeficiente para kernels rbf, poly, sigmoid. Define o "alcance" de uma
    # amostra de treino na fronteira de decisão.
    # 'scale' = 1 / (n_features * X.var())  (padrão sklearn moderno).
    # 'auto'  = 1 / n_features  (padrão antigo).
    # Gamma alto = fronteira curvada ao redor de poucos pontos (overfitting).
    # Gamma baixo = fronteira suave e geraliza melhor (underfitting se muito baixo).
}

# ------------------------------------------------------------------------------
# 10. INICIALIZAÇÃO DO RANDOMIZEDSEARCHCV
# ------------------------------------------------------------------------------
random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    # SVM com kernel selecionado na busca. random_state para reprodutibilidade.

    param_distributions=param_dist,
    # distribuição (listas) de candidatos. Não é grade exaustiva: são domínios de amostragem.

    n_iter=20,
    # n_iter: número de combinações aleatórias a serem amostradas e testadas (aqui 20).
    # Comparado com GridSearch (27 × 5 folds), testa bem menos modelos → mais rápido.
    # Compensa explorar menos combinações com boa chance de achar região próxima do ótimo.

    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42  # semente para a amostragem aleatória ser reproduzível.
)

# ------------------------------------------------------------------------------
# 11. EXECUÇÃO DA BUSCA ALEATÓRIA (fit)
# ------------------------------------------------------------------------------
# Amostra 20 combinações aleatórias de (C, kernel, gamma) e, para cada uma,
# executa 5-fold CV. Guarda a combinação com maior acurácia média.

random_search.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# 12. EXTRAÇÃO DOS MELHORES HIPERPARÂMETROS (Randomized)
# ------------------------------------------------------------------------------
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print(f"Best Parameters (RandomizedSearchCV): {best_params_random}")
print(f"Best Cross-Validation Accuracy (RandomizedSearchCV): {best_score_random:.4f}")

# ------------------------------------------------------------------------------
# 13. AVALIAÇÃO FINAL DO MELHOR MODELO SVC NO TESTE
# ------------------------------------------------------------------------------
best_random_model = random_search.best_estimator_

y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print(f"Test Accuracy (RandomizedSearchCV): {accuracy_random:.4f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred_random))

# ==============================================================================
# RESUMO CONCEITUAL (Ciência da Computação + Matemática)
# ==============================================================================
# 1. SUPERVISED LEARNING: aprendizado supervisionado (X com rótulo y conhecido).
# 2. TRAIN/TEST SPLIT: estima erro de generalização em dados não vistos.
# 3. CROSS-VALIDATION (k-fold): reduz variância da estimativa de desempenho;
#    erro médio de CV ≈ erro esperado do modelo em novos dados.
# 4. HIPERPARÂMETROS vs PARÂMETROS:
#    - Hiperparâmetros (n_estimators, C, gamma...): definidos antes, buscados via Grid/Random.
#    - Parâmetros (pesos das árvores, vetores suporte do SVM): aprendidos no .fit().
# 5. GRID SEARCH: exploração exaustiva (custo alto, garantia de achar o melhor na grade).
# 6. RANDOMIZED SEARCH: amostragem aleatória (custo baixo, bom para espaços grandes).
# 7. GRADIENT BOOSTING: ensemble sequencial; cada árvore corrige resíduos da anterior
#    por descida de gradiente no espaço funcional (F_m = F_{m-1} + eta * h_m).
# 8. SVM (SVC): maximiza margem entre classes; o kernel expande o espaço para tornar
#    classes linearmente separáveis (teoria de Cover / Representer Theorem).
# 9. ACURÁCIA = (TP + TN) / (P + N). Em datasets balanceados (Iris é), acurácia é
#    métrica válida; em desbalanceados, prefere-se F1-score / AUC-ROC.
# ==============================================================================
