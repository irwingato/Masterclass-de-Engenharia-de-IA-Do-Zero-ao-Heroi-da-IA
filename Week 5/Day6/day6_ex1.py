"""
day6_ex1.py — Comparação entre Regressão Logística e k-NN no dataset Iris.

OBJETIVO COMPUTACIONAL:
    Comparar dois paradigmas de classificação supervisionada:
    1) Modelo paramétrico + probabilístico (Regressão Logística)
    2) Modelo não-paramétrico + baseado em instâncias (k-NN)

DATASET IRIS:
    - 150 amostras, 4 features numéricas contínuas:
      [comprimento_sépala, largura_sépala, comprimento_pétala, largura_pétala] em cm
    - 3 classes balanceadas (50 cada): 0=setosa, 1=versicolor, 2=virginica
    - Problema pequeno, bem comportado e quase linearmente separável → ideal para didática.
"""

# ----------------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------------
from sklearn.datasets import load_iris  # Função que carrega o dataset Iris embutido no sklearn
from sklearn.model_selection import train_test_split  # Divide dados em treino/teste (holdout)
from sklearn.preprocessing import StandardScaler  # Padronização z-score: z = (x - média) / desvio_padrão
from sklearn.neighbors import KNeighborsClassifier  # Classificador k-vizinhos mais próximos
from sklearn.metrics import accuracy_score, classification_report  # Métricas de avaliação
from sklearn.linear_model import LogisticRegression  # Regressão Logística (classificador linear probabilístico)

# ----------------------------------------------------------------------------
# 2. CARREGAMENTO DOS DADOS
# ----------------------------------------------------------------------------
# load_iris() retorna um objeto Bunch (parecido com dict) com:
#   .data   -> matriz X de shape (150, 4)
#   .target -> vetor y de shape (150,) com rótulos {0, 1, 2}
#   .feature_names, .target_names -> metadados descritivos
data = load_iris()
X, y = data.data, data.target  # X = variáveis independentes (features), y = variável dependente (classe)

# ----------------------------------------------------------------------------
# 3. DIVISÃO TREINO / TESTE (HOLDOUT)
# ----------------------------------------------------------------------------
# train_test_split embaralha e divide:
#   test_size=0.2  -> 20% teste (30 amostras), 80% treino (120 amostras)
#   random_state=42 -> semente do gerador aleatório → reprodutibilidade (mesma divisão toda vez)
# MATEMÁTICA/COMPUTAÇÃO:
#   - Estimativa honesta do erro de generalização: o modelo NUNCA vê X_test durante o fit.
#   - Custo: menos dados para treino (trade-off viés/variância da estimativa).
# OBSERVAÇÃO: aqui não foi usado stratify=y. O ideal seria stratify=y para preservar
#   a proporção das 3 classes nos dois conjuntos (evita, por azar, um conjunto desbalanceado).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------
# 4. PADRONIZAÇÃO DAS FEATURES (STANDARDIZATION)
# ----------------------------------------------------------------------------
# StandardScaler aplica, por coluna j:  z_ij = (x_ij - μ_j) / σ_j
#   onde μ_j e σ_j são média e desvio-padrão da coluna j calculados NO TREINO.
# POR QUE FAZER ISSO?
#   - Regressão Logística: otimizada por gradiente (solver LBFGS). Features em escalas
#     diferentes deformam a superfície de custo e lentificam/impedem a convergência.
#   - k-NN: usa distância Euclidiana d(a,b) = sqrt(Σ(a_j - b_j)²). Sem padronizar,
#     a feature com maior magnitude (ex: comprimento em cm) domina a distância.
# REGRA DE OURO ANTI-VAZAMENTO (DATA LEAKAGE):
#   - fit_transform(X_train): aprende μ,σ no treino E transforma o treino.
#   - transform(X_test):      REUSA o μ,σ do treino para transformar o teste.
#     (Nunca dar fit no teste, senão informação do teste "vaza" para o treino.)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Aprende μ,σ do treino e padroniza o treino (média 0, variância 1)
X_test = scaler.transform(X_test)        # Aplica a MESMA transformação no teste

# ----------------------------------------------------------------------------
# 5. REGRESSÃO LOGÍSTICA (MODELO PARAMÉTRICO)
# ----------------------------------------------------------------------------
# MATEMÁTICA (caso multinomial com softmax, que o sklearn usa por padrão):
#   Para cada classe c:  score_c = w_c·x + b_c   (função linear)
#   Probabilidade:      P(y=c|x) = exp(score_c) / Σ_k exp(score_k)   (softmax)
#   Treino = minimizar a entropia cruzada (log-loss):
#     J(W,b) = - (1/m) Σ_i log P(y=y_i | x_i) + (λ/2)||W||²
#   O termo (λ/2)||W||² é regularização L2 (C=1.0 por padrão, onde C = 1/λ).
# COMPUTAÇÃO:
#   - Otimizador padrão 'lbfgs' (quasi-Newton) aproxima a Hessiana para convergir rápido.
#   - max_iter=200 = nº máximo de iterações do otimizador. Se for baixo demais,
#     o modelo não converge (aviso ConvergenceWarning); 200 é margem segura aqui.
#   - Complexidade de treino ≈ O(m·n·iterações); predição O(n·n_classes) — muito rápida.
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)  # Encontra W,b que minimizam J (aprende 3 vetores w, um por classe)

# ----------------------------------------------------------------------------
# 6. PREDIÇÃO + AVALIAÇÃO DA REGRESSÃO LOGÍSTICA
# ----------------------------------------------------------------------------
# predict(x) = argmax_c P(y=c|x), ou seja, escolhe a classe de maior probabilidade.
y_pred_lr = log_reg.predict(X_test)

# accuracy_score = (nº acertos) / (nº total) = (1/m) Σ 1(y_pred == y_true)
# É intuitiva, mas pode enganar se as classes forem desbalanceadas (aqui são balanceadas, ok).
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy: ", accuracy_lr)

# ----------------------------------------------------------------------------
# 7. BLOCO DE RE-PADRONIZAÇÃO — ATENÇÃO: REDUNDANTE / ERRO DIDÁTICO
# ----------------------------------------------------------------------------
# O código original repetia:
#   scaler = StandardScaler()
#   X_train = scaler.fit_transform(X_train)  # <- ERRO: X_train JÁ está padronizado!
#   X_test = scaler.transform(X_test)
# POR QUE É ERRADO?
#   1) Desperdício computacional (refaz O(m·n) à toa).
#   2) Conceitualmente errado: dar fit() de novo em X_train padronizado calcula
#      μ≈0 e σ≈1 novamente — o resultado numérico muda pouco, mas quebra o
#      princípio de "aprender a transformação UMA vez no treino original".
#   3) Se fosse fit_transform(X_test), seria vazamento de dados (data leakage).
# CORREÇÃO: bloco removido/comentado. Mantemos X_train/X_test da etapa 4.
# (Se este arquivo for re-executado do zero, a etapa 4 já deixa tudo padronizado.)

# ----------------------------------------------------------------------------
# 8. k-NN — k-NEAREST NEIGHBORS (MODELO BASEADO EM INSTÂNCIAS)
# ----------------------------------------------------------------------------
# MATEMÁTICA:
#   Distância Euclidiana entre teste x e cada treino x_i:
#     d(x, x_i) = sqrt( Σ_j (x_j - x_ij)² )
#   O algoritmo encontra os k vizinhos com menor d e faz VOTAÇÃO MAJORITÁRIA:
#     y_hat = moda(y dos k vizinhos)
# COMPUTAÇÃO:
#   - Treino é "preguiçoso" (lazy): só memoriza os dados → fit é O(1)/O(m·n) memória.
#   - Predição é cara: O(m_treino · n_features) por amostra de teste (força bruta;
#     sklearn pode usar KD-Tree/Ball-Tree para acelerar em baixa dimensão).
#   - Hiperparâmetro k controla viés-variância:
#       k pequeno (ex: 1) → fronteira complexa, baixo viés, alta variância (overfitting,
#                            sensível a ruído/outliers).
#       k grande           → fronteira suave, alto viés, baixa variância (underfitting).
#   - k=5 ímpar evita empate na votação binária; em 3 classes empate ainda é possível
#     (sklearn desempatra pela ordem dos vizinhos).
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)  # Instancia o k-NN com k=5 e métrica euclidiana (p=2) padrão
knn.fit(X_train, y_train)  # No k-NN, "treinar" = apenas armazenar X_train/y_train em estrutura de busca
y_pred_knn = knn.predict(X_test)  # Para cada x em X_test: calcula distâncias, acha 5 vizinhos, vota
accuracy_knn = accuracy_score(y_test, y_pred_knn)  # Mesma fórmula de acurácia da seção 6
print(f"k-NN Accuracy k ={best_k}: ", accuracy_knn)

# ----------------------------------------------------------------------------
# 9. COMPARAÇÃO DETALHADA (CLASSIFICATION REPORT)
# ----------------------------------------------------------------------------
# classification_report calcula, POR CLASSE e médias globais:
#   Precision = TP / (TP + FP)  → "quando previu a classe, com que frequência acertou?"
#   Recall    = TP / (TP + FN)  → "de todos os exemplos reais da classe, quantos achou?"
#   F1-score  = 2·P·R / (P + R) → média harmônica (pune modelo desbalanceado entre P e R)
#   Support   = nº de amostras reais da classe no teste
#   macro avg = média simples entre classes | weighted avg = média ponderada pelo support
# CORREÇÃO DE BUG DO CÓDIGO ORIGINAL:
#   - O 1º report dizia "Logistic Regression" mas passava y_pred_knn (errado!).
#     Aqui corrigido para y_pred_lr.
#   - O 2º título dizia "k-NN Regression" → o correto é "k-NN Classification"
#     (regressão prevê valor contínuo; aqui é classificação).
print("\n Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))  # CORRIGIDO: era y_pred_knn

print("\n k-NN Classification Report:")  # CORRIGIDO: era "k-NN Regression"
print(classification_report(y_test, y_pred_knn))

# ----------------------------------------------------------------------------
# 10. EXPERIMENTO COM DIFERENTES VALORES DE k (CÓDIGO COMENTADO)
# ----------------------------------------------------------------------------
# Ideia: varrer k = 1..10 para ver a curva viés-variância na prática.
#   - Para cada k: instancia novo KNeighborsClassifier, treina, prevê e mede acurácia.
#   - Complexidade total ≈ 10 × custo de predição do k-NN.
#   - Como escolher o melhor k de verdade? Com VALIDAÇÃO CRUZADA (ex: StratifiedKFold),
#     não só no teste único (que pode dar sorte/azar). Ver GridSearchCV.
#   - Descomente para executar:
# # Experiment with different values of k
# for k in range(1, 11):
#     # Initialize k-NN model
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#
#     # Predict on test data
#     y_pred = knn.predict(X_test)
#
#     # Evaluate performance
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"k = {k}, Accuracy = {accuracy:.2f}")
