"""
DAY 1 - EXERCÍCIO: ENSEMBLE COM VOTING CLASSIFIER NO DATASET IRIS
===============================================================================
OBJETIVO COMPUTACIONAL:
    Comparar 3 classificadores individuais (Regressão Logística, Árvore de
    Decisão e k-NN) com um Ensemble por voto majoritário (VotingClassifier).

OBJETIVO MATEMÁTICO:
    Mostrar que combinar hipóteses diferentes reduz a variância do erro
    (teoria de ensembles: se os erros dos modelos forem parcialmente
    não-correlacionados, o voto da maioria erra menos que a média dos erros).

PIPELINE:
    1. Carregar dados -> 2. Dividir treino/teste -> 3. Padronizar ->
    4. Treinar 3 modelos -> 5. Combinar via voto -> 6. Medir acurácia.
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
# COMPUTAÇÃO: importamos apenas classes/funções do scikit-learn (sklearn),
# que implementa os algoritmos em Python + Cython/C para serem rápidos.
from sklearn.datasets import load_iris            # Carrega o dataset Iris embutido
from sklearn.model_selection import train_test_split  # Divide dados em treino/teste
from sklearn.preprocessing import StandardScaler  # Padronização z-score
from sklearn.linear_model import LogisticRegression  # Modelo linear probabilístico
from sklearn.tree import DecisionTreeClassifier   # Modelo hierárquico de regras
from sklearn.neighbors import KNeighborsClassifier  # Modelo baseado em distâncias
import numpy as np  # (importado aqui, útil para manipulação numérica de arrays)
from sklearn.ensemble import VotingClassifier  # Meta-modelo que combina classificadores
from sklearn.metrics import accuracy_score  # Métrica: fração de acertos

# ---------------------------------------------------------------------------
# 2. CARREGAR O DATASET
# ---------------------------------------------------------------------------
# MATEMÁTICA / DADOS:
#   Iris = 150 amostras, 4 features (X) contínuas:
#     [comprimento sépala, largura sépala, comprimento pétala, largura pétala] em cm.
#   Alvo (y) = 3 classes balanceadas: 0=setosa, 1=versicolor, 2=virginica (50 cada).
# COMPUTAÇÃO:
#   load_iris() retorna um objeto Bunch (dict-like) com .data (matriz 150x4)
#   e .target (vetor 150,). X, y são numpy arrays, estrutura padrão do sklearn.
data = load_iris()
X, y = data.data, data.target

# ---------------------------------------------------------------------------
# 3. DIVIDIR EM TREINO E TESTE
# ---------------------------------------------------------------------------
# MATEMÁTICA:
#   Holdout 80/20: treinamos em 80% e avaliamos nos 20% NUNCA vistos.
#   Isso estima o ERRO DE GENERALIZAÇÃO: E[(y - f(x))²] em dados novos,
#   evitando overfitting (decorar o treino).
# COMPUTAÇÃO:
#   train_test_split embaralha os índices (shuffle=True por padrão) e corta:
#     150 * 0.2 = 30 amostras p/ teste, 120 p/ treino.
#   random_state=42 fixa a semente do gerador pseudo-aleatório -> resultado
#   reproduzível. Sem isso, cada execução daria uma divisão diferente.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# 4. PADRONIZAÇÃO DAS FEATURES (StandardScaler)
# ---------------------------------------------------------------------------
# MATEMÁTICA (z-score, por coluna j):
#     z = (x - μ) / σ
#   onde μ = média da coluna no TREINO, σ = desvio-padrão no TREINO.
#   Resultado: cada feature passa a ter média 0 e variância 1.
#   POR QUÊ? Regressão Logística (otimização por gradiente) e k-NN (distância
#   Euclidiana) são sensíveis à ESCALA. Sem padronizar, a feature com maior
#   magnitude (ex: comprimento em cm) dominaria a distância Euclidiana:
#     d(x, x') = sqrt( Σ_j (x_j - x'_j)² )
#   e o gradiente descendente convergiria mais devagar / de forma instável.
# COMPUTAÇÃO:
#   fit_transform(X_train): 1) calcula μ, σ do treino (fit) e 2) aplica a
#     fórmula (transform). Complexidade O(n*d).
#   transform(X_test): REUSA o μ, σ do treino. NUNCA faça fit no teste,
#     pois isso seria "data leakage" (vazamento): o modelo veria estatísticas
#     dos dados de avaliação antes da hora.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 5. CRIAR OS 3 CLASSIFICADORES INDIVIDUAIS
# ---------------------------------------------------------------------------
# IDEIA DO ENSEMBLE: combinar modelos com VIÉSES INDUTIVOS diferentes, para que
# errem em pontos diferentes. O voto da maioria então corrige erros isolados.

# --- 5a. Regressão Logística ---
# MATEMÁTICA (multiclasse, multinomial por padrão):
#   Para cada classe k, calcula um score linear: z_k = w_kᵀ·x + b_k
#   Converte em probabilidade via SOFTMAX: P(y=k|x) = exp(z_k) / Σ_j exp(z_j)
#   Treino = minimizar a ENTROPIA CRUZADA + regularização L2:
#     J(w) = -Σ log P(y_i|x_i) + λ||w||²
#   Otimizado por gradiente (solver 'lbfgs'). É um modelo LINEAR: a fronteira
#   de decisão é um hiperplano.
# COMPUTAÇÃO: rápido, O(n*d*iterações), bom baseline calibrado.
log_model = LogisticRegression()

# --- 5b. Árvore de Decisão ---
# MATEMÁTICA:
#   Divide o espaço recursivamente com regras "se x_j <= threshold".
#   Cada divisão busca maximizar o GANHO DE INFORMAÇÃO, i.e., reduzir a
#   impureza do nó. Impureza padrão = GINI: G = 1 - Σ_k p_k²,
#   onde p_k = fração da classe k no nó. G=0 => nó puro.
#   É um modelo NÃO-LINEAR, de ALTA VARIÂNCIA: muda muito com pequenas
#   mudanças nos dados (propenso a overfitting se não podado).
# COMPUTAÇÃO: treino guloso O(n*d*log n), inferência O(profundidade).
dt_model = DecisionTreeClassifier()

# --- 5c. k-NN (k-Nearest Neighbors) ---
# MATEMÁTICA:
#   Não há treino paramétrico ("lazy learning"): apenas memoriza X_train.
#   Para prever um ponto x, calcula a DISTÂNCIA EUCLIDIANA a todos os treinos:
#     d(x, x_i) = sqrt( Σ_j (x_j - x_ij)² )
#   Pega os k=5 vizinhos mais próximos (padrão) e vota (moda das classes).
#   É NÃO-LINEAR e NÃO-PARAMÉTRICO: a fronteira se adapta localmente aos dados.
# COMPUTAÇÃO: treino O(1) (só armazena), predição O(n*d) por amostra
#   (cara se n for grande). Por isso a padronização acima foi essencial.
knn_model = KNeighborsClassifier()

# --- 5d. TREINO (fit) ---
# COMPUTAÇÃO: cada .fit(X_train, y_train) ajusta os parâmetros internos:
#   log_model: encontra w, b ótimos; dt_model: constrói os nós/folhas;
#   knn_model: apenas indexa os pontos de treino na memória.
log_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 6. ENSEMBLE COM VOTING CLASSIFIER (VOTO DURO)
# ---------------------------------------------------------------------------
# MATEMÁTICA (hard voting / voto majoritário):
#   Para cada amostra x, cada modelo m dá um voto de classe c_m ∈ {0,1,2}.
#   Predição final = MODA: ŷ = argmax_k Σ_m 1(c_m == k)
#   Ex: se log=0, árvore=1, knn=1 -> votos {0:1, 1:2} -> prediz 1.
#   Se os erros forem independentes e cada modelo acertar com p > 0.5,
#   a probabilidade de a maioria errar cai (análogo ao teorema de Condorcet).
# COMPUTAÇÃO:
#   VotingClassifier é um META-ESTIMADOR: guarda a lista de (nome, estimador)
#   e delega fit/predict para cada um internamente. voting='hard' = conta votos
#   de classes; voting='soft' (alternativa) = média das PROBABILIDADES
#   P(y|x) e aí escolheria a maior média (exige predict_proba em todos).
#
# NOTA: o código original definia esse ensemble DUAS vezes seguidas e dava
# dois .fit(). Isso é redundante (o segundo sobrescreve o primeiro).
# Mantemos UMA única definição + UM único fit abaixo.
ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', log_model),          # apelido 'log_reg' -> Regressão Logística
        ('decision_tree', dt_model),     # apelido 'decision_tree' -> Árvore
        ('knn', knn_model)               # apelido 'knn' -> k-NN
    ],
    voting='hard'  # 'hard' = maioria simples das classes preditas
)

# TREINO DO ENSEMBLE:
# COMPUTAÇÃO: chama .fit() de cada estimador de novo (re-treina internamente).
# Como log/dt/knn já foram treinados acima, isso os re-ajusta aos mesmos dados
# (custo pequeno aqui: n=120). Em pipelines reais, normalmente só se treina via
# o ensemble, sem o fit individual separado.
ensemble_model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 7. PREDIÇÃO COM O ENSEMBLE
# ---------------------------------------------------------------------------
# COMPUTAÇÃO: .predict(X_test) pede a predição de cada um dos 3 modelos para
# as 30 amostras de teste (matriz 30x4) e aplica a moda linha a linha.
# Resultado y_pred: vetor de 30 rótulos {0,1,2}.
y_pred = ensemble_model.predict(X_test)

# ---------------------------------------------------------------------------
# 8. AVALIAÇÃO: ACURÁCIA
# ---------------------------------------------------------------------------
# MATEMÁTICA:
#   Acurácia = (nº de acertos) / (nº total) = (1/n) Σ 1(ŷ_i == y_i)
#   Varia de 0 a 1. No Iris com 80/20, espera-se ~0.95-1.00, pois as classes
#   são quase linearmente separáveis (setosa é totalmente separável).
#   Limitação: com classes balanceadas ok, mas com classes desbalanceadas a
#   acurácia pode enganar (ex: 99% de uma classe). Aí usar F1/ROC.
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy:.2f}")  # :.2f = 2 casas decimais

# ---------------------------------------------------------------------------
# 9. AVALIAÇÃO DOS MODELOS INDIVIDUAIS (para comparar com o ensemble)
# ---------------------------------------------------------------------------
# COMPUTAÇÃO: reutiliza os modelos já treinados para prever o mesmo X_test.
# Isso permite responder: "o ensemble superou (ou empatou com) o melhor
# modelo sozinho?" — teste empírico da hipótese de ensembles.
y_pred_log = log_model.predict(X_test)  # predições da Regressão Logística
y_pred_dt = dt_model.predict(X_test)    # predições da Árvore de Decisão
y_pred_knn = knn_model.predict(X_test)  # predições do k-NN

# Cada accuracy_score compara o vetor predito com o gabarito y_test.
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.2f}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(f"k-NN Model Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
print(f"Ensemble Model Accuracy: {accuracy:.2f}")  # repete p/ facilitar comparação
