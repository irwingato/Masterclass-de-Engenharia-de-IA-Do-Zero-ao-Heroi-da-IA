# ============================================================
# DAY 6 EX 1 - CLASSIFICAÇÃO BINÁRIA COM REGRESSÃO LOGÍSTICA
# Dataset: Iris (150 flores, 4 features, 3 espécies)
# Objetivo: distinguir "Setosa (classe 0)" vs "Não-Setosa"
# ============================================================

# --- 1. IMPORTS ---
# Ciência da Computação: usamos a biblioteca scikit-learn, padrão de facto
# para ML clássico em Python. Ela implementa estimadores com interface
# uniforme .fit()/.predict().
from sklearn.datasets import load_iris  # função que carrega o dataset Iris em memória
from sklearn.model_selection import train_test_split  # utilitário p/ divisão treino/teste (holdout)
from sklearn.linear_model import LogisticRegression  # classificador linear probabilístico
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay  # métricas de classificação
import matplotlib.pyplot as plt  # biblioteca de visualização

# --- 2. CARREGAMENTO DOS DADOS ---
# load_iris() retorna um objeto Bunch (dict-like) com:
#   .data   -> matriz X de shape (150, 4): [sepal length, sepal width, petal length, petal width] em cm
#   .target -> vetor y de shape (150,) com valores {0:Setosa, 1:Versicolor, 2:Virginica}
data = load_iris()
X = data.data  # Matriz de features. Matemática: X ∈ R^(n×d), n=150 amostras, d=4 dimensões.

# Binarização do problema: transformamos um problema multinomial (3 classes)
# em um problema binário (2 classes) para usar Regressão Logística binária.
#   y_i = 1 se target == 0 (Setosa), 0 caso contrário.
# Computação: (data.target == 0) cria array booleano, .astype(int) converte True->1, False->0.
y = (data.target == 0).astype(int)  # Vetor alvo binário: y ∈ {0,1}^n

# --- 3. DIVISÃO TREINO / TESTE (HOLDOUT) ---
# Ciência da Computação: precisamos avaliar generalização, não memorização.
# Separamos 80% para treino e 20% para teste (test_size=0.2).
#   random_state=42 -> semente do gerador pseudo-aleatório (reprodutibilidade).
#   Sem isso, cada execução embaralharia diferente e o resultado não seria comparável.
# Matemática: amostragem i.i.d. Esperamos que X_train, X_test venham da mesma distribuição P(X,y).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TREINO DA REGRESSÃO LOGÍSTICA ---
# Matemática da Regressão Logística:
#   1. Modelo linear: z = w^T x + b, onde w ∈ R^d são pesos, b ∈ R é o bias.
#   2. Função sigmoide (logística): h(x) = σ(z) = 1 / (1 + e^(-z)) ∈ (0,1).
#      Interpretação probabilística: P(y=1 | x) = h(x).
#   3. Função de custo (Log Loss / Entropia Cruzada Binária):
#      J(w,b) = -1/m * Σ [ y_i*log(h_i) + (1-y_i)*log(1-h_i) ]
#      Deriva da máxima verossimilhança sob modelo Bernoulli.
#   4. Otimização: sklearn usa por defeito o solver 'lbfgs' (quasi-Newton),
#      que usa gradiente ∇J e aproximação da Hessiana para achar w*,b* que minimizam J.
#      É convexo -> tem mínimo global único (diferente de redes neurais).
#   5. Regra de decisão: prevê 1 se h(x) >= 0.5 (i.e. z >= 0), senão 0.
#      Geometricamente, w^T x + b = 0 é um hiperplano (num espaço 4D aqui).
model = LogisticRegression()  # cria o estimador com hiperparâmetros padrão (C=1.0, regularização L2)
model.fit(X_train, y_train)  # resolve o problema de otimização acima nos dados de treino

# --- 5. PREDIÇÃO ---
# Computação: aplica a regra de decisão aprendida a cada x em X_test.
# Retorna vetor y_predict ∈ {0,1}^k, k = nº amostras de teste (~30).
y_predict = model.predict(X_test)

# --- 6. MATRIZ DE CONFUSÃO ---
# Ciência da Computação: tabela 2x2 que cruza Rótulo Real x Rótulo Previsto:
#                Prev 0 (Not Class 0) | Prev 1 (Class 0)
#   Real 0 (Not) |       TN           |      FP (Erro Tipo I)
#   Real 1 (Is)  |       FN (Tipo II) |      TP
# Matemática: daqui derivam todas as métricas de classificação.
cm = confusion_matrix(y_test, y_predict)  # calcula TN, FP, FN, TP

# Visualização da matriz com cores (cmap="Blues": quanto mais escuro, maior a contagem).
# display_labels deve estar na ordem [classe 0, classe 1] do vetor y.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Class 0", "Class 0"])
disp.plot(cmap="Blues")  # cria a figura matplotlib com anotações numéricas
plt.title("Confusion Matrix")  # título do gráfico
plt.show()  # renderiza a janela/figura (bloqueante em scripts locais)

# --- 7. RELATÓRIO DE CLASSIFICAÇÃO ---
# Matemática das métricas (com TP,FP,FN,TN):
#   Accuracy  = (TP+TN) / total              -> fração total de acertos.
#   Precision = TP / (TP+FP)                  -> "dos que eu disse que são Setosa, quantos acertaram?"
#   Recall    = TP / (TP+FN) = TPR            -> "das Setosas reais, quantas eu capturei?" (sensibilidade)
#   F1-score  = 2*P*R/(P+R)                   -> média harmônica de Precision e Recall (penaliza extremos).
#   Support   = nº ocorrências reais de cada classe no teste.
# Ciência da Computação: accuracy sozinha engana em dados desbalanceados.
# Aqui ~1/3 é Setosa, então precision/recall por classe contam a história real.
print("\n Classification Report: ")
print(classification_report(y_test, y_predict))
