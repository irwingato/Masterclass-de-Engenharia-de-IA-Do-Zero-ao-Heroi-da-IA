# --- Imports ---
# LogisticRegression: classificador linear probabilístico.
#   Matemática (multiclasse, Iris tem 3 classes): usa softmax
#     P(y=k|x) = exp(w_k·x + b_k) / soma_j exp(w_j·x + b_j)
#   Treino minimiza entropia cruzada + regularização L2:
#     J(W) = -(1/m) * soma_i log P(y_i|x_i) + (C^-1)*||W||²
#   Otimizado por LBFGS (quasi-Newton). max_iter=200 abaixo é o
#   nº máximo de iterações desse otimizador.
from sklearn.linear_model import LogisticRegression
# load_iris: X (150,4), y (150,) com 3 classes balanceadas.
from sklearn.datasets import load_iris
# confusion_matrix: matriz C onde C[i,j] = nº de amostras com
#   rótulo verdadeiro i preditas como j. Diagonal = acertos.
# ConfusionMatrixDisplay: visualização da matriz.
# classification_report: calcula Precision, Recall, F1 por classe:
#   Precision_k = TP_k / (TP_k + FP_k)  -> "quando prevejo k, quanto acerto?"
#   Recall_k    = TP_k / (TP_k + FN_k)  -> "dos que são k, quanto recupero?"
#   F1_k = 2*P*R/(P+R) (média harmônica: pune se um dos dois for baixo)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# train_test_split: holdout simples (1 treino + 1 teste).
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# --- Carga dos dados ---
data = load_iris()
X, y = data.data, data.target

# --- Split treino/teste (holdout) ---
# test_size=0.2: 20% para teste -> 120 treino, 30 teste (150*0.2=30).
# random_state=42: sorteio reprodutível.
# Matemática: estimamos o risco real R = E[l(y, y_hat)] pelo risco
# empírico no teste: R_hat = (1/30) * soma l(y_i, y_hat_i).
# Atenção: sem stratify=y, as proporções das 3 classes no teste podem
# desbalancear no sorteio. Ideal seria stratify=y para manter ~10 de cada.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Treino ---
# max_iter=200: limite de iterações do LBFGS para convergir os pesos W,b.
# fit resolve argmin_W J(W) no X_train (120 amostras).
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# --- Predição ---
# Para cada x em X_test (30,4): calcula P(y=k|x) via softmax e retorna
#   y_pred = argmax_k P(y=k|x)  (classe mais provável).
y_pred = model.predict(X_test)

# --- Matriz de confusão ---
# cm é 3x3 (3 classes). Ex: cm[1,2] = versicolor classificada como virginica.
# Acurácia global = traço(cm)/soma(cm) = soma diagonal / 30.
# Off-diagonal = confusões: no Iris, o erro típico é 1 vs 2
# (versicolor vs virginica não são linearmente separáveis; setosa é).
cm = confusion_matrix(y_test, y_pred)

# --- Plot ---
# display_labels: nomes das classes nos eixos (setosa, versicolor, virginica).
# cmap="Blues": intensidade do azul = contagem na célula.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap="Blues")
plt.show()

# --- Relatório por classe ---
# Mostra por classe: precision, recall, f1-score e support (nº de amostras
# reais da classe no teste). No final mostra:
#   macro avg: média simples das 3 classes (trata todas igual).
#   weighted avg: média ponderada pelo support (dá mais peso a classe frequente).
print("\nClassification Report:\n", classification_report(y_test, y_pred))