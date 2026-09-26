import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# --- 1. Geração de dados sintéticos ---
# Fixa a semente para reprodutibilidade (mesmos números aleatórios a cada execução).
np.random.seed(42)
n_samples = 200
# X ~ Uniforme[0, 10) com 2 colunas: coluna 0 = "Age", coluna 1 = "Salary" (valores fictícios, não realistas).
X = np.random.rand(n_samples, 2) * 10
# Rótulo linearmente separável: y = 1 se 1.5*x0 + x1 > 15, senão 0.
# Matemática: fronteira verdadeira é a reta 1.5*x0 + x1 = 15  <=>  x1 = -1.5*x0 + 15.
# É o caso ideal para a Regressão Logística, que aprende exatamente uma fronteira linear.
y = (X[:, 0] * 1.5 + X[:, 1] > 15).astype(int)

# Organiza em DataFrame para manipulação (nomes de colunas ajudam no split e no plot).
df = pd.DataFrame(X, columns=['Age', 'Salary'])
df['Purchase'] = y

# --- 2. Divisão treino/teste ---
# 80% treino / 20% teste (test_size=0.2), random_state=42 garante o mesmo embaralhamento.
# Objetivo: estimar o erro de generalização em dados não vistos (evita avaliar no próprio treino).
X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'Salary']], df['Purchase'], test_size=0.2, random_state=42)

# --- 3. Treino da Regressão Logística ---
# Modelo: p(y=1|x) = sigma(z), onde z = w1*Age + w2*Salary + b (escore linear) e sigma = sigmoide.
# Hipótese de decisão: prevê 1 se p >= 0.5 <=> z >= 0 (fronteira: w.x + b = 0, uma reta).
# Função de custo minimizada (log-loss / entropia cruzada binária):
#   J(w,b) = -1/m * soma[ y*log(p) + (1-y)*log(1-p) ] (+ regularização L2 por padrão no sklearn, C=1.0).
# Otimização interna do sklearn (solver lbfgs): usa gradiente dJ/dw = 1/m * X^T (p - y).
model = LogisticRegression()
model.fit(X_train, y_train)

# Aplica a regra p >= 0.5 nos dados de teste para obter classes 0/1.
y_pred = model.predict(X_test)

# --- 4. Avaliação ---
# Accuracy  = (TP+TN)/(total): fração de acertos. Enganosa se classes desbalanceadas.
# Precision = TP/(TP+FP): de tudo que previu como 1, quanto acertou (evita falso positivo).
# Recall    = TP/(TP+FN): de todos os 1 reais, quantos capturou (evita falso negativo).
# F1 = 2*P*R/(P+R): média harmônica entre Precision e Recall.
# classification_report detalha isso por classe (support = nº de amostras de cada classe).
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# --- 5. Visualização da fronteira de decisão ---
# Limites dos eixos com margem de 1 para enquadrar todos os pontos.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# Cria grade fina (passo 0.1) cobrindo o plano Age x Salary: xx, yy têm shape (ny, nx).
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Para cada ponto da grade, prevê a classe (0/1) -> estima onde fica a reta w.x + b = 0.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# contourf pinta as duas regiões de decisão (fundo); scatter sobrepõe os pontos reais de teste,
# coloridos pelo rótulo verdadeiro (c=y_test) com borda preta para ver erros (cor do fundo ≠ cor do ponto = erro).
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_test['Age'], X_test['Salary'], c=y_test, edgecolor="k", cmap="coolwarm")
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()