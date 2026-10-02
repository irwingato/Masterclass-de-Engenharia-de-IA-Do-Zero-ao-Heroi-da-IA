"""
===============================================================================
 Titanic - Encoding Categórico + Regressão Logística
-------------------------------------------------------------------------------
 OBJETIVO (Ciência da Computação / Machine Learning):
   - Demonstrar o pipeline clássico de ML supervisionado:
     1. Carregar dados tabulares (CSV remoto)
     2. Transformar variáveis categóricas em números (Encoding)
     3. Dividir em treino/teste (avaliação de generalização)
     4. Treinar um classificador linear (Regressão Logística)
     5. Medir acurácia.

 CONCEITO MATEMÁTICO GERAL:
   - Modelos matemáticos operam em R^n (vetores reais), não em strings
     como "male"/"female" ou "S"/"C"/"Q".
     Por isso precisamos de uma função de codificação:
       f: Categoria -> vetor numérico em R^d
===============================================================================
"""

# ----------------------------------------------------------------------------
# 1. IMPORTAÇÕES
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - `pandas`: biblioteca para DataFrames (tabela 2D em memória, O(n*m) espaço,
#   onde n=linhas, m=colunas). Operações vetorizadas em C/numpy -> O(n) rápido.
# - `sklearn`: biblioteca de ML com API padrão: .fit(X,y) treina, .predict(X)
#   prediz. Abstrai otimização numérica do usuário.
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # codificador ordinal 0..k-1
from sklearn.model_selection import train_test_split  # divisão treino/teste
from sklearn.linear_model import LogisticRegression  # classificador linear
from sklearn.metrics import accuracy_score  # métrica de avaliação


# ----------------------------------------------------------------------------
# 2. CARREGAMENTO DO DATASET
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - Leitura de CSV remoto via HTTP GET. `pd.read_csv(url)` faz:
#   parsing do texto -> inferência de tipos -> alocação de DataFrame.
#   Complexidade de tempo O(n*m) para n linhas e m colunas.
# MATEMÁTICA / ESTATÍSTICA:
# - Dataset Titanic: cada passageiro i é uma amostra (x_i, y_i),
#   onde y_i ∈ {0,1} (0=morreu, 1=sobreviveu) -> classificação binária.
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# ----------------------------------------------------------------------------
# 3. INSPEÇÃO INICIAL
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - `.info()` mostra schema: nº de entradas, tipos (int64, float64, object),
#   memória usada e valores não-nulos. Essencial para detectar NaN (ex: Age,
#   Cabin) que quebram modelos matemáticos (NaN propaga em soma/produto).
# - `.head()` mostra as 5 primeiras linhas: custo O(1), só para inspeção humana.
#   Não altera os dados.
# Display dataset information
print("Dataset Info:")
print(df.info())

# Preview the first few rows
print("\n Dataset Preview:")
print(df.head())

# ----------------------------------------------------------------------------
# 4. ONE-HOT ENCODING (Codificação Um-Quente)
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - Variáveis como `Sex` e `Embarked` são NOMINAIS: não há ordem
#   (male < female não faz sentido). Não podemos usar 0/1/2 arbitrário pois
#   o modelo interpretaria ordem falsa.
# - Solução: expandir 1 coluna categórica com k valores em k colunas binárias.
# MATEMÁTICA:
# - Se C = {c1, c2, ..., ck}, o one-hot de cj é o vetor canônico e_j ∈ R^k:
#     ex: Sex={male,female} -> male=[1,0], female=[0,1]
# - `drop_first=True` remove 1 coluna para evitar a "dummy trap":
#   Álgebra Linear: k colunas one-hot somam 1 (colinearidade perfeita),
#   matriz X^T*X vira singular (não-inversível). Com k-1 colunas recuperamos
#   a informação (a categoria removida é quando todas são 0) e mantemos
#   posto (rank) completo. Ex: só `Sex_male` basta: 1=male, 0=female.
# - Custo: aumenta a dimensão d (curse of dimensionality se k for enorme).
# Applt One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Display encoded dataset
print("\n One-Hot Encoded dataset:")
print(df_one_hot.head())

# ----------------------------------------------------------------------------
# 5. LABEL ENCODING (Codificação por Rótulo)
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - `LabelEncoder` cria um dicionário bijetivo categoria <-> inteiro:
#   fit() aprende o vocabulário (ordenado alfabeticamente), transform()
#   aplica o mapeamento em O(n).
# - Ideal para variável ORDINAL (há ordem natural).
# MATEMÁTICA:
# - Função f: C -> {0,1,...,k-1}. Ex: Pclass={1,2,3} -> {0,1,2} ou similar.
#   ATENÇÃO: impõe distância artificial: |2-1|=1. Só é válido se a ordem
#   importar (1ª classe > 2ª > 3ª em status). Para nominal cria ordem falsa
#   que confunde modelos lineares.
# Apply Label Encoding
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# Display encoded dataset
print("\n Label Encoded Dataset:")
print(df[['Pclass', 'Pclass_encoded']].head())

# ----------------------------------------------------------------------------
# 6. FREQUENCY ENCODING (Codificação por Frequência)
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - Para alta cardinalidade (ex: `Ticket` tem centenas de códigos únicos),
#   one-hot criaria centenas de colunas esparsas (matriz gigante, overfitting).
#   Alternativa compacta: trocar a categoria pela sua contagem/frequência.
#   `value_counts()` conta ocorrências em O(n) com hash map, `map()` substitui.
# MATEMÁTICA:
# - Seja N = nº total de linhas, count(t) = nº de vezes que o ticket t aparece.
#   Codificação: f(t) = count(t)  (ou frequência empírica p(t)=count(t)/N).
# - Ideia probabilística: tickets compartilhados (famílias/grupos) têm count>1,
#   o que pode correlacionar com sobrevivência (embarcar em grupo).
#   Transformamos informação categórica em um prior estatístico numérico.
# APply Frequency Encoding
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# Display frequency encoded feature
print("\n Frequence ENcoded Feature: ")
print(df[['Ticket', 'Ticket_frequency']].head())

# ----------------------------------------------------------------------------
# 7. DEFINIÇÃO DE X (features) e y (alvo)
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - Paradigma supervisionado: X = matriz de design (n x d), y = vetor rótulo (n,).
# - Removemos colunas inutilizáveis diretamente: `Survived` (é o y, vazamento
#   se ficar em X), `Name/Ticket/Cabin` (strings de alta cardinalidade, precisam
#   de NLP/engenharia própria), `Age` (contém NaN -> LogisticRegression puro
#   não aceita NaN, então aqui é descartada por simplicidade; ideal seria imputar).
# MATEMÁTICA:
# - Cada passageiro vira um ponto x_i ∈ R^d. O modelo aprenderá um hiperplano
#   que separa sobreviventes de não-sobreviventes nesse espaço.
X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'Age'])
y = df['Survived']

# ----------------------------------------------------------------------------
# 8. DIVISÃO TREINO / TESTE
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - `train_test_split(test_size=0.2)`: embaralha (shuffle O(n)) e separa 80%
#   treino / 20% teste. `random_state=42` fixa a semente do gerador
#   pseudo-aleatório (PRNG) -> reprodutibilidade: mesma divisão toda vez.
# MATEMÁTICA / TEORIA DE ML:
# - Objetivo: estimar ERRO DE GENERALIZAÇÃO. Treinar e testar nos mesmos dados
#   subestima o erro (overfitting/decoreba).
# - Validação hold-out: minimizamos risco empírico em D_train e estimamos
#   risco real em D_test (dados i.i.d. nunca vistos). É a base do dilema
#   viés-variância: mais treino = melhor ajuste, mas precisamos de teste
#   suficiente para estimativa estável.
# SPlit dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------
# 9. REGRESSÃO LOGÍSTICA (Classificador Linear Probabilístico)
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - Apesar do nome "regressão", é CLASSIFICAÇÃO. API: .fit() otimiza pesos,
#   .predict() aplica limiar 0.5. `max_iter=200` = nº máximo de iterações do
#   otimizador (L-BFGS por padrão). Se não convergir, aumenta-se max_iter.
# MATEMÁTICA (o coração do modelo):
# - Modelo linear: z = w^T * x + b, onde w ∈ R^d (pesos), b ∈ R (bias).
# - Função sigmoide/logística comprime z em probabilidade:
#     σ(z) = 1 / (1 + e^{-z})  ∈ (0,1)
#     P(y=1|x) = σ(w^T x + b)
# - Treino = minimizar a LOG-LOSS (entropia cruzada binária):
#     L(w,b) = -1/N * Σ [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]
#   via gradiente descendente / quasi-Newton. É função convexa -> tem mínimo
#   global único (ótimo para otimização).
# - Decisão: ŷ = 1 se p >= 0.5 (fronteira = hiperplano w^T x + b = 0).
# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# 10. PREDIÇÃO E AVALIAÇÃO (ACURÁCIA)
# ----------------------------------------------------------------------------
# CIÊNCIA DA COMPUTAÇÃO:
# - `predict(X_test)` faz multiplicação matriz-vetor X_test @ w + b e aplica
#   sigmoide + limiar, tudo vetorizado O(n_test * d).
# MATEMÁTICA:
# - Acurácia = fração de acertos:
#     acc = (TP + TN) / (TP + TN + FP + FN) = 1/N_test * Σ 1[ŷ_i == y_i]
#   onde TP=True Positive, etc. (matriz de confusão).
# - Limitação: se classes são desbalanceadas, acurácia engana (ex: 90% de uma
#   classe). Ideal complementar com precisão, recall, F1 e ROC-AUC.
# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy with One-Hot Encoding:", accuracy_score(y_test, y_pred))
