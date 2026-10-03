# ============================================================================
# DAY 4 - SELEÇÃO DE FEATURES: CORRELAÇÃO, INFORMAÇÃO MÚTUA E RANDOM FOREST
# Dataset: Diabetes (sklearn) - prever progressão da doença (regressão)
# ============================================================================

# --- 1. IMPORTS: CARREGAMENTO DE DADOS E VISUALIZAÇÃO ---
from sklearn.datasets import load_diabetes  # Dataset pronto do sklearn: 442 pacientes x 10 features fisiológicas
import pandas as pd  # DataFrames: tabela 2D eficiente (NumPy por baixo) para manipular dados
import seaborn as sns  # Heatmaps estatísticos bonitos (construído sobre matplotlib)
import matplotlib.pyplot as plt  # Plotagem base do Python
from sklearn.feature_selection import mutual_info_regression  # Estima Informação Mútua p/ regressão

# --- 2. LOAD DO DATASET ---
# COMPUTAÇÃO: load_diabetes() retorna um objeto Bunch (dict-like) com:
#   .data (matriz 442x10 normalizada), .target (vetor 442,), .feature_names (10 nomes)
#   Ex: age, sex, bmi, bp (pressão), s1..s6 (medidas sanguíneas)
# MATEMÁTICA: é um problema de REGRESSÃO SUPERVISIONADA: aprender f: R^10 -> R
#   onde y = progressão da doença 1 ano após baseline.
data = load_diabetes()
# Cria DataFrame: cada coluna = uma feature. Isso dá rótulos e facilita df.corr(), df.drop(), etc.
df = pd.DataFrame(data.data, columns=data.feature_names)
# Adiciona a variável alvo como coluna 'target' -> df fica 442 linhas x 11 colunas
df['target'] = data.target

# --- 3. INSPEÇÃO INICIAL (comentado) ---
# Display Dataset information
# df.head() = mostra as 5 primeiras linhas (checar escala, NaNs, tipos)
# df.info() = nº de linhas, tipos (float64), memória, valores não-nulos -> detecta dados faltantes
# print(df.head())
# print(df.info())

# --- 4. MATRIZ DE CORRELAÇÃO ---
# COMPUTAÇÃO: df.corr() calcula correlação de Pearson par-a-par entre as 11 colunas.
#   Complexidade O(p² * n) onde p=11 colunas, n=442 linhas -> aqui é trivial (~121*442 ops).
# MATEMÁTICA: Para cada par (X, Y), coeficiente de Pearson:
#   r = Cov(X,Y) / (std(X) * std(Y)) = Σ(xi-x̄)(yi-ȳ) / sqrt(Σ(xi-x̄)² * Σ(yi-ȳ)²)
#   r ∈ [-1, +1]: +1 = linear positiva perfeita, -1 = linear negativa perfeita, 0 = sem relação LINEAR.
#   LIMITAÇÃO: só captura relações LINEARES. Uma parábola y=x² teria r≈0 mas é totalmente dependente!
#   Por isso depois usamos Informação Mútua + Random Forest (capturam não-linearidades).
correlation_matrix = df.corr()

# --- 5. HEATMAP DA CORRELAÇÃO (comentado) ---
# COMPUTAÇÃO: sns.heatmap desenha a matriz 11x11 como cores; annot=True escreve o valor r em cada célula.
#   cmap="coolwarm" = azul (negativo) -> branco (0) -> vermelho (positivo). figsize=(10,8) = tamanho em polegadas.
# INTERPRETAÇÃO: linha/coluna 'target' mostra quais features têm |r| alto com o alvo.
#   Também revela MULTICOLINEARIDADE (features correlacionadas entre si, ex: s1 vs s2),
#   o que pode confundir modelos lineares.
# Plot heatmap
# plt.figure(figsize=(10,8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

# --- 6. RANKING POR CORRELAÇÃO COM O TARGET ---
# COMPUTAÇÃO: pega a coluna 'target' da matriz (10 correlações feature-alvo + 1.0 do target consigo mesmo)
#   e ordena decrescente. O topo = candidatas mais promissoras para modelo LINEAR.
# CUIDADO: correlação ≠ causalidade, e feature com r baixo ainda pode ser útil em conjunto / não-linearmente.
# Select features with high correlation to the target
correlated_features = correlation_matrix['target'].sort_values(ascending=False)
# print("Features Most Correlated with Target:")
# print(correlated_features)

# --- 7. SEPARAÇÃO X (features) e y (alvo) ---
# COMPUTAÇÃO/PARADIGMA ML: padrão supervisionado -> X = matriz de entrada (442, 10), y = vetor saída (442,).
#   df.drop(columns=['target']) retorna cópia sem o alvo (evita vazamento de dados / data leakage).
# Seperate featured and target
X = df.drop(columns=['target'])
y = df['target']

# --- 8. INFORMAÇÃO MÚTUA (MUTUAL INFORMATION) ---
# COMPUTAÇÃO: mutual_info_regression(X, y) estima I(X_j; y) para cada uma das 10 features.
#   Usa método de Kraskov et al. com k-vizinhos mais próximos (k=3 por padrão), não-paramétrico (sem supor Gaussiana).
# MATEMÁTICA: Informação Mútua mede REDUÇÃO DA INCERTEZA:
#   I(X;Y) = Σ p(x,y) * log[ p(x,y) / (p(x)*p(y)) ]   (caso discreto)
#        = H(X) + H(Y) - H(X,Y),  onde H = entropia de Shannon H(X) = -Σ p(x) log p(x)
#   I(X;Y) >= 0, em nats (log natural). I=0 ⟺ X e Y INDEPENDENTES.
#   DIFERENÇA P/ PEARSON: I detecta QUALQUER dependência (linear, quadrática, senoidal...),
#   enquanto r só detecta reta. Por isso mi_df complementa correlated_features.
#   OBS: é univariada (feature-a-feature) e estocástica (pequena variação entre runs por causa dos vizinhos).
# Calculate mutual information
mutual_info = mutual_info_regression(X, y)

# --- 9. DATAFRAME DE INFORMAÇÃO MÚTUA ---
# COMPUTAÇÃO: empacota o array mutual_info (shape (10,)) num DataFrame com nome da feature ao lado,
#   ordena do maior I para o menor -> ranking de relevância não-linear.
# Create a Dataframe for better visualization
mi_df = pd.DataFrame({'Feature': X.columns, "Mutual Information": mutual_info})
mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

# print("Mutual Information Scores:")
# print(mi_df)

from sklearn.ensemble import RandomForestRegressor  # Ensemble de árvores de decisão para regressão (bagging)
import numpy as np  # Aqui: base numérica (neste script, importado mas não usado diretamente)

# --- 10. TREINO DO RANDOM FOREST ---
# COMPUTAÇÃO: RandomForestRegressor(random_state=42) cria floresta com padrão n_estimators=100 árvores.
#   random_state=42 = semente fixa -> reprodutibilidade (mesmo bootstrap e mesmos splits toda vez).
#   model.fit(X, y): para cada árvore, sorteia amostra bootstrap (442 com reposição) + em cada nó testa
#   subconjunto aleatório de features (max_features=1.0 p/ regressão = todas, mas threshold aleatório).
# MATEMÁTICA: cada árvore particiona R^10 em retângulos minimizando VARIÂNCIA/MSE no split:
#   escolhe (feature j, limiar t) que maximiza Δ = Var(pai) - [n_esq/n * Var(esq) + n_dir/n * Var(dir)]
#   Predição final = MÉDIA das 100 árvores: ŷ = (1/100) Σ T_b(x). Isso reduz VARIÂNCIA sem aumentar muito o viés
#   (bagging). Custo de treino ≈ O(B * n log n * p), B=100 árvores.
# Train a Random Forest Model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# --- 11. FEATURE IMPORTANCE (IMPORTÂNCIA POR IMPUREZA) ---
# COMPUTAÇÃO: model.feature_importances_ retorna vetor de 10 valores que somam 1.0.
# MATEMÁTICA: "Mean Decrease in Impurity" (MDI): para cada feature j,
#   soma sobre TODOS os nós que dividiram em j, em TODAS as árvores:
#     importância(j) += (n_nó / n_total) * ΔVariância(nó)
#   depois normaliza para somar 1. Valor alto = feature muito usada para reduzir o MSE.
#   VANTAGEM vs correlação/MI: é MULTIVARIADA (considera interação entre features) e não-linear.
#   CUIDADO: tem viés p/ features contínuas de alta cardinalidade e não indica direção do efeito (+/-).
#   Para direção, usar correlação (sinal de r) ou SHAP/Partial Dependence.
# Get feature importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance from Random Forest:")
print(importance_df)

# --- 12. PLOT DE BARRAS HORIZONTAIS ---
# COMPUTAÇÃO: plt.barh(y=nomes, width=importância) desenha barras horizontais (melhor p/ ler nomes).
#   gca().invert_yaxis() inverte o eixo Y para o maior valor ficar no TOPO (pois barh começa embaixo).
# LEITURA ESPERADA NESTE DATASET: 'bmi' e 's5' costumam liderar nos 3 métodos (correlação, MI e RF),
#   o que dá confiança de que são os sinais mais robustos. Divergências entre os rankings são normais
#   e informativas: ex. feature com MI alta mas r baixo = relação não-linear pura.
# PLot feature importance
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance from Random Forest")
plt.show()