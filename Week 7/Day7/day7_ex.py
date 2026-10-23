"""
================================================================================
 day7_ex.py - Classificacao de Churn de Clientes (Telco)
================================================================================

Objetivo: prever se um cliente vai cancelar o servico (Churn = 1) ou nao (Churn = 0)
usando classificacao supervisionada com Random Forest, XGBoost e LightGBM.

Observacao importante:
    O codigo do professor NAO continha o arquivo .csv (dataset de entrada).
    O arquivo .csv mais correto para usar neste exercicio e o da Semana 8, Dia 7
    (Telco-Customer-Churn.csv), que deve estar na mesma pasta deste script.

Dependencias necessarias (instale se necessario):
    pip install pandas scikit-learn imbalanced-learn xgboost lightgbm
================================================================================
"""

# ==============================================================================
# 1. IMPORTACOES
# ==============================================================================
import pandas as pd  # Manipulacao de dados tabulares (DataFrames)

from sklearn.preprocessing import LabelEncoder, StandardScaler  # Codificacao e normalizacao
from imblearn.over_sampling import SMOTE  # Balanceamento de classes (minoria)
from sklearn.model_selection import train_test_split  # Particao treino/teste
from sklearn.ensemble import RandomForestClassifier  # Floresta aleatoria (bagging)
from xgboost import XGBClassifier  # Extreme Gradient Boosting
from lightgbm import LGBMClassifier  # Light Gradient Boosting Machine
from sklearn.metrics import classification_report, roc_auc_score  # Avaliacao de modelo

# ==============================================================================
# 2. CARREGAMENTO DO DATASET
# ==============================================================================
# pd.read_csv le um arquivo de valores separados por virgula e devolve um DataFrame.
# Cada linha e um cliente; cada coluna e uma feature (atributo) ou o alvo (Churn).
df = pd.read_csv("Telco-Customer-Churn.csv")

# Exibe informacoes gerais (tipos de dados, valores nulos, memoria)
print("Dataset Info: \n")
print(df.info())

# Mostra a distribuicao da classe alvo (quantos clientes fizeram Churn vs nao fizeram)
print("\n Class Distribution: \n")
print(df['Churn'].value_counts())

# Exibe as primeiras 5 linhas para inspecao visual
print("\n Sample Data:\n", df.head())

# ==============================================================================
# 3. TRATAMENTO DE VALORES AUSENTES (MISSING VALUES)
# ==============================================================================
# 'TotalCharges' pode vir como string com espacos em branco; pd.to_numeric converte
# para numero, transformando valores invalidos em NaN (erros='coerce').
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Preenche NaN com a mediana da coluna (imputacao por mediana).
# A mediana e robusta a outliers (ao contrario da media).
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

# ==============================================================================
# 4. CODIFICACAO DE VARIAVEIS CATEGORICAS (ENCODING)
# ==============================================================================
label_encoder = LabelEncoder()

# Percorre todas as colunas do tipo objeto (texto/categorico).
# LabelEncoder mapeia cada categoria para um inteiro: ex. {'No':0, 'Yes':1}.
# Nao codificamos 'Churn' aqui porque ele e tratado separadamente como alvo.
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = label_encoder.fit_transform(df[column])

# Codifica a variavel alvo (target): 'No' -> 0, 'Yes' -> 1 (classificacao binaria).
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# ==============================================================================
# 5. ESCALAMENTO / NORMALIZACAO DE FEATURES NUMERICAS
# ==============================================================================
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# StandardScaler aplica padronizamento (Z-score):
#     z = (x - media) / desvio_padrao
# Resultado: media = 0 e desvio padrao = 1 para cada feature.
# Importante para algoritmos sensíveis à escala (distancias, gradientes).
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# ==============================================================================
# 6. SEPARACAO EM FEATURES (X) E ALVO (y)
# ==============================================================================
# X: matriz de features (tudo menos a coluna alvo 'Churn').
# y: vetor alvo (0 ou 1).
X = df.drop(columns=['Churn'])
y = df['Churn']

# ==============================================================================
# 7. PARTICAO TREINO / TESTE
# ==============================================================================
# train_test_split divide os dados em 80% treino e 20% teste.
# random_state=42 garante reprodutibilidade (mesma divisao a cada execucao).
# stratify nao foi usado aqui, mas e recomendado quando classes estao desbalanceadas.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================================================================
# 8. BALANCEAMENTO DE CLASSES COM SMOTE
# ==============================================================================
# Problema: classes desbalanceadas (poucos clientes fazem Churn).
# O modelo tende a prever sempre a classe majoritaria e ainda assim ter alta acuracia.
#
# SMOTE (Synthetic Minority Over-sampling Technique):
#   Para cada amostra da classe minoria, escolhe-se um de seus k-vizinhos mais
#   proximos (espaco de features) e gera um novo ponto sintético por interpolacao
#   linear entre os dois:
#
#       x_sintetico = x_i + aleatorio * (x_vizinho - x_i),   aleatorio in [0,1]
#
# Isso aumenta a classe minoria artificialmente, forçando o classificador a
# aprender a fronteira de decisao da minoria em vez de ignorá-la.
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Mostra a distribuicao apos o balanceamento (agora classes estao equilibradas).
print("\n Class Distribution after SMOTE: \n")
print(pd.Series(y_train_resampled).value_counts())

# ==============================================================================
# 9. TREINAMENTO E AVALIACAO DOS MODELOS
# ==============================================================================

# ----------------------------------------------------------------
# 9.1 Random Forest (Bagging de arvores de decisao)
# ----------------------------------------------------------------
# RandomForest constroi N arvores de decisao em subconjuntos aleatorios de
# linhas e colunas (bagging + feature randomness). A previsao final eh a
# votos majoritario (classificacao) ou media (regressao).
# Reduz variancia (overfitting) em relacao a uma arvore unica.
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# predict retorna a classe prevista (0 ou 1).
y_pred_rf = rf_model.predict(X_test)

# predict_proba(...[:, 1]) retorna a probabilidade estimada da classe positiva (1).
# ROC-AUC usa essas probabilidades, nao as classes binarias.
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# ----------------------------------------------------------------
# 9.2 XGBoost (Extreme Gradient Boosting)
# ----------------------------------------------------------------
# Gradient Boosting constroi arvores sequencialmente: cada nova arvore corrige
# os erros (residuos/gradientes) da combinacao das anteriores.
# XGBoost e uma implementacao otimizada com regularizacao (L1/L2), tratamento
# de valores ausentes e paralelizacao.
# eval_metric='logloss' monitora a perda de entropia cruzada durante o treino.
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

# ----------------------------------------------------------------
# 9.3 LightGBM (Light Gradient Boosting Machine)
# ----------------------------------------------------------------
# Semelhante ao XGBoost, mas cresce as arvores folha-a-folha (leaf-wise)
# escolhendo a folha que mais reduz a perda, ao inves de nível-a-nível.
# Mais rapido e consome menos memoria em datasets grandes.
lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_lgb = lgb_model.predict(X_test)
roc_auc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])

# ==============================================================================
# 10. RELATORIOS DE CLASSIFICACAO
# ==============================================================================
# classification_report exibe, para cada classe:
#   - Precision:     TP / (TP + FP)  -> quantos previstos como positivos realmente sao
#   - Recall (Sensibilidade): TP / (TP + FN) -> quantos reais positivos foram capturados
#   - F1-Score:      2 * (P * R) / (P + R)  -> media harmonica de precisao e recall
#   - Support:       quantidade de amostras reais de cada classe no teste
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))
print("XGBoost Report: \n", classification_report(y_test, y_pred_xgb))
print("LightGBM Report: \n", classification_report(y_test, y_pred_lgb))

# ==============================================================================
# 11. COMPARACAO POR ROC-AUC
# ==============================================================================
# ROC-AUC (Area Under the Receiver Operating Characteristic Curve):
#   - Plota Taxa de Verdadeiros Positivos (TPR) vs Taxa de Falsos Positivos (FPR)
#     variando o limiar de classificacao.
#   - AUC = probabilidade de que o modelo atribua maior nota a um exemplo positivo
#     aleatorio do que a um exemplo negativo aleatorio.
#   - 0.5 = chute aleatorio; 1.0 = classificacao perfeita.
# Usar AUC (e nao apenas acuracia) e essencial em classes desbalanceadas.
print("ROC-AUC Scores: \n")
print(f"Random Forest: {roc_auc_rf:.2f}")
print(f"XGBoost: {roc_auc_xgb:.2f}")
print(f"LightGBM: {roc_auc_lgb:.2f}")
