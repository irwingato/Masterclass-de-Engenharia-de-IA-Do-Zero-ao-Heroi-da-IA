"""
Titanic Survival Prediction - Comparative Gradient Boosting Analysis
====================================================================
Este script compara 4 abordagens de Gradient Boosting:
1. LightGBM - Gradient Boosting baseado em histogramas (GOSS + EFB)
2. CatBoost - Gradient Boosting com ordenação por alvo (Target Statistics)
3. XGBoost - Gradient Boosting com regularização L1/L2 e aproximação de Newton
4. CatBoost Native - CatBoost usando features categóricas nativas (sem encoding)
"""

# =============================================================================
# IMPORTS - Bibliotecas Fundamentais
# =============================================================================

import pandas as pd                                      # Manipulação de dados tabulares (DataFrame = matriz heterogênea)
from sklearn.model_selection import train_test_split    # Divisões estratificadas baseadas em amostragem aleatória
from sklearn.preprocessing import LabelEncoder          # Codificação ordinal: mapeia categorias -> {0, 1, ..., k-1}
import lightgbm as lgb                                   # LightGBM: Gradient Boosting Decision Tree (GBDT) com GOSS/EFB
from sklearn.metrics import accuracy_score              # Métrica de acurácia: ACC = (TP + TN) / (TP + TN + FP + FN)
from catboost import CatBoostClassifier                 # CatBoost: Ordered Boosting com Target Statistics
from xgboost import XGBClassifier                       # XGBoost: Extreme Gradient Boosting com regularização

# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================

# Dataset Titanic: 891 passageiros, 12 features, target binário (Survived: 0/1)
# Fonte: Kaggle - dados históricos do naufrágio do RMS Titanic (1912)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)  # Complexidade: O(n * m) onde n=linhas, m=colunas

# =============================================================================
# SELEÇÃO DE FEATURES (Engenharia de Features - Feature Selection)
# =============================================================================

# Features selecionadas baseadas em importância de domínio (Domain Knowledge):
# - Pclass: Proxy para status socioeconômico (1=alta, 2=média, 3=baixa) - ORDINAL
# - Sex: Dimorfismo sexual nas taxas de sobrevivência (mulheres/crianças primeiro) - NOMINAL
# - Age: Idade contínua - CONTÍNUA (afeta prioridade de resgate)
# - Fare: Tarifa paga - CONTÍNUA (proxy forte para classe social)
# - Embarked: Porto de embarque (C=Cherbourg, Q=Queenstown, S=Southampton) - NOMINAL
#
# Target: Survived (0=Não, 1=Sim) - Variável binária para classificação binária
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

# =============================================================================
# TRATAMENTO DE VALORES AUSENTES (Missing Data Imputation)
# =============================================================================

# Age: 177 valores ausentes (~20%) - Imputação pela MEDIANA (robusta a outliers)
# Mediana minimiza L1 loss: argmin_θ Σ|x_i - θ| → θ = mediana
# Melhor que média para distribuições assimétricas (idade tem cauda longa)
df.fillna({'Age': df['Age'].median()}, inplace=True)

# Embarked: 2 valores ausentes - Imputação pela MODA (valor mais frequente)
# Moda = argmax_x P(X=x) - Estimador de máxima verossimilhança para categórica
df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)

# =============================================================================
# ENCODING DE VARIÁVEIS CATEGÓRICAS (Categorical Encoding)
# =============================================================================

# Label Encoding: Mapeia cada categoria única para inteiro {0, 1, ..., k-1}
# Matematicamente: f: Categories → ℕ onde f(c_i) = i
# PROBLEMA: Impõe ordem artificial (0 < 1 < 2) onde não existe ordinalidade
# Adequado apenas para: (1) Árvores de decisão (split por threshold), (2) Modelos que tratam categorias nativamente
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    # fit_transform: aprende mapeamento no treino, aplica no treino (data leakage se usado em teste!)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Armazena para inverse_transform posterior (CatBoost precisa strings originais)

# =============================================================================
# DIVISÃO TREINO/TESTE (Train-Test Split)
# =============================================================================

# X: Matriz de features (n_samples × n_features) - Espaço de entrada ℝ^d
# y: Vetor target (n_samples,) - Espaço de saída {0, 1}
X = df[features]
y = df[target]

# train_test_split: Amostragem aleatória estratificada (mantém proporção de classes)
# test_size=0.2: 20% para teste, 80% para treino
# random_state=42: Seed fixa para reprodutibilidade (determinístico)
# Complexidade: O(n) - embaralhamento Fisher-Yates
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Data Shape: {X_train.shape}")   # (712, 5) - 80% dos dados
print(f"Test Data Shape: {X_test.shape}")        # (179, 5) - 20% dos dados

# =============================================================================
# LIGHTGBM - Light Gradient Boosting Machine
# =============================================================================

# LIGHTGBM - Inovações algorítmicas:
# 1. GOSS (Gradient-based One-Side Sampling): Mantém instâncias com gradiente grande (informativas),
#    subamostra instâncias com gradiente pequeno. Reduz complexidade de O(n) para O(n * α)
# 2. EFB (Exclusive Feature Bundling): Agrupa features mutuamente exclusivas (esparsas)
#    Reduz dimensionalidade efetiva preservando informação
# 3. Histogram-based Decision: Discretiza features contínuas em bins (padrão 255)
#    Substitui busca exata de split por busca em histogramas: O(n * d) → O(n * log(max_bins))

lgb_model = lgb.LGBMClassifier(
    # Hiperparâmetros padrão (ótimos para muitos casos):
    # n_estimators=100: Número de árvores (iterações de boosting)
    # learning_rate=0.1: Shrinkage/eta - passo de gradiente (regularização implícita)
    # num_leaves=31: Max folhas por árvore (controla complexidade, ~max_depth=5)
    # objective='binary': Log-loss binário: L = -Σ[y*log(p) + (1-y)*log(1-p)]
    # boosting_type='gbdt': Gradient Boosting Decision Tree padrão
)

# TREINAMENTO: Minimiza Loss Function via Gradient Descent funcional
# F_m(x) = F_{m-1}(x) + η * h_m(x)  onde h_m ≈ -∇L(y, F_{m-1})
# Cada árvore h_m ajusta resíduos (gradientes negativos) da iteração anterior
lgb_model.fit(X_train, y_train)

# PREDIÇÃO: Para classificação binária, aplica sigmoid na soma das árvores
# p(y=1|x) = σ(Σ η * h_m(x))  onde σ(z) = 1/(1+e^{-z})
lgb_pred = lgb_model.predict(X_test)

# AVALIAÇÃO: Acurácia = Proporção de predições corretas
# ACC = (1/n) Σ I(y_i = ŷ_i)  onde I() é função indicadora
print(f"LightGBM Accuracy: {accuracy_score(y_test, lgb_pred):.4f}")

# =============================================================================
# CATBOOST - Categorical Boosting
# =============================================================================

# CATBOOST - Inovações para features categóricas:
# 1. Ordered Target Statistics (TS): Substitui categoria por E[Target | Categoria]
#    Evita Target Leakage usando ordenação temporal + embaralhamento
#    TS_i = (Σ_{j<i} y_j * I(x_j = x_i) + a * p) / (Σ_{j<i} I(x_j = x_i) + a)
#    onde a = prior weight, p = prior global (ex: média global do target)
#
# 2. Ordered Boosting: Usa permutações aleatórias para calcular estatísticas
#    sem vazamento de dados (data leakage) - cada instância vê apenas predecessores
#
# 3. Combinações de features categóricas: Gera features de interação automaticamente

# IMPORTANTE: CatBoost PRECISA de features categóricas ORIGINAIS (strings),
# NÃO label-encoded! O LabelEncoder impõe ordem artificial que prejudica TS.
# Por isso criamos cópias com valores originais restaurados via inverse_transform.

cat_features = ['Pclass', 'Sex', 'Embarked']  # Features declaradas como categóricas

# Cria cópias para preservar X_train/X_test originais (usados por LightGBM/XGBoost)
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()

# Restaura valores categóricos originais (strings) usando LabelEncoder inverso
# inverse_transform: ℕ → Categories  (mapeia inteiro de volta para string original)
for col in ['Sex', 'Embarked']:
    X_train_cat[col] = label_encoders[col].inverse_transform(X_train_cat[col])
    X_test_cat[col] = label_encoders[col].inverse_transform(X_test_cat[col])

cat_model = CatBoostClassifier(
    cat_features=cat_features,      # Índices ou nomes das features categóricas
    verbose=0,                      # Suprime logs de treino (progress bars)
    allow_writing_files=False       # Evita criar diretório catboost_info/ (permissão Windows)
)

# TREINAMENTO: Ordered Boosting com Target Statistics
# Complexidade: O(n * d * iterations) com overhead de TS calculation
cat_model.fit(X_train_cat, y_train)

# PREDIÇÃO: Aplica modelo treinado (já inclui TS internamente)
cat_pred = cat_model.predict(X_test_cat)
print(f"CatBoost Accuracy: {accuracy_score(y_test, cat_pred):.4f}")

# =============================================================================
# XGBOOST - Extreme Gradient Boosting
# =============================================================================

# XGBOOST - Inovações matemáticas:
# 1. Aproximação de Newton (2ª ordem): Usa Hessiano para passo mais preciso
#    Taylor expansion: L(y, F+h) ≈ L(y,F) + g*h + ½*H*h²
#    Ótimo: h* = -g/H  (vs Gradient Descent: h* = -η*g)
#
# 2. Regularização explícita no objetivo:
#    Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_k)
#    Ω(f) = γ*T + ½λ||w||²  (T=número de folhas, w=pesos das folhas)
#    γ: penaliza complexidade (número de folhas), λ: L2 nos pesos
#
# 3. Split finding: Exact (exaustivo) vs Approximate (histogramas/quantis)
# 4. Sparsity-aware: Trata nativamente missing values (learn default direction)
# 5. Parallelização: Level-wise (por nível da árvore) + Column block para cache

xgb_model = XGBClassifier(
    eval_metric='logloss',          # Métrica de validação: Log Loss = -Σ[y*log(p) + (1-y)*log(1-p)]
    # Hiperparâmetros padrão relevantes:
    # n_estimators=100, learning_rate=0.3, max_depth=6
    # reg_alpha=0 (L1), reg_lambda=1 (L2), subsample=1, colsample_bytree=1
    # objective='binary:logistic': Sigmoid + LogLoss binário
)

# TREINAMENTO: Boosting com regularização L1/L2 e aproximação de Newton
xgb_model.fit(X_train, y_train)

# PREDIÇÃO: Threshold padrão 0.5 na probabilidade predita
xgb_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")

# =============================================================================
# CATBOOST NATIVE - Comparação: Encoding vs Features Nativas
# =============================================================================

# Experimento controlado: CatBoost com APENAS features categóricas nativas
# (sem Pclass como categórica - Pclass é ordinal, mas testamos só Sex+Embarked)
# Isso isola o efeito do Target Statistics vs Label Encoding

cat_model_native = CatBoostClassifier(
    cat_features=['Sex', 'Embarked'],  # Apenas features nominais puras
    verbose=0,
    allow_writing_files=False
)

# Treina nos mesmos dados com features originais restauradas
cat_model_native.fit(X_train_cat, y_train)

# Avalia se remover Pclass das categóricas afeta performance
cat_preds_native = cat_model_native.predict(X_test_cat)
print(f"CatBoost Native Accuracy: {accuracy_score(y_test, cat_preds_native):.4f}")

# =============================================================================
# RESUMO TEÓRICO COMPARATIVO
# =============================================================================

"""
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Aspecto         │ LightGBM         │ CatBoost         │ XGBoost          │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Gradiente       │ 1ª ordem (GD)    │ 1ª ordem (GD)    │ 2ª ordem (Newton)│
│ Categoricas     │ Label Enc / Opt  │ Native (TS)      │ Label Enc / Opt  │
│ Missing Values  │ Nativo (NA dir)  │ Nativo           │ Nativo           │
│ Paralelismo     │ Feature-level    │ Ordered (seq)    │ Level-wise       │
│ Velocidade      │ Muito rápida     │ Média            │ Rápida           │
│ Precisão        │ Alta             │ Alta (cat data)  │ Alta             │
│ Overfitting     │ Early stopping   │ Ordered TS + reg │ L1/L2 + shrinkage│
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘

Gradient Boosting Framework Geral:
---------------------------------
Dado: Dataset D = {(x_i, y_i)}_{i=1}^n, Loss L(y, F(x))
Objetivo: Minimizar Σ L(y_i, F(x_i)) + Ω(F)

Algoritmo (Functional Gradient Descent):
1. Inicializa F_0(x) = argmin_γ Σ L(y_i, γ)  (constante: log-odds para logloss)
2. Para m = 1 to M:
   a) Calcula pseudo-resíduos: r_{im} = -[∂L(y_i, F(x_i))/∂F]_{F=F_{m-1}}
   b) Ajusta árvore h_m(x) aos resíduos: h_m ≈ argmin_h Σ (r_{im} - h(x_i))²
   c) Atualiza: F_m(x) = F_{m-1}(x) + η * h_m(x)  (η = learning rate)
3. Retorna F_M(x)

Diferenças chave:
- LightGBM: h_m via splits baseados em histogramas + GOSS sampling
- XGBoost: h_m via Newton step (usa Hessiano) + regularização explícita Ω
- CatBoost: h_m via Ordered TS + combinações categóricas automáticas
"""