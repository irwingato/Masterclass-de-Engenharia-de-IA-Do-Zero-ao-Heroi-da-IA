"""
additional_practice_2.py — Múltiplos recursos + padronização + polinómio grau 2
===============================================================================
OBJETIVO: mostrar o impacto da regularização quando há VÁRIOS recursos com
escalas diferentes e termos correlacionados (quadráticos + interações).

RECURSOS USADOS (3):
  MedInc   = rendimento mediano (~0-15)
  HouseAge = idade média da casa (~1-52 anos)
  AveRooms = média de quartos (~0-20)
Alvo: MedHouseVal (preço mediano em $100k).

MATEMÁTICA:
1) Padronização (StandardScaler):  z = (x - mu) / sigma  -> média 0, desvio 1.
   ESSENCIAL antes de Ridge/Lasso porque a penalização soma w_j^2 ou |w_j|
   tratando todos os w por igual. Sem escalar, HouseAge (até 52) precisaria de
   um w minúsculo e MedInc de um w grande; a penalização puniria injustamente
   o w grande mesmo que a variável fosse importante.

2) Expansão polinomial grau 2 em 3 variáveis:
   Nº de colunas = C(n+d, d) - 1 = C(3+2,2) - 1 = 10 - 1 = 9:
     [MedInc, HouseAge, AveRooms,           <- 3 lineares
      MedInc^2, MedInc*HouseAge, MedInc*AveRooms,  <- 3 quadráticos/interações
      HouseAge^2, HouseAge*AveRooms, AveRooms^2]   <- 3 restantes
   Estes 9 termos são fortemente correlacionados (multicolinearidade) -> OLS
   sobreajusta; regularização estabiliza.

3) Ridge vs Lasso (ver practice_1 para as fórmulas):
   Ridge:  J = ||y-Xw||^2 + alpha*||w||2^2 -> encolhe, nunca zera.
   Lasso:  J = (1/2n)||y-Xw||^2 + alpha*||w||1 -> zera (esparsidade).
   Contamos "Coefs_Zerados" com tolerância 1e-5 (|w|<1e-5 conta como zero
   porque o otimizador raramente devolve exatamente 0.0 em float).

4) Resultado esperado (confirmado na execução):
   Ridge MSE ~0.642 nos 3 alphas, 0 zeros (robusto).
   Lasso MSE 0.68 (6 zeros) -> 1.26 (8 zeros) -> 1.31 (9 zeros = só intercepto,
   modelo colapsa para prever a média). Prova de que L1 excessivo destrói o modelo.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset California Housing (as_frame=True devolve DataFrame pandas)
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Selecionar múltiplos recursos e o alvo
# MedInc: Renda média | HouseAge: Idade da casa | AveRooms: Média de quartos
# Passar de 1 para 3 recursos permite capturar efeitos que o rendimento sozinho não explica.
features = ['MedInc', 'HouseAge', 'AveRooms']
X = df[features]
y = df['MedHouseVal']

# Nota importante: Como as variáveis têm escalas muito diferentes,
# precisamos padronizá-las para que a regularização funcione corretamente.
# Matemática: z_ij = (x_ij - mu_j) / sigma_j. Fit no conjunto todo aqui por
# simplicidade didática (rigoroso seria fit só no treino + transform no teste).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Transformar para características polinomiais (grau 2)
# Cria termos quadráticos e interações (ex: MedInc * AveRooms)
# Matemática: permite superfícies curvas e efeitos conjuntos, ex. "rendimento alto
# COM casa nova vale mais que a soma isolada". São 9 colunas (ver docstring).
# include_bias=False evita coluna de 1s (o intercepto já existe nos modelos).
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# 4. Dividir os dados em treino (80%) e teste (20%)
# random_state=42 = partição reproduzível. O teste mede generalização (MSE fora da amostra).
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Valores de alpha para o experimento (0.1 fraco, 1 médio, 10 forte)
alphas = [0.1, 1, 10]
resultados = []

# -------------------------------------------------------------------------
# Executar Experimentos (Ridge e Lasso)
# Para cada alpha treinamos os 2 modelos e medimos MSE + nº de coeficientes zerados.
# -------------------------------------------------------------------------
for alpha in alphas:
    # Treinar Ridge: resolve (X^T X + alpha*I)^-1 X^T y. Esperamos 0 zeros sempre.
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)  # y_hat = X_test @ w + b
    ridge_mse = mean_squared_error(y_test, ridge_preds)  # MSE = média(y-y_hat)^2
    # Contar quantos coeficientes são EXATAMENTE zero (tolerância 1e-5 p/ erro float)
    ridge_zeros = np.sum(np.isclose(ridge.coef_, 0, atol=1e-5))

    resultados.append({
        'Modelo': 'Ridge', 'Alpha': alpha, 'MSE': ridge_mse, "Coefs_Zerados": ridge_zeros
    })

    # Treinar Lasso: coordinate descent com soft-thresholding. Esperamos zeros a crescer
    # com alpha (6 -> 8 -> 9) e MSE a degradar quando a esparsidade é excessiva.
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_preds = lasso.predict(X_test)
    lasso_mse = mean_squared_error(y_test, lasso_preds)
    lasso_zeros = np.sum(np.isclose(lasso.coef_, 0, atol=1e-5))

    resultados.append({
        'Modelo': 'Lasso', 'Alpha': alpha, 'MSE': lasso_mse, 'Coefs_Zerados': lasso_zeros
    })

# Converter resultados para DataFrame (formato longo: 6 linhas = 3 alphas x 2 modelos)
df_res = pd.DataFrame(resultados)

# -------------------------------------------------------------------------
# Visualização da Performance (MSE)
# pivot: linhas=Alpha, colunas=Modelo, valores=MSE -> gráfico de barras agrupadas.
# Leitura esperada: barras Ridge baixas e planas (~0.64); barras Lasso a subir
# abruptamente (underfit por excesso de zeros). Quanto menor a barra, melhor.
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
pivot_df = df_res.pivot(index='Alpha', columns='Modelo', values='MSE')
pivot_df.plot(kind='bar', color=['orange', 'green'], ax=plt.gca(), edgecolor='black')
plt.title("Impacto do Alpha na Performance (MSE) com Múltiplos Recursos")
plt.ylabel("Mean Squared Errors (MSE) - Quanto menor, melhor")
plt.xlabel("Valor de Alpha")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Exibir tabela comparativa no console
# Total de Atributos: 9 (ver cálculo combinatório na docstring do topo)
print("\n" + "="*65)
print("  COMPARAÇÃO COM MÚLTIPLOS RECURSOS (Total de Atributos: 9)")
print("="*65)
print(df_res.to_string(index=False, formatters={'MSE': '{:,.4f}'.format}))
print("="*65)
