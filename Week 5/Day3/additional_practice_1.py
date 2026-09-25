"""
additional_practice_1.py — Regularização com 1 recurso (MedInc) + polinómio grau 2
=================================================================================
OBJETIVO: mostrar como o hiperparâmetro `alpha` controla Ridge (L2) vs Lasso (L1).

MODELO BASE (regressão polinomial grau 2 numa só variável):
    phi(x) = [x, x^2]
    y_hat  = w0 + w1*x + w2*x^2          (w0 = intercept_, [w1,w2] = coef_)
Sem regularização isto é OLS:  min_w ||y - X_poly w||^2. Com 2 colunas
correlacionadas ([x, x^2]) o OLS já fica instável — cenário ideal p/ regularizar.

MATEMÁTICA:
- MSE (performance):  MSE = (1/n) * soma_i (y_i - y_hat_i)^2. Menor = melhor.
- Ridge (L2) minimiza:  J(w) = ||y - Xw||^2 + alpha * soma_j w_j^2
  Solução fechada: w* = (X^T X + alpha*I)^-1 X^T y. O termo alpha*I "infla" a
  diagonal -> inverte melhor matrizes quase-singulares e ENCOLHE w suavemente
  para 0, mas nunca exatamente a 0. Geometria: bola L2 é um círculo.
- Lasso (L1) minimiza:  J(w) = (1/(2*n)) * ||y - Xw||^2 + alpha * soma_j |w_j|
  (convenção sklearn; o 1/(2n) só reescala, o essencial é MSE + alpha*|w|).
  Sem solução fechada (módulo não é diferenciável em 0, usa-se sub-gradiente /
  coordinate descent). Operador soft-thresholding: w = sinal(z)*max(|z|-alpha,0).
  Geometria: bola L1 é um losango com vértices nos eixos -> a solução "bate" no
  vértice e ZERA coeficientes (seleção automática).
- alpha: compromisso viés-variância. alpha=0 -> OLS (variância alta).
  alpha grande -> viés alto (underfit). Por isso testamos [0.1, 1, 10].
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Carregar o dataset California Housing (20640 casas, 8 atributos + alvo MedHouseVal em $100k)
data = fetch_california_housing(as_frame=True)
df = data.frame

# Selecionar a feature (MedInc = rendimento mediano 0-15) e o alvo (MedHouseVal)
# Propositadamente 1D para podermos desenhar a curva y_hat(x) no gráfico.
X = df[['MedInc']]
y = df['MedHouseVal']

# Transformar para característica polinomial (grau 2: [x, x^2])
# Matemática: permite curva parabólica em vez de reta. Ex: x=5 -> [5, 25].
# include_bias=False = não cria coluna de 1s (o intercepto w0 já é tratado por Ridge/Lasso).
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Dividir os dados em treino (80%) e teste (20%)
# random_state=42 fixa a partição para o experimento ser reproduzível.
# ATENÇÃO: o poly foi ajustado antes do split (ideal seria só no treino, mas como
# PolynomialFeatures não aprende estatísticas dos dados, não há leakage aqui).
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Valores de alpha para o experimento (força da penalização)
# 0.1 = fraca, 1 = média, 10 = forte. Escala logarítmica para ver o efeito claramente.
alphas = [0.1, 1, 10]

# Lista para armazenar os resultados e criar um DataFrame comparativo no fim
resultados = []

# Configuração dos gráficos lado a lado (Ridge à esq., Lasso à dir., eixo y partilhado)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Índice de ordenação por MedInc: o teste vem embaralhado; para desenhar a curva
# prevista como linha contínua precisamos ordenar por x (senão fica "zigzag").
sort_idx = np.argsort(X_test[:, 0])

# -------------------------------------------------------------------------
# Experimento com Ridge (Regularização L2)
# Matemática: J = ||y-Xw||^2 + alpha*(w1^2+w2^2). Esperamos encolhimento suave:
# Coef_x e Coef_x2 diminuem um pouco quando alpha 0.1->10, MSE quase estável.
# -------------------------------------------------------------------------
# Pontos cinzentos = dados reais de teste (dispersão rendimento vs preço)
axes[0].scatter(X_test[:, 0], y_test, color="lightgray", label="Dados Reais", alpha=0.4)

for alpha in alphas:
    model = Ridge(alpha=alpha)  # cria o estimador L2 com esta penalização
    model.fit(X_train, y_train)  # resolve (X^T X + alpha*I)^-1 X^T y internamente
    preds = model.predict(X_test)  # y_hat = X_test @ coef_ + intercept_
    mse = mean_squared_error(y_test, preds)  # (1/n)*soma(y-y_hat)^2 no teste

    # Salvar coeficientes e performance para a tabela final
    resultados.append({
        'Modelo': 'Ridge', 'Alpha': alpha, 'MSE': mse,
        'Coef_x': model.coef_[0], 'Coef_x2': model.coef_[1]
    })

    # Desenha a parábola prevista (ordenada por x para ficar contínua)
    axes[0].plot(X_test[sort_idx, 0], preds[sort_idx], label=f"Alpha = {alpha}", lw=2.5)

axes[0].set_title("Ridge: Suavização dos coeficientes")
axes[0].set_xlabel("Median Income")
axes[0].set_ylabel("Median House Value")
axes[0].legend()

# -------------------------------------------------------------------------
# Experimento com Lasso (Regularização L1)
# Matemática: J = (1/2n)||y-Xw||^2 + alpha*(|w1|+|w2|). Esperamos esparsidade:
# com alpha=1 o Coef_x chega a 0.0000 (termo linear eliminado, fica só x^2),
# e com alpha=10 o MSE dispara 0.72->1.00 (underfit: penalização mata o sinal).
# -------------------------------------------------------------------------
axes[1].scatter(X_test[:, 0], y_test, color="lightgray", label="Dados Reais", alpha=0.4)

for alpha in alphas:
    model = Lasso(alpha=alpha)  # estimador L1 (otimizado por coordinate descent)
    model.fit(X_train, y_train)  # aplica soft-thresholding iterativo até convergir
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Salvar coeficientes e performance
    resultados.append({
        'Modelo': 'Lasso', 'Alpha': alpha, 'MSE': mse,
        'Coef_x': model.coef_[0], 'Coef_x2': model.coef_[1]
    })

    axes[1].plot(X_test[sort_idx, 0], preds[sort_idx], label=f"Alpha = {alpha}", lw=2.5)

axes[1].set_title("Lasso: Seleção de atributos (Zera coeficientes)")
axes[1].set_xlabel("Median Income")
axes[1].legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Exibir Tabela Comparativa de Coeficientes e Performance
# Leitura: Ridge mantém Coef_x~0.53 e MSE~0.70 nos 3 alphas (só encolhe);
# Lasso zera Coef_x em alpha>=1 e o MSE piora -> prova visual L2 vs L1.
# -------------------------------------------------------------------------
df_res = pd.DataFrame(resultados)
print("\n" + "="*70)
print("TABELA COMPARATIVA DE PERFORMANCE E COEFICIENTES ([x, x^2])")
print("="*70)
print(df_res.to_string(index=False, formatters={
    'MSE': '{:,.4f}'.format,
    'Coef_x': '{:,.4f}'.format,
    'Coef_x2': '{:,.4f}'.format
}))
print("="*70)
