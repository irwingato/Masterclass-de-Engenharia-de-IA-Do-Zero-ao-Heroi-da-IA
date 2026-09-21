import numpy as np
import pandas as pd
import statsmodels.api as sm

# 1. Configurando uma semente para reprodutibilidade
np.random.seed(42)
n_amostras = 100

# 2. Configurando as variáveis independentes (X)
X1 = np.random.normal(50, 10, n_amostras)      # Ex: Idade
X2 = np.random.normal(10, 2, n_amostras)       # Ex: Anos de estudo
X3 = np.random.randint(0, 2, n_amostras)       # Ex: Categoria (0 ou 1)

X = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

# 3. Criando as múltiplas variáveis dependentes (Y) com um pouco de ruído aleatório
# Y1 e Y2 possuem comportamentos e coeficientes diferentes baseados nos mesmos Xs
Y1 = 5 + 0.5*X1 + 1.2*X2 + 3*X3 + np.random.normal(0, 2, n_amostras)
Y2 = 10 + 0.2*X1 - 0.5*X2 + 5*X3 + np.random.normal(0, 1, n_amostras)

Y = pd.DataFrame({'Y1': Y1, 'Y2': Y2})

# 4. Adicionando a constante (intercepto) exigida pelo statsmodels
X_com_constante = sm.add_constant(X)

# 5. Ajustando o modelo de regressão multivariada (MOLS)
# O statsmodels aceita matrizes bidimensionais diretamente no argumento de resposta
modelo = sm.OLS(Y, X_com_constante)
resultado = modelo.fit()

# 6. Exibindo os coeficientes estimados para ambos os alvos (Y1 e Y2)
print("--- Coeficientes Estimados ---")
coeficientes = resultado.params
coeficientes.columns = ['Y1', 'Y2']
print(coeficientes)