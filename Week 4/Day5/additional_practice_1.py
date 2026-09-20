"""additional_practice_1.py — ANOVA de DUAS vias (fatorial 3x2) com interação.

OBJETIVO: testar efeitos de Dieta e Exercício sobre a perda de peso,
incluindo se o efeito de um fator DEPENDE do outro (interação).

MATEMÁTICA — modelo fatorial de dois fatores:
    Y_ijk = mu + alfa_i + beta_j + (alfa*beta)_ij + epsilon_ijk,
    epsilon_ijk ~ N(0, sigma^2) independentes,
  onde alfa_i = efeito da Dieta i (i = 3 níveis), beta_j = efeito do
  Exercício j (j = 2 níveis), (alfa*beta)_ij = interação.

- Decomposição: SST = SS_A + SS_B + SS_AB + SS_erro.
- Três testes F (Tipo II, design balanceado n = 10/célula, N = 60):
    F_A  = MSA / MSE  (gl 2, 54) — H0: sem efeito de Dieta,
    F_B  = MSB / MSE  (gl 1, 54) — H0: sem efeito de Exercício,
    F_AB = MSAB / MSE (gl 2, 54) — H0: sem interação.
- Interação significativa => o efeito da Dieta muda conforme o Exercício
  (ex.: 'Intermitente + Cardio' com média 6.5 injetada propositalmente).
- Pressupostos: independência, normalidade dos resíduos, homocedasticidade.
"""
# pandas: monta o DataFrame; numpy: gera dados sintéticos; statsmodels: OLS+ANOVA.
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------------------------------------------------------------------------
# 1) Dataset fictício BALANCEADO: 3 dietas x 2 exercícios x 10 repetições = 60.
# ---------------------------------------------------------------------------
np.random.seed(42)  # semente => resultados reproduzíveis
n_por_grupo = 10    # n igual por célula => design balanceado (Tipo II ideal)

# Fator A (Dieta): cada nível repetido 2*n vezes (uma para cada Exercício).
dieta = np.repeat(['Low_Carb', 'Intermitente', 'Controle'], n_por_grupo * 2)
# Fator B (Exercício): padrão [Cardio x10, Musculacao x10] ladrilhado 3 vezes.
exercicio = np.tile(np.repeat(['Cardio', 'Musculacao'], n_por_grupo), 3)

# Variável resposta Y ~ Normal(media_celula, 1). Médias injetadas:
# Low_Carb: 5.0/4.0 | Intermitente: 6.5/3.0 (gap grande => interação forte) |
# Controle: 4.0/2.2. Desvio 1.0 = ruído intra-célula (sigma do modelo).
perda_peso = (
    np.random.normal(5.0, 1.0, n_por_grupo).tolist() +  np.random.normal(4.0, 1.0, n_por_grupo).tolist() + # Low_Carb
    np.random.normal(6.5, 1.0, n_por_grupo).tolist() +  np.random.normal(3.0, 1.0, n_por_grupo).tolist() + # Intermitente (Cardio funciona melhor)
    np.random.normal(4.0, 1.0, n_por_grupo).tolist() +  np.random.normal(2.2, 1.0, n_por_grupo).tolist() # Controle
)

# Junta fatores + resposta num DataFrame (uma linha = um participante).
df = pd.DataFrame({'Dieta': dieta, 'Exercicio': exercicio, 'Perda_Peso': perda_peso})

# ---------------------------------------------------------------------------
# 2) Modelo OLS com interação: 'C(A) * C(B)' expande para A + B + A:B.
#    C() = trata a coluna como categórica (dummies). .fit() = MQO.
# ---------------------------------------------------------------------------
# CORREÇÃO APLICADA: nomes idênticos aos das colunas ('Perda_Peso',
# 'Exercicio' sem acento) — antes falhava no Patsy por divergência de nomes.
modelo = ols('Perda_Peso ~ C(Dieta) * C(Exercicio)', data=df).fit()
# anova_lm typ=2: somas de quadrados Tipo II (correto p/ balanceado; cada
# efeito ajustado pelo outro efeito principal, sem a interação).
tabela_anova = sm.stats.anova_lm(modelo, typ=2) # Tipo 2 é o padrão para designs balanceados

# ---------------------------------------------------------------------------
# 3) Saída: colunas sum_sq (SS), df (gl), F, PR(>F) (valor-p).
#    Esperado: F_B grande, F_AB significativo (p < 0.05) pela interação injetada.
# ---------------------------------------------------------------------------
print(tabela_anova)
