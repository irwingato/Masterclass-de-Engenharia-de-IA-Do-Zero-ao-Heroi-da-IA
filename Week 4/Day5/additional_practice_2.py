"""additional_practice_2.py — ANOVA de duas vias com DADOS REAIS (ISO-3166).

OBJETIVO: aplicar o mesmo fatorial do practice_1 a países reais, testando se
`region` e `sub-region` explicam o `country-code` numérico ISO.
(gênero/classe citado antes era só exemplo de outro domínio.)

MATEMÁTICA — mesmo modelo Y = mu + alfa_i + beta_j + (alfa*beta)_ij + erro:
- H0_1: sem efeito de `region` (todas as médias regionais iguais).
- H0_2: sem efeito de `sub_region`.
- H0_3: sem interação region:sub_region.
- F = MS_efeito / MS_resíduo; p = P(F >= F_obs | H0).
- ATENÇÃO DIDÁTICA: `country-code` é um identificador arbitrário (não uma
  medida contínua com sentido causal) e `sub_region` é ANINHADA em `region`
  (cada sub-região pertence a uma só região) — o design não é fatorial
  cruzado completo, daí o aviso SingularMatrixWarning (matriz rank-deficiente)
  e gl residuais reduzidos. Vale como exercício de código/hipóteses, não como
  inferência geográfica substantiva.
- Tipo II: adequado ao desbalanceamento (nº de países varia por região).
"""
# pandas: leitura/limpeza; statsmodels: OLS + tabela ANOVA via fórmula.
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------------------------------------------------------------------------
# 1) Carrega CSV real (ISO-3166 com códigos regionais, ~249 linhas).
#    Colunas usadas: 'region', 'sub-region', 'country-code' (+ 'name' p/ exibir).
# ---------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
df = pd.read_csv(url)  # leitura direta da web para DataFrame

# ---------------------------------------------------------------------------
# 2) Limpeza:
#    a) '' -> NA: o CSV traz strings vazias (Antarctica, Taiwan) que dropna()
#       sozinho NÃO remove (só remove NaN). replace() resolve.
#    b) dropna nas 3 colunas de análise.
#    c) rename: hífen quebra a sintaxe de fórmula do Patsy ('a-b' vira subtração),
#       então 'sub-region' -> 'sub_region' etc.
#    d) to_numeric: '004' (texto) -> 4 (int); errors='coerce' vira NaN se falhar,
#       seguido de novo dropna para garantir Y numérico.
# ---------------------------------------------------------------------------
# CORREÇÃO: dropna() sozinho não remove strings vazias ('') presentes no CSV
# (ex.: Antarctica e Taiwan ficam com region='' em vez de NaN).
# Convertemos '' -> NA antes do dropna e garantimos country_code numérico.
df = df.replace('', pd.NA)
# Removemos registros com valores ausentes nas colunas que vamos analisar
df = df.dropna(subset=['region', 'sub-region', 'country-code'])

# Formando os nomes das colunas para evitar problemas com hifens na fórmula
df = df.rename(columns={'sub-region': 'sub_region', 'country-code': 'country_code'})

# Garante tipo numérico (ex.: '004' -> 4) e remove eventuais falhas de conversão
df['country_code'] = pd.to_numeric(df['country_code'], errors='coerce')
df = df.dropna(subset=['country_code'])

# Mostra 5 primeiras linhas (sanity check: nomes, região, código).
print("Amostra dos dados reais utilizados:")
print(df[['name', 'region', 'sub_region', 'country_code']].head())

# ---------------------------------------------------------------------------
# 3) Modelo: 'C(region) * C(sub_region)' = efeitos principais + interação.
#    Equivale a Y = mu + alfa_region + beta_sub + (alfa*beta) + erro.
# ---------------------------------------------------------------------------
# 3. Modelando a ANOVA de duas vias com efeito de interação (*)
# HO_1: A região geográfica não impacta o código numérico do país.
# H0_2: A sub-região não impacta o código numérico do país.
# H0_3: Não há efeito de interação entre região e sub-região.
formula = 'country_code ~ C(region) * C(sub_region)'
modelo = ols(formula, data=df).fit()  # MQO: estima mu, alfas, betas e interações

# Tabela ANOVA Tipo II (cada efeito principal ajustado pelo outro).
# Executando a análise (Tipo 2 devido ao desbalanceamento no número de países por região)
# Nota: o aviso SingularMatrixWarning é esperado aqui — sub_region é aninhada
# dentro de region (cada sub-região pertence a uma só região), logo o design
# não é fatorial cruzado completo. O teste segue válido como exemplo didático
# com dados reais (gênero/classe era só um exemplo de outro domínio).
tabela_anova = sm.stats.anova_lm(modelo, typ=2)

# Saída típica: C(sub_region) p << 0.05; C(region) e interação n.s. — coerente
# com códigos ISO alocados por blocos de sub-região, não por região ampla.
print("TABELA ANOVA DE DUAS VIAS:")
print(tabela_anova)  # colunas: sum_sq, df, F, PR(>F)
