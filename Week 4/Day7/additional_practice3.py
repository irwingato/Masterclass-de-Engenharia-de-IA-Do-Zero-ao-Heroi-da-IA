"""
additional_practice3.py — Regressão MÚLTIPLA em dados reais de VENDAS (Fortune 1000).
==================================================================================
DATASET: fortune1000.csv — 1000 maiores empresas EUA (~2016).
Colunas-chave: Company, Sector, Revenue (faturamento $M), Profits (lucro $M).
URL página: https://github.com/sivabalanb/Data-Analysis-with-Pandas-and-Python/blob/master/fortune1000.csv
URL raw  : https://raw.githubusercontent.com/sivabalanb/Data-Analysis-with-Pandas-and-Python/master/fortune1000.csv

OBJETIVO: prever lucro a partir de faturamento + ser ou não de Tecnologia.
TÉCNICAS: conversão blob->raw, dummy setorial, dropna, train_test_split,
  OLS múltiplo, R², interpretação econômica + scatterplot com hue.

MODELO:
  Profits = b0 + b1*Revenue + b2*is_tech + erro
  is_tech = 1 se Sector=='Technology', senão 0.
  b1 = margem marginal ($lucro por $venda extra); b2 = prêmio/desconto tech.
"""

import matplotlib.pyplot as plt  # figuras e limites de eixo
import numpy as np  # (tipos numéricos; quantile usa pandas por baixo)
import pandas as pd  # read_csv, dummies, dropna, quantile
import seaborn as sns  # scatterplot com cor por setor
from sklearn.linear_model import LinearRegression  # OLS
from sklearn.metrics import r2_score  # R² teste
from sklearn.model_selection import train_test_split  # split 80/20

# ---------------------------------------------------------------------------
# 1. URL blob -> raw (detalhe computacional de GitHub)
# A URL /blob/master/... é PÁGINA HTML, não CSV. pd.read_csv precisa do arquivo
# puro: raw.githubusercontent.com/.../master/.... Conversão:
#   "github.com" -> "raw.githubusercontent.com"  (troca o host)
#   "/blob/" -> "/"                              (remove segmento de UI)
# Erro clássico: trocar por "githubusercontent.com" com "://" duplicado gera
# "https://://..." inválida. A forma acima produz URL válida (HTTP 200).
# ---------------------------------------------------------------------------
url = "https://github.com/sivabalanb/Data-Analysis-with-Pandas-and-Python/blob/master/fortune1000.csv"

# Ajuste automático para converter a URL do site no link do arquivo de dados puro (raw)
raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

# Carregar o Banco de Dados Real
# Computacional: CSV pequeno (~1000 linhas). Revenue/Profits vêm como int
# (podem ter negativos = prejuízo, ex: -6126). Sector/Company como object.
df = pd.read_csv(raw_url)

# Inspeção rápida: imprime 5 empresas para conferir parse (vírgulas em
# "Bentonville, AR" estão entre aspas no CSV — parser lida com isso).
print("--- Primeiras Linhas do Dataset Fortune 1000 ---")
print(df[["Company", "Sector", "Revenue", "Profits"]].head(), "\n")

# ---------------------------------------------------------------------------
# 2. DUMMY SETORIAL is_tech (engenharia de recursos)
# Matemática: transforma categoria em número para a equação linear:
#   is_tech = 1_{Sector == 'Technology'}.
# É um "intercepto deslocado": não-tech prevê b0+b1*Rev; tech prevê
# (b0+b2)+b1*Rev — duas retas PARALELAS com gap b2. Generaliza para one-hot
# com k setores (k-1 dummies para evitar multicolinearidade perfeita).
# Computacional: comparação vetorizada (O(n)) + astype(int).
# ---------------------------------------------------------------------------
# Criamos uma variável dummy: 1 se for do setor de Tecnologia, 0 para outros setores
df["is_tech"] = (df["Sector"] == "Technology").astype(int)

# Limpeza: dropna nas 3 colunas do modelo. Se Revenue/Profits tiver NaN ou
# string suja, .astype(float) quebraria; dropna garante matriz numérica limpa.
df_clean = df.dropna(subset=["Revenue", "Profits", "is_tech"])

# ---------------------------------------------------------------------------
# 3. MATRIZ X / VETOR y
# X = [[Revenue_i, is_tech_i]] (n,2); y = [Profits_i] (n,).
# .astype(float) uniformiza int->float64 para o solver SVD do sklearn.
# ---------------------------------------------------------------------------
# Queremos prever o Lucro (Profits) usando as Vendas (Revenue) e o indicador de Tecnologia (is_tech)
X = df_clean[["Revenue", "is_tech"]].astype(float)
y = df_clean["Profits"].astype(float)

# ---------------------------------------------------------------------------
# 4. SPLIT 80/20 (mesma lógica do practice2)
# Treino ajusta b0,b1,b2; teste estima generalização. random_state=42 fixa
# a partição para reproduzir o R² entre execuções.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 5. OLS MÚLTIPLO: min ||y - Xb||² -> b = (X^T X)^-1 X^T y (via SVD).
# p=2 features, n~800 treino: barato O(n*p²). b1 esperado ~0.05-0.15
# (margem média); b2 pode ser positivo (tech mais lucrativa a igual receita).
# ---------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 6. MÉTRICAS: R² no teste = 1 - SSres/SStot.
# Interpretação econômica: R² alto = receita prevê bem o lucro; R² baixo =
# lucro depende de setor/custos não capturados. Em cross-section de empresas,
# outliers (Walmart gigante, prejuízos bilionários) puxam SSres e derrubam R².
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("--- Resultados da Regressão ---")
print(f"Intercepto (Lucro Base): ${model.intercept_:.2f} milhões")
print(f"R² (Poder de Explicação): {r2:.4f}\n")

print("Coeficientes por Variável:")
print(f"  - Faturamento (Revenue): {model.coef_[0]:.4f}")  # dLucro/dReceita
print(f"  - Setor de Tecnologia (is_tech): ${model.coef_[1]:.2f} milhões")  # gap tech

# ---------------------------------------------------------------------------
# 7. LEITURA DE NEGÓCIO (ceteris paribus)
# b1: "cada $1M extra de venda associa-se a +$b1M de lucro, setor fixo".
# b2: "a igual faturamento, tech lucra $b2M a mais/menos". Não é causal puro:
# tech pode ter custos menores ou poder de preço — o modelo não distingue.
# ---------------------------------------------------------------------------
print("\n--- Análise de Impacto Econômico ---")
print(
    f"• Vendas: Para cada $1 milhão a mais em faturamento, o lucro estimado sobe cerca de ${model.coef_[0]:.4f} milhões."
)
if model.coef_[1] > 0:
    print(
        f"• Efeito Setorial: Mantendo o mesmo faturamento, empresas de Tecnologia lucram, em média, **${model.coef_[1]:.2f} milhões a MAIS** do que as de outros setores."
    )
else:
    print(
        f"• Efeito Setorial: Mantendo o mesmo faturamento, empresas de Tecnologia lucram, em média, ${abs(model.coef_[1]):.2f} milhões a MENOS do que as de outros setores."
    )

# ---------------------------------------------------------------------------
# 8. SCATTER Revenue x Profits com cor = Sector
# Visual: cada ponto = empresa; x = escala, y = rentabilidade; hue = setor.
# hue revela heterogeneidade: nuvens por setor têm inclinação própria —
# indício de que um modelo com interação (Revenue*Sector) explicaria mais.
# Truque computacional: xlim(0, quantile 95%) e ylim(quantil 1%, 98%) cortam
# outliers (ex: Walmart 482k, prejuízos <-10k) que achatariam a nuvem.
# quantile(q) = inversa da CDF empírica: valor que deixa q% abaixo.
# bbox_to_anchor joga a legenda para fora; tight_layout evita corte.
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_clean,
    x="Revenue",
    y="Profits",
    hue="Sector",
    palette="tab20",
    alpha=0.7,  # transparência: resolve sobreposição de pontos
)

plt.title("Relação entre Faturamento (Vendas) e Lucros - Fortune 1000")
plt.xlabel("Faturamento/Vendas (Revenue em Milhões de $)")
plt.ylabel("Lucro Líquido (Profits em Milhões de $)")

# Limitamos os eixos para o gráfico ficar legível (removendo a distorção de outliers gigantes)
plt.xlim(0, df_clean["Revenue"].quantile(0.95))
plt.ylim(df_clean["Profits"].quantile(0.01), df_clean["Profits"].quantile(0.98))

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Setores")
plt.tight_layout()
plt.show()
