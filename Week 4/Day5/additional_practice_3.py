"""additional_practice_3.py — VISUALIZAÇÃO dos dados do teste ANOVA (ISO-3166).

OBJETIVO: inspecionar graficamente o que a ANOVA de duas vias do practice_2
testa (distribuição de `country-code` por `region`/`sub-region`) e o
balanceamento do design (contagens por categoria).

MATEMÁTICA DOS GRÁFICOS:
- Gráfico de barras de CONTAGEM: altura = n_k (tamanho amostral da categoria).
  Mostra desbalanceamento (África n=60 vs Oceania n=29) — justifica ANOVA Tipo II.
- BOXPLOT (caixa e bigodes) por grupo: resume a distribuição de Y condicional:
    Q1 (25%), mediana Q2 (50%), Q3 (75%); IQR = Q3 - Q1 (altura da caixa);
    bigodes até Q1 - 1.5*IQR e Q3 + 1.5*IQR; pontos além = outliers.
  A ANOVA compara justamente essas distribuições condicionais:
    F = MSB/MSW grande <=> caixas/medianas bem separadas frente à dispersão
    interna (caixas altas/largas). Boxplots sobrepostos => F pequeno, p alto.
- Barras de MÉDIA por região: altura = Y_barra_k = (1/n_k)*soma Y_ki.
  Diferenças entre alturas = numerador do F (SSB); a ANOVA formaliza se essas
  diferenças excedem o ruído (SSW).

GRÁFICOS GERADOS (5 PNGs ao lado do script):
 1. Barras: nº de países por `region`.
  2. Barras horizontais: nº de países por `sub-region`.
  3. Boxplot: `country-code` por `region` (efeito principal testado).
  4. Boxplot: `country-code` por `sub-region` (ordenado pela mediana).
  5. Barras: média de `country-code` por `region`.
"""
# pandas: dados; matplotlib (backend Agg = sem janela, só salva PNG); seaborn:
# estilos + countplot/boxplot; Path: caminho da pasta do script p/ salvar PNGs.
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend sem janela (permite rodar sem display)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT_DIR = Path(__file__).parent  # salva os PNGs ao lado do script

# ---------------------------------------------------------------------------
# 1) Carrega o mesmo CSV do practice_2 (colunas com hífen no nome original).
# ---------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
df = pd.read_csv(url)

# ---------------------------------------------------------------------------
# 2) Limpeza idêntica à do practice_2:
#    dropna remove NaN; rename evita hífen na sintaxe; filtro != '' remove
#    resíduos vazios (Antarctica/Taiwan). Sem isso, categorias fantasmas '' e
#    NaN poluíram contagens e boxplots.
# ---------------------------------------------------------------------------
df = df.dropna(subset=['region', 'sub-region', 'country-code'])
df = df.rename(columns={'sub-region': 'sub_region', 'country-code': 'country_code'})
# Remove strings vazias que restam (ex.: Antarctica / Taiwan)
df = df[(df['region'] != '') & (df['sub_region'] != '')]

# Sanity checks numéricos: N total e top-10 + tabela de contingência 1-D.
print(f"Total de países analisados: {len(df)}")
print(df[['name', 'region', 'sub_region', 'country_code']].head(10).to_string(index=False))
print("\nContagem por região:")  # n_k por região => avalia balanceamento
print(df['region'].value_counts())

sns.set_theme(style="whitegrid")  # grade clara: facilita ler alturas/medianas

# ---------------------------------------------------------------------------
# 3) Gráfico 1 — BARRAS de contagem por região (n_k).
#    countplot conta ocorrências de cada categoria em x. Ordem decrescente.
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
ordem_regiao = df['region'].value_counts().index  # categorias do maior p/ menor n
sns.countplot(data=df, x='region', order=ordem_regiao, color='steelblue', edgecolor='black')
plt.title('Nº de países por Região (ISO-3166)')
plt.xlabel('Região')
plt.ylabel('Nº de países')  # altura da barra = n_k
plt.xticks(rotation=15)
plt.tight_layout()  # evita corte de rótulos
plt.savefig(OUT_DIR / 'grafico_barras_por_regiao.png', dpi=150)
plt.close()  # trocado show() por close() para rodar headless

# ---------------------------------------------------------------------------
# 4) Gráfico 2 — BARRAS HORIZONTAIS por sub-região (muitas categorias => eixo y).
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 7))
ordem_sub = df['sub_region'].value_counts().index
sns.countplot(data=df, y='sub_region', order=ordem_sub, color='teal', edgecolor='black')
plt.title('Nº de países por Sub-região (ISO-3166)')
plt.xlabel('Nº de países')
plt.ylabel('Sub-região')
plt.tight_layout()
plt.savefig(OUT_DIR / 'grafico_barras_por_subregiao.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5) Gráfico 3 — BOXPLOT de Y | região (o coração visual da ANOVA).
#    Cada caixa = distribuição de country_code dentro de uma região.
#    Medianas alinhadas => F pequeno; medianas escalonadas => F grande.
#    hue='region' + legend=False: só colore sem duplicar legenda (evita warning).
# ---------------------------------------------------------------------------
# Visualiza o que a ANOVA de duas vias testa: a distribuição da variável
# numérica `country_code` difere entre regiões?
plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x='region', y='country_code', hue='region', palette='Set2', legend=False)
plt.title('Boxplot: country-code por Região\n(visualização do efeito testado na ANOVA)')
plt.xlabel('Região')
plt.ylabel('Country code (numérico ISO)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT_DIR / 'boxplot_countrycode_por_regiao.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6) Gráfico 4 — BOXPLOT de Y | sub-região, ORDENADO PELA MEDIANA.
#    groupby().median().sort_values() define a ordem das caixas (leitura fácil:
#    códigos baixos à esquerda, altos à direita). Rotação 45° evita sobreposição.
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
# Ordena sub-regiões pela mediana para facilitar a leitura
ordem_mediana = df.groupby('sub_region')['country_code'].median().sort_values().index
sns.boxplot(data=df, x='sub_region', y='country_code', order=ordem_mediana, hue='sub_region', palette='Set3', legend=False)
plt.title('Boxplot: country-code por Sub-região')
plt.xlabel('Sub-região')
plt.ylabel('Country code (numérico ISO)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUT_DIR / 'boxplot_countrycode_por_subregiao.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7) Gráfico 5 — BARRAS das médias Y_barra_k por região.
#    groupby().mean() = estimador de mu + alfa_k; sort_values ordena alturas.
#    É o "resumo de 1 número" do boxplot: compara centros, ignorando dispersão.
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
media_regiao = df.groupby('region')['country_code'].mean().sort_values()
media_regiao.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Média de country-code por Região')
plt.xlabel('Região')
plt.ylabel('Média de country-code')  # altura = Y_barra_k
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT_DIR / 'grafico_barras_media_countrycode.png', dpi=150)
plt.close()

print("\nGráficos exibidos e salvos como PNG no diretório atual.")
