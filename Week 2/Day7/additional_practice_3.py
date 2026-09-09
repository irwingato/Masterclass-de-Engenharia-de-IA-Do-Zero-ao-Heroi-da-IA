# additional_practice_3.py
# ------------------------------------------------------------------
# Objetivo: Criar um DASHBOARD com as descobertas do EDA sobre o
# dataset Iris usando PLOTLY.
#
# Diferente de um servidor web (ex.: Dash), este script gera um único
# arquivo HTML standalone com todos os gráficos combinados em subplots.
# Basta abrir o HTML no navegador — nada de servidor em execução.
# ------------------------------------------------------------------

# Importa as bibliotecas utilizadas:
# os   -> manipular caminhos de arquivo (salvar o HTML junto ao script)
# pandas -> manipulação de dados
# plotly.express (px)  -> criação rápida de gráficos interativos
# plotly.graph_objects (go) -> (importado por conveniência; usado via px)
# make_subplots -> combinar vários gráficos em uma grade de subplots
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Carrega o dataset Iris a partir de uma URL (CSV público no GitHub)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Lista das colunas numéricas do dataset (usada no heatmap de correlação)
num_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Scatter interativo: comprimento da pétala vs comprimento da sépala
# color=coloriza os pontos por espécie; size=define o tamanho do marcador
scatter = px.scatter(
    df,
    x="sepal_length",
    y="petal_length",
    color="species",
    size="petal_width",
    title="Petal Length vs Sepal Length",
    template="plotly_white",
)
scatter.update_layout(height=400)  # Define a altura do gráfico

# Boxplot interativo: comprimento da pétala distribuído por espécie
box = px.box(
    df,
    x="species",
    y="petal_length",
    color="species",
    title="Petal Length por Espécie",
    template="plotly_white",
)
box.update_layout(height=400)

# Histograma interativo: distribuição do comprimento da sépala
# color=separa as espécies; marginal="box" adiciona um mini-boxplot ao lado
hist = px.histogram(
    df,
    x="sepal_length",
    color="species",
    marginal="box",
    title="Distribuição de Sepal Length",
    template="plotly_white",
)
hist.update_layout(height=400)

# Heatmap de correlação: corr() calcula a matriz de correlação das variáveis
# numéricas; text_auto exibe os valores, zmin/zmax fixam a escala de -1 a 1
corr = df[num_cols].corr()
heat = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlação entre Variáveis",
    template="plotly_white",
)
heat.update_layout(height=400)

# Combina todos os gráficos em um único dashboard 2x2 (subplots)
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=("Scatter", "Boxplot", "Histograma", "Correlação"),
    vertical_spacing=0.12,   # espaço entre linhas
    horizontal_spacing=0.08, # espaço entre colunas
)

# Adiciona cada gráfico na posição correspondente da grade
for trace in scatter.data:
    fig.add_trace(trace, row=1, col=1)  # scatter no canto superior esquerdo
for trace in box.data:
    fig.add_trace(trace, row=1, col=2)  # boxplot no canto superior direito
for trace in hist.data:
    fig.add_trace(trace, row=2, col=1)  # histograma no canto inferior esquerdo
for trace in heat.data:
    fig.add_trace(trace, row=2, col=2)  # heatmap no canto inferior direito

# Mantém a legenda apenas para os 3 traços do scatter (um por espécie).
# update_traces(showlegend=False) esconde todas; depois reativa só as primeiras.
fig.update_traces(showlegend=False)
for i in range(len(scatter.data)):
    fig.data[i].showlegend = True

# Ajustes gerais do dashboard (título, altura, tema)
fig.update_layout(
    title="Dashboard EDA - Iris Dataset",
    height=900,
    template="plotly_white",
    showlegend=True,
)

# Rótulos dos eixos de cada subplot para ficarem claros no dashboard
fig.update_xaxes(title_text="Sepal Length", row=1, col=1)
fig.update_yaxes(title_text="Petal Length", row=1, col=1)
fig.update_yaxes(title_text="Sepal Length", row=2, col=1)

# Saída 1: gera um HTML standalone (abre no navegador, sem servidor).
# O arquivo é salvo na mesma pasta deste script (__file__ = caminho do script).
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.write_html(os.path.join(out_dir, "iris_dashboard.html"))

# Saída 2 (opcional): exportar como imagem estática.
# Para isso, instale o kaleido (pip install kaleido) e descomente a linha abaixo.
# fig.write_image(os.path.join(out_dir, "iris_dashboard.png"))

print("Dashboard salvo em:", os.path.join(out_dir, "iris_dashboard.html"))

# Para exportar como PNG/PDF local, instale kaleido e descomente a linha acima.