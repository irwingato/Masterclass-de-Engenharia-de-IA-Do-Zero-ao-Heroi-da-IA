import matplotlib.pyplot as plt  # Plotagem básica (histogramas e organização da figura)
import numpy as np               # Geração de dados numéricos e aleatórios
import pandas as pd              # Estruturação dos dados em DataFrame (tabela)
import seaborn as sns            # Gráficos estatísticos (violino, box plot, KDE)

# ---------------------------------------------------------------------------
# 1. Gerar os dados e estruturar o DataFrame
# ---------------------------------------------------------------------------

# Semente fixa = dados sempre idênticos a cada execução (reprodutibilidade).
np.random.seed(42)

# Dicionário com 3 arrays de valores normais (500 valores cada):
#   loc = média, scale = desvio padrão, size = quantidade.
dados = {
    'Grupo A': np.random.normal(loc=50, scale=10, size=500),
    'Grupo B': np.random.normal(loc=65, scale=12, size=500),
    'Grupo C': np.random.normal(loc=40, scale=8, size=500)
}

# pd.DataFrame(dados) transforma o dicionário em uma tabela (1 coluna por grupo).
# .melt(var_name='Categoria', value_name='Valores') converte para o formato
# "long-form" = 2 colunas:
#   - Categoria: repete o nome do grupo para cada valor
#   - Valores:   todos os números empilhados
# Esse é o "jeito certo" de passar dados para o Seaborn (um eixo = categorias,
# outro eixo = valores numéricos).
df = pd.DataFrame(dados).melt(var_name='Categoria', value_name='Valores')

# ---------------------------------------------------------------------------
# 2. Criar a grade de subplots (2 linhas e 2 colunas)
# ---------------------------------------------------------------------------

# plt.subplots() cria uma FIGURA com vários subgráficos em grade.
#   - nrows=2, ncols=2 -> 2 linhas x 2 colunas = 4 quadrantes (índices [0,0], [0,1], [1,0], [1,1])
#   - figsize=(14, 10) -> largura 14 e altura 10 polegadas para a janela inteira
# Retorna:
#   - fig  -> o objeto da figura inteira (permite títulos globais)
#   - axes -> a matriz 2x2 de eixos (cada posição é um gráfico independente)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Configuração global de estilo do Seaborn:
# style="whitegrid" aplica fundo branco com linhas de grade, padrão visual "limpo".
sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# Quadrante [0, 0] - Superior Esquerdo: Histograma Sobreposto (Matplotlib)
# --------------------------------------------------------------------------

# axes[0, 0].hist() chama o hist() DENTRO do primeiro subplot.
# 3 chamadas no mesmo eixo = histogramas sobrepostos.
# Os parâmetros são os mesmos dos exercícios anteriores (alpha=transparência,
# bins=nº de barras, label=legenda, color=cor, edgecolor=contorno).
axes[0, 0].hist(dados['Grupo A'], bins=25, alpha=0.5, label='Grupo A', color='#1f77b4', edgecolor='black')
axes[0, 0].hist(dados['Grupo B'], bins=25, alpha=0.5, label='Grupo B', color='#ff7f0e', edgecolor='black')
axes[0, 0].hist(dados['Grupo C'], bins=25, alpha=0.5, label='Grupo C', color='#2ca02c', edgecolor='black')

# Título e legenda específicos deste quadrante.
axes[0, 0].set_title('1. Histograma Sobreposto')
axes[0, 0].legend()

# --------------------------------------------------------------------------
# Quadrante [0, 1] - Superior Direito: Gráfico de Violino (Seaborn)
# --------------------------------------------------------------------------

# sns.violinplot() desenha um GRÁFICO DE VIOLINO: mostra a distribuição completa
# dos dados (ele é "grosso" onde há mais dados e "fino" onde há menos).
#   - data=df      -> DataFrame no formato long-form
#   - x='Categoria' -> eixo X = os 3 grupos (categorias)
#   - y='Valores'   -> eixo Y = os valores numéricos
#   - hue='Categoria' -> cores DISTINTAS para cada categoria
#   - palette='Set2'  -> paleta de cores suaves
#   - ax=axes[0, 1]   -> desenha NESTE quadrante específico
sns.violinplot(data=df, x='Categoria', y='Valores', hue='Categoria', palette='Set2', ax=axes[0, 1])
axes[0, 1].set_title('2. Gráfico de Violino')

# --------------------------------------------------------------------------
# Quadrante [1, 0] - Inferior Esquerdo: Box Plot Tradicional (Seaborn)
# --------------------------------------------------------------------------

# sns.boxplot() desenha um BOX PLOT (diagrama de caixa): resumo estatístico =
# mediana (linha no meio), quartis (caixa), e outliers (pontos fora).
# Parâmetros idênticos ao violinplot, mudando apenas o tipo de gráfico.
sns.boxplot(data=df, x='Categoria', y='Valores', hue='Categoria', palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title('3. Box Plot Tradicional')

# --------------------------------------------------------------------------
# Quadrante [1, 1] - Inferior Direito: Curva de Densidade KDE (Seaborn)
# --------------------------------------------------------------------------

# sns.kdeplot() desenha a CURVA DE DENSIDADE (KDE): uma versão "suavizada" do
# histograma — estima a probabilidade de cada valor.
#   - hue='Categoria' -> uma curva para cada grupo, com cor diferente
#   - fill=True       -> preenche a área abaixo da curva
#   - alpha=0.4       -> transparência de 40% (curvas sobrepostas ficam legíveis)
sns.kdeplot(data=df, x='Valores', hue='Categoria', fill=True, palette='Set2', alpha=0.4, ax=axes[1, 1])
axes[1, 1].set_title('4. Curva de Densidade (KDE)')

# --------------------------------------------------------------------------
# Ajustes Finais da Figura
# --------------------------------------------------------------------------

# plt.suptitle() define um TÍTULO PRINCIPAL para a FIGURA INTEIRA (acima dos 4
# subplots). y=0.98 posiciona o título perto do topo.
plt.suptitle('Análise Comparativa de Distribuições', fontsize=16, fontweight='bold', y=0.98)

# Ajusta o espaçamento entre os 4 quadrantes para títulos e rótulos não
# se sobreporem.
plt.tight_layout()

# Abre a janela exibindo a figura inteira com os 4 gráficos.
plt.show()