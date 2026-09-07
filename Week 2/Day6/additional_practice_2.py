import matplotlib.pyplot as plt  # Importa o módulo de plotagem do Matplotlib (funções de gráfico)
import numpy as np               # Importa o NumPy (geração de dados numéricos e aleatórios)

# ---------------------------------------------------------------------------
# 1. Gerar três conjuntos de dados (Distribuições Normais)
# ---------------------------------------------------------------------------

# Fixa a "semente" do gerador de números aleatórios: os dados gerados serão
# sempre os mesmos a cada execução (reprodutibilidade dos resultados).
np.random.seed(42)

# np.random.normal(loc=..., scale=..., size=...) gera uma distribuição normal:
#   - loc   = média (posição do "pico" da curva)
#   - scale = desvio padrão (dispersão/largura da curva)
#   - size  = quantidade de valores gerados
# Aqui usamos 1000 valores por grupo (amostras maiores = histograma mais suave).
grupo_A = np.random.normal(loc=50, scale=10, size=1000)  # Média 50, Desvio Padrão 10
grupo_B = np.random.normal(loc=65, scale=12, size=1000)  # Média 65, Desvio Padrão 12
grupo_C = np.random.normal(loc=40, scale=8, size=1000)   # Média 40, Desvio Padrão 8

# ---------------------------------------------------------------------------
# 2. Configurar a figura do gráfico
# ---------------------------------------------------------------------------

# plt.figure() cria uma nova janela/figura para o gráfico.
# figsize=(10, 6) define a largura (10 polegadas) e altura (6 polegadas).
# Gráfico mais largo/horizontal facilita a leitura de histogramas sobrepostos.
plt.figure(figsize=(10, 6))

# ---------------------------------------------------------------------------
# 3. Plotar os histogramas sobrepostos usando o parâmetro alpha (transparência)
# ---------------------------------------------------------------------------

# plt.hist() desenha um HISTOGRAMA (conta quantos valores caem em cada intervalo).
# Chamados 3 vezes no MESMO gráfico -> histogramas SOBREPOSTOS.
#   - bins=30   = 30 barras/intervalos no eixo X
#   - alpha=0.5 = transparência de 50%. ESSE é o segredo da sobreposição:
#                 onde dois histogramas se cruzam, as cores se misturam e
#                 ficam visíveis ao mesmo tempo.
#   - label=    = texto usado na legenda
#   - color=    = cor das barras em hexadecimal
#   - edgecolor= = contorno das barras (preto destaca os limites de cada uma)
plt.hist(grupo_A, bins=30, alpha=0.5, label='Grupo A (Média=50)', color='#1f77b4', edgecolor='black')
plt.hist(grupo_B, bins=30, alpha=0.5, label='Grupo B (Média=65)', color='#ff7f0e', edgecolor='black')
plt.hist(grupo_C, bins=30, alpha=0.5, label='Grupo C (Média=40)', color='#2ca02c', edgecolor='black')

# ---------------------------------------------------------------------------
# 4. Customizar títulos, legendas e eixos
# ---------------------------------------------------------------------------

# Título principal: fontsize=tamanho da fonte, fontweight='bold'=negrito,
# pad=15 = espaço (em pixels) entre o título e o topo do gráfico.
plt.title('Histograma com Múltiplos Conjuntos de Dados Sobrepostos', fontsize=14, fontweight='bold', pad=15)

# Rótulos dos eixos: descrevem o que cada eixo representa.
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)

# Ativa a legenda usando os 'label=' definidos nos histogramas.
# loc='upper right' = canto superior direito.
plt.legend(loc='upper right', fontsize=11)

# Grade tracejada e semitransparente para facilitar a leitura dos valores.
plt.grid(True, linestyle='--', alpha=0.5)

# ---------------------------------------------------------------------------
# 5. Renderizar o gráfico de forma organizada
# ---------------------------------------------------------------------------

# Ajusta automaticamente o espaçamento para nada (títulos, legendas, rótulos)
# ficar cortado ou sobreposto.
plt.tight_layout()

# Exibe a janela com o gráfico pronto.
plt.show()