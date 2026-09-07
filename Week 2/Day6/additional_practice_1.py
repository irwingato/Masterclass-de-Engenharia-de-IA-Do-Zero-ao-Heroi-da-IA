import matplotlib.pyplot as plt  # Importa o módulo de plotagem do Matplotlib (funções de gráfico)
import numpy as np               # Importa o NumPy (geração de dados numéricos e aleatórios)

# ---------------------------------------------------------------------------
# 1. Gerar três conjuntos de dados (Distribuições Normais)
# ---------------------------------------------------------------------------

# Fixa a "semente" do gerador de números aleatórios. Isso garante que os dados
# gerados sejam SEMPRE os mesmos a cada execução (reprodutibilidade dos resultados).
np.random.seed(42)

# np.random.normal(loc=..., scale=..., size=...) gera uma distribuição normal (curva de sino):
#   - loc   = média (onde o "pico" da curva fica)
#   - scale = desvio padrão (quão "larga"/dispersa a curva é)
#   - size  = quantidade de valores gerados
# Cada grupo representa uma POPULAÇÃO de 100 valores numéricos com perfil diferente.
grupo_A = np.random.normal(loc=50, scale=10, size=100) # Média 50, Desvio Padrão 10
grupo_B = np.random.normal(loc=65, scale=12, size=100) # Média 65, Desvio Padrão 12
grupo_C = np.random.normal(loc=40, scale=8, size=100)  # Média 40, Desvio Padrão 8

# ---------------------------------------------------------------------------
# 3. Plotar os histogramas sobrepostos usando o parâmetro alpha (transparência)
# ---------------------------------------------------------------------------

# plt.hist() desenha um HISTOGRAMA (conta quantos valores caem em cada intervalo/barras).
# Por serem chamados 3 vezes seguidas no MESMO gráfico, os histogramas ficam SOBREPOSTOS.
#   - bins=30   = divide o eixo X em 30 intervalos (barras)
#   - alpha=0.5 = transparência de 50%. É isso que permite VÊ-LOS SOBREPOSTOS,
#                 pois as áreas de interseção ficam com cores misturadas/mais escuras.
#   - label=    = nome que aparece na legenda
#   - color=    = cor das barras (formato hexadecimal)
#   - edgecolor= = cor do contorno das barras (preto para destacar os limites)
plt.hist(grupo_A, bins=30, alpha=0.5, label='Grupo A (Média=50)', color='#1f77b4', edgecolor='black')
plt.hist(grupo_B, bins=30, alpha=0.5, label='Grupo B (Média=65)', color='#ff7f0e', edgecolor='black')
plt.hist(grupo_C, bins=30, alpha=0.5, label='Grupo C (Média=40)', color='#2ca02c', edgecolor='black')

# ---------------------------------------------------------------------------
# 4. Customizar títulos, legendas e eixos
# ---------------------------------------------------------------------------

# plt.title() define o título principal do gráfico:
#   - fontsize = tamanho da fonte
#   - fontweight='bold' = texto em negrito
#   - pad=15 = espaço extra (em pixels) entre o título e o gráfico
plt.title('Histograma com Múltiplos Conjuntos de Dados Sobrepostos', fontsize=14, fontweight='bold', pad=15)

# Rótulos dos eixos X e Y (descrevem o que cada eixo representa)
plt.xlabel('Valores', fontsize=12)
plt.ylabel('Frequência', fontsize=12)

# plt.legend() ativa a legenda usando os 'label=' definidos nos histogramas.
# loc='upper right' posiciona a legenda no canto superior direito.
plt.legend(loc='upper right', fontsize=11)

# plt.grid() desenha linhas de grade para facilitar a leitura dos valores:
#   - linestyle='--' = linhas tracejadas
#   - alpha=0.5 = transparência de 50% (grade mais suave, menos poluída)
plt.grid(True, linestyle='--', alpha=0.5)

# ---------------------------------------------------------------------------
# 5. Renderizar o gráfico de forma organizada
# ---------------------------------------------------------------------------

# plt.tight_layout() ajusta automaticamente os espaçamentos para que títulos,
# legendas e rótulos não se cortem nem se sobreponham.
plt.tight_layout()

# plt.show() abre a janela com o gráfico pronto para visualização.
# (É o comando que renderiza/exibe o resultado na tela.)
plt.show()