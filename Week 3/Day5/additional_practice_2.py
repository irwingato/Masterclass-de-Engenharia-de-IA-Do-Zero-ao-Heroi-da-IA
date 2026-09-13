"""
additional_practice_2.py
========================
OBJETIVO: Comparar a distribuição GAUSSIANA (Normal) com a distribuição
UNIFORME CONTÍNUA para dados contínuos.

Conceitos:
- Gaussiana (Normal, média=0, desvio=1): forma de sino, valores se concentram
  no centro e caem nas caudas. Ex: altura, erros de medição, notas.
- Uniforme Contínua (min=-2, max=2): todos os valores no intervalo têm a MESMA
  chance. Gráfico ideal seria um retângulo plano. Ex: gerador aleatório.

O que este script faz:
1. Gera 100.000 pontos de cada distribuição.
2. Mostra estatísticas descritivas no terminal (média, desvio, min, max).
3. Plota lado a lado: (a) histograma + KDE sobrepostos, (b) boxplot comparativo.
"""

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
import numpy as np  # Usada para gerar números aleatórios normal() e uniform()
import matplotlib.pyplot as plt  # Usada para criar a figura, eixos e linhas de referência
import seaborn as sns  # Usada para histograma com KDE (histplot) e boxplot
import pandas as pd  # Usada para organizar os dados em DataFrame e calcular describe()

# --- 2. CONFIGURAÇÃO INICIAL ---
np.random.seed(42)  # Fixa a semente para que os números sorteados sejam sempre iguais
n_pontos = 100000  # Quantidade de amostras por distribuição (grande = curvas suaves e estáveis)

# --- 3. GERANDO DADOS DA DISTRIBUIÇÃO GAUSSIANA (NORMAL) ---
# np.random.normal(loc=0, scale=1): loc = média (mu), scale = desvio-padrão (sigma).
# Resultado: ~68% dos valores entre -1 e 1, ~95% entre -2 e 2, caudas raras além disso.
dados_gaussiana = np.random.normal(loc=0, scale=1, size=n_pontos)

# --- 4. GERANDO DADOS DA DISTRIBUIÇÃO UNIFORME CONTÍNUA ---
# np.random.uniform(low=-2, high=2): sorteia qualquer valor real entre -2 e 2
# com probabilidade igual. Escolhemos -2 a 2 para a base coincidir visualmente
# com a região onde a Gaussiana concentra ~95% dos dados, facilitando a comparação.
dados_uniforme = np.random.uniform(low=-2, high=2, size=n_pontos)

# --- 5. ESTATÍSTICAS DESCRITIVAS NO TERMINAL (prova numérica da diferença) ---
# Monta um DataFrame com as duas colunas para usar o método describe()
df_comp = pd.DataFrame({'Gaussiana': dados_gaussiana, 'Uniforme': dados_uniforme})
print("=== Estatísticas: Gaussiana vs Uniforme ===")
# describe() mostra contagem, média, desvio, min, quartis e max de cada coluna
print(df_comp.describe().round(4))
# Teoria para conferir: Gaussiana ~ média 0, desvio 1 / Uniforme(-2,2) ~ média 0, desvio ~1.1547
print("\nMédia Gaussiana:", round(dados_gaussiana.mean(), 4), "| Desvio:", round(dados_gaussiana.std(), 4))
print("Média Uniforme:", round(dados_uniforme.mean(), 4), "| Desvio:", round(dados_uniforme.std(), 4))

# --- 6. CRIAÇÃO DA FIGURA COM 2 GRÁFICOS LADO A LADO ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 linha, 2 colunas, tamanho 14x6 polegadas

# ----- Gráfico 1 (esquerda): Histograma + Curva de densidade (KDE) sobrepostos -----
# Histograma da Gaussiana: bins=60 barras, densidade normalizada (stat='density'),
# cor azul, transparência 0.35 para ver a sobreposição com a Uniforme
sns.histplot(dados_gaussiana, bins=60, stat='density', color='royalblue',
             alpha=0.35, label='Gaussiana (hist)', ax=axes[0])

# Histograma da Uniforme: mesma configuração, cor laranja, para comparar o formato
sns.histplot(dados_uniforme, bins=60, stat='density', color='darkorange',
             alpha=0.35, label='Uniforme (hist)', ax=axes[0])

# Curva KDE da Gaussiana: estimativa suave da densidade, linha grossa azul, área preenchida
# CORREÇÃO: label antigo tinha erro de LaTeX ('Min=-2$, Max=2$'); aqui está corrigido.
sns.kdeplot(dados_gaussiana, label='Gaussiana (Normal)\n$\\mu=0, \\sigma=1$',
            color='royalblue', fill=True, alpha=0.25, linewidth=2.5, ax=axes[0])

# Curva KDE da Uniforme: deve ficar quase plana entre -2 e 2 (com leve arredondamento
# nas bordas por efeito do suavizador KDE — isso é normal e está explicado no título)
sns.kdeplot(dados_uniforme, label='Uniforme Contínua\n$Min=-2, Max=2$',
            color='darkorange', fill=True, alpha=0.20, linewidth=2.5, ax=axes[0])

# Linha vertical tracejada na média (=0) das duas distribuições para mostrar o centro comum
axes[0].axvline(0, color='black', linestyle='--', linewidth=1.2, label='Média = 0')

axes[0].set_title('Densidade: Gaussiana (sino) vs. Uniforme (plana)', fontsize=12, fontweight='bold')  # Título
axes[0].set_xlabel('Valores dos Dados', fontsize=11)  # Rótulo do eixo X
axes[0].set_ylabel('Densidade de Probabilidade', fontsize=11)  # Rótulo do eixo Y
axes[0].set_xlim(-4, 4)  # Limita o eixo X para mostrar as caudas da Gaussiana sem cortar a Uniforme
axes[0].grid(axis='y', linestyle='--', alpha=0.5)  # Grade horizontal suave para ler a densidade
axes[0].legend(fontsize=9, loc='upper right')  # Legenda no canto superior direito

# ----- Gráfico 2 (direita): Boxplot comparativo (mediana, quartis e outliers) -----
# O boxplot resume cada distribuição: caixa = 50% centrais, linha = mediana, pontos = outliers.
# Gaussiana terá caudas com outliers; Uniforme terá caixa simétrica e sem outliers típicos.
sns.boxplot(data=df_comp, palette={'Gaussiana': 'royalblue', 'Uniforme': 'darkorange'}, ax=axes[1])

axes[1].set_title('Boxplot: dispersão e outliers', fontsize=12, fontweight='bold')  # Título
axes[1].set_ylabel('Valores', fontsize=11)  # Rótulo do eixo Y
axes[1].grid(axis='y', linestyle='--', alpha=0.5)  # Grade horizontal para facilitar a leitura

# --- 7. AJUSTE FINAL E EXIBIÇÃO ---
plt.suptitle('Comparação Gaussiana vs. Uniforme para Dados Contínuos', fontsize=14, fontweight='bold')  # Título geral
plt.tight_layout()  # Ajusta espaçamentos para nada sobrepor ou cortar
plt.show()  # Exibe a janela com os 2 gráficos
