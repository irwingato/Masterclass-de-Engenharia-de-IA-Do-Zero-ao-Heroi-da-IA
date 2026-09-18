import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Objetivo: visualizar o que significa "95% de confiança" repetindo o experimento 25x.
# Matemática: cada IC = media_amostral ± t * SE, com t = quantil 97.5% da t (df=n-1).
# Interpretação frequentista: se repetirmos infinitas vezes, ~95% dos intervalos conterão
# a média verdadeira. O gráfico mostra isso: ~24 de 25 linhas verdes (contêm) e ~1 vermelha (não).

# 1. PARÂMETROS DA POPULAÇÃO E DA SIMULAÇÃO
np.random.seed(42)
MEDIA_REAL = 50.0       # A verdadeira média da população (desconhecida na prática)
DESVIO_PADRAO = 10.0   # Desvio padrão populacional
TAMANHO_AMOSTRA = 30   # Tamanho de cada amostra (n)
N_AMOSTRAGENS = 25     # Quantidade de experimentos/amostras independentes
CONFIANCA = 0.95       # Nível de confiança (95%)

# Configurando o gráfico
plt.figure(figsize=(12, 7))

# 2. LOOP DE SIMULAÇÃO E PLOTAGEM
for i in range(N_AMOSTRAGENS):
    # Coleta uma amostra aleatória da população normal
    # Cada iteração é um experimento independente, como coletar 30 pessoas de novo.
    amostra = np.random.normal(loc=MEDIA_REAL, scale=DESVIO_PADRAO, size=TAMANHO_AMOSTRA)
    # Média dessa amostra específica: varia a cada repetição em torno de 50.
    media_amostral = np.mean(amostra)
    # Erro-padrão estimado = s/sqrt(n). Mede a incerteza dessa média.
    erro_padrao = stats.sem(amostra) # Erro padrão estimado da amostra

    # Calcula os limites do Intervalo de Confiança (t-Student)
    # t.interval faz media ± t(0.975, df=29)*SE. Com n=30, t≈2.045 (pouco maior que 1.96).
    lim_inf, lim_sup = stats.t.interval(CONFIANCA, df=TAMANHO_AMOSTRA-1, loc=media_amostral, scale=erro_padrao)

    # Verifica se o intervalo gerado CAPTUROU a média real da população
    capturou = (lim_inf <= MEDIA_REAL <= lim_sup)

    # Define a cor: Verde se contiver a média real, Vermelho de falhar
    cor = '#2ca02c' if capturou else '#d62728'
    # Linha vermelha mais grossa para destacar a falha (que deve ser rara: ~5%).
    largura_linha = 2.0 if capturou else 3.5

    # Desenha a linha do intervalo de confiança
    plt.plot([lim_inf, lim_sup], [i, i], color=cor, linewidth=largura_linha, zorder=1)
    # Desenha o ponto da média amostral obtida
    plt.scatter(media_amostral, i, color=cor, edgecolor='black', s=45, zorder=2)

# 3. ELEMENTOS VISUAIS E DE REFERÊNCIA
# Linha vertical preta indicando o verdadeiro parâmetro populacional
plt.axvline(x=MEDIA_REAL, color='black', linestyle='--', linewidth=2, label=f'Média Real População ({MEDIA_REAL})')

# Estilização dos eixos
# Cada linha y = um experimento (Amostra 1..25). Eixo x = escala dos valores.
plt.yticks(range(N_AMOSTRAGENS), [f'Amostra {i+1}' for i in range(N_AMOSTRAGENS)], fontsize=9)
plt.xlabel('Valores da Escala / Estimativas da Média', fontsize=11, fontweight='bold')
plt.title(f'Visualização de {N_AMOSTRAGENS} Intervalos de Confiança Diferentes ({CONFIANCA*100:.0f}%)', fontsize=13, fontweight='bold', pad=15)
plt.grid(axis='x', linestyle=':', alpha=0.6)

# Legenda customizada
# Truque: plot vazio só para criar item de legenda verde/vermelho.
plt.plot([], [], color='#2ca02c', linewidth=2, label='Contém a Média Real')
plt.plot([], [], color='#d62728', linewidth=3, label='Não Contém a Média Real')
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()
plt.show()
