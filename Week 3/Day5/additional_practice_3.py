"""
additional_practice_3.py
========================
OBJETIVO: Usar distribuições de probabilidade para SIMULAR e ANALISAR conjuntos
de dados do mundo real (do tipo que se encontra em sites públicos de dados).

Conjuntos simulados (inspirados em dados reais de sites diferentes):
1. Tempo de atendimento de suporte de TI  -> Distribuição EXPONENCIAL
   Inspiração: logs de helpdesk / call center (ex: datasets do Kaggle "Helpdesk",
   UCI, data.gov). A maioria dos chamados é rápida, poucos demoram muito.
2. Preços de imóveis de uma cidade       -> Distribuição LOG-NORMAL
   Inspiração: Kaggle "House Prices - Advanced Regression Techniques" e portais
   imobiliários (Zillow, VivaReal). Muitas casas médias/baratas, poucas mansões caras.
3. Requisições por minuto em um servidor -> Distribuição de POISSON
   Inspiração: logs de servidores web (ex: datasets da AWS, Google Cloud, UCI
   "Web Traffic"). Conta eventos por intervalo de tempo com taxa média conhecida.

O que este script faz:
1. Gera 5000 amostras sintéticas de cada cenário com parâmetros realistas.
2. Plota os 3 histogramas lado a lado com linha da média.
3. Imprime estatísticas (describe) e 3 perguntas de negócio com probabilidade.
"""

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
import numpy as np  # Gera as amostras aleatórias: exponential(), lognormal(), poisson()
import pandas as pd  # Monta o DataFrame resumo e calcula estatísticas com describe()
import matplotlib.pyplot as plt  # Cria a figura com 3 subplots e as linhas de média
import seaborn as sns  # Desenha histogramas bonitos com curva KDE (histplot)

# --- 2. CONFIGURAÇÃO INICIAL ---
np.random.seed(42)  # Fixa a semente: mesmos números a cada execução (reprodutibilidade)
n_amostras = 5000  # Tamanho de cada conjunto simulado (5000 linhas = amostra robusta)

# --- 3. SIMULAÇÃO 1: Tempo de atendimento de suporte (EXPONENCIAL) ---
# Por que Exponencial? Modela TEMPO DE ESPERA entre eventos: muitos valores pequenos
# e cauda longa à direita (poucos atendimentos muito demorados). É sem memória.
# Parâmetro scale=8 significa MÉDIA de 8 minutos por atendimento (típico de helpdesk).
# Fonte real análoga: tickets de TI do Kaggle / logs de call center.
dados_suporte = np.random.exponential(scale=8, size=n_amostras)

# --- 4. SIMULAÇÃO 2: Preços de imóveis (LOG-NORMAL) ---
# Por que Log-Normal? Preços nunca são negativos e têm ASSIMETRIA positiva:
# concentram-se em valores médios e esticam para mansões milionárias.
# mu=5.5 e sigma=0.6 são os parâmetros no espaço LOGARÍTMICO (não em reais).
# Multiplicamos por 1000 para converter a escala para reais (R$).
# Média teórica = exp(mu + sigma²/2)*1000 ≈ exp(5.68)*1000 ≈ R$ 293 mil (realista).
# Fonte real análoga: Kaggle "House Prices" / portais VivaReal, Zillow.
mu, sigma = 5.5, 0.6  # mu = centro no espaço log, sigma = espalhamento no espaço log
dados_imoveis = np.random.lognormal(mean=mu, sigma=sigma, size=n_amostras) * 1000

# --- 5. SIMULAÇÃO 3: Requisições por minuto no servidor (POISSON) ---
# Por que Poisson? Modela CONTAGEM de eventos discretos por intervalo fixo de tempo,
# quando os eventos são independentes e têm taxa média constante.
# Parâmetro lam=15 significa MÉDIA de 15 requisições por minuto no servidor.
# Fonte real análoga: logs de acesso web (AWS CloudWatch, NASA web logs da UCI).
dados_servidor = np.random.poisson(lam=15, size=n_amostras)

# --- 6. CRIAÇÃO DA FIGURA COM 3 GRÁFICOS LADO A LADO ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 linha, 3 colunas, figura larga 18x5

# ----- Gráfico 1 (esquerda): Suporte -> Exponencial -----
# histplot com kde=True mostra barras + curva suave; bins=40 divide em 40 faixas de tempo
sns.histplot(dados_suporte, kde=True, ax=axes[0], color='crimson', bins=40)
# Linha vertical tracejada na média para marcar o centro (deve ficar ≈ 8 min)
axes[0].axvline(dados_suporte.mean(), color='black', linestyle='--', label=f"Média {dados_suporte.mean():.1f} min")
axes[0].set_title('Logs de TI: Tempo de Suporte\n(Distribuição Exponencial)')  # Título com contexto + distribuição
axes[0].set_xlabel('Tempo de Atendimento (minutos)')  # Eixo X: minutos
axes[0].set_ylabel('Frequência')  # Eixo Y: quantos atendimentos caíram em cada faixa
axes[0].legend()  # Mostra a legenda da linha de média

# ----- Gráfico 2 (centro): Imóveis -> Log-Normal -----
# Mesmo estilo de histograma; a cauda direita longa é a marca da Log-Normal
sns.histplot(dados_imoveis, kde=True, ax=axes[1], color='teal', bins=40)
# Linha da média (≈ R$ 293 mil) para mostrar que ela fica à DIREITA do pico (assimetria)
axes[1].axvline(dados_imoveis.mean(), color='black', linestyle='--', label=f"Média R$ {dados_imoveis.mean():,.0f}")
axes[1].set_title('Kaggle: Preço de Venda de Imóveis\n(Distribuição Log-Normal)')  # Título
axes[1].set_xlabel('Preço do Imóvel (R$)')  # Eixo X: valor em reais
axes[1].set_ylabel('Frequência')  # Eixo Y: quantidade de imóveis por faixa de preço
axes[1].set_xlim(0, 1500000)  # Corta o eixo em R$ 1.5M para a cauda extrema não achatar o gráfico
axes[1].legend()  # Mostra a legenda da média

# ----- Gráfico 3 (direita): Servidor -> Poisson -----
# kde=False porque Poisson é DISCRETA (números inteiros); discrete=True centraliza
# cada barra em um número inteiro de requisições (..., 14, 15, 16, ...)
sns.histplot(dados_servidor, kde=False, discrete=True, ax=axes[2], color='darkorange', alpha=0.7)
# Linha da média (≈ 15) que deve coincidir com o pico, pois Poisson é quase simétrica aqui
axes[2].axvline(dados_servidor.mean(), color='black', linestyle='--', label=f"Média {dados_servidor.mean():.1f} req/min")
axes[2].set_title('Logs Web: Requisições por Minuto\n(Distribuição de Poisson)')  # Título
axes[2].set_xlabel('Quantidade de Requisições')  # Eixo X: nº de requisições no minuto
axes[2].set_ylabel('Frequência')  # Eixo Y: quantos minutos tiveram aquela contagem
axes[2].legend()  # Mostra a legenda da média

# --- 7. AJUSTE E EXIBIÇÃO DOS GRÁFICOS ---
plt.tight_layout()  # Ajusta margens para os 3 títulos não colidirem
plt.show()  # Abre a janela com os 3 histogramas

# --- 8. ANÁLISE ESTATÍSTICA COMPLEMENTAR ---
# Junta as 3 simulações em um DataFrame: cada coluna = um dataset do "mundo real"
df_resumo = pd.DataFrame({
    'Tempo Suporte (Min)': dados_suporte,  # Coluna 1: minutos de cada atendimento simulado
    'Preço Imóvel (R$)': dados_imoveis,  # Coluna 2: preço de cada imóvel simulado
    'Requisições Servidor': dados_servidor  # Coluna 3: requisições em cada minuto simulado
})

# describe() calcula: count, mean, std, min, 25%, 50% (mediana), 75%, max de cada coluna.
# .round(2) arredonda para 2 casas decimais para leitura fácil.
print("=== Resumo estatístico dos 3 datasets simulados ===")
print(df_resumo.describe().round(2))

# --- 9. PERGUNTAS DE NEGÓCIO (probabilidades empíricas = frequência relativa) ---
# Calculamos P(evento) como (nº de casos favoráveis / total), direto dos dados simulados.
p_suporte_longo = (dados_suporte > 15).mean()  # Prob. de um atendimento passar de 15 min
p_imovel_caro = (dados_imoveis > 500000).mean()  # Prob. de um imóvel custar mais de R$ 500 mil
p_servidor_pico = (dados_servidor > 20).mean()  # Prob. de um minuto ter mais de 20 requisições (pico)

print("\n=== Análises de negócio (probabilidades empíricas) ===")
print(f"P(Suporte > 15 min) = {p_suporte_longo:.2%} -> útil para dimensionar equipe/SLA.")  # Exponencial: cauda
print(f"P(Imóvel > R$ 500 mil) = {p_imovel_caro:.2%} -> útil para segmentar mercado de luxo.")  # Log-Normal: cauda
print(f"P(Servidor > 20 req/min) = {p_servidor_pico:.2%} -> útil para alerta de sobrecarga.")  # Poisson: pico
