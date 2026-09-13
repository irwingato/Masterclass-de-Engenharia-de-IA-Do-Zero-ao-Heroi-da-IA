"""
additional_practice_1.py
========================
OBJETIVO: Criar uma visualização da distribuição MULTINOMIAL para múltiplas classes de dados.

O que é a Multinomial?
- É a generalização da Binomial para MAIS de 2 resultados possíveis.
- Exemplo: jogar um dado viciado de 4 faces 1000 vezes, onde cada face
  (Classe A, B, C, D) tem uma probabilidade diferente.
- Cada experimento retorna QUANTAS VEZES cada classe apareceu.

O que este script faz:
1. Simula 1 experimento grande (1000 tentativas) e compara Observado x Esperado.
2. Repete o experimento 1000 vezes para mostrar a variabilidade estatística
   (distribuição de densidade de cada classe).
"""

# --- 1. IMPORTAÇÃO DAS BIBLIOTECAS ---
import numpy as np  # Biblioteca numérica: usada para gerar números aleatórios (multinomial, arange)
import pandas as pd  # Biblioteca de tabelas: usada para organizar as 1000 simulações em DataFrame
import matplotlib.pyplot as plt  # Biblioteca de gráficos base: usada para criar barras e a figura
import seaborn as sns  # Biblioteca estatística sobre o matplotlib: usada para curvas KDE (densidade)

# --- 2. DEFINIÇÃO DOS PARÂMETROS DA DISTRIBUIÇÃO MULTINOMIAL ---
np.random.seed(42)  # Fixa a semente aleatória: garante o mesmo resultado a cada execução (reprodutibilidade)

# Lista com os nomes das 4 classes (categorias) que serão simuladas
classes = ['Classe A', 'Classe B', 'Classe C', 'Classe D']

# Probabilidade teórica de cada classe em UMA tentativa.
# REGRA OBRIGATÓRIA: a soma tem que ser 1.0 (100%). Aqui: 0.4+0.3+0.2+0.1 = 1.0
# CORREÇÃO: o comentário antigo dizia "soma deve ser 0.1", o correto é 1.0.
probabilidades = [0.4, 0.3, 0.2, 0.1]

# Número de tentativas (jogadas do "dado") dentro de CADA experimento
n_tentativas = 1000

# Trava de segurança: interrompe o programa se as probabilidades não somarem 1.0
assert abs(sum(probabilidades) - 1.0) < 1e-9, "A soma das probabilidades tem que ser 1.0!"

# --- 3. SIMULAÇÃO DE UMA ÚNICA AMOSTRA (1 experimento com 1000 tentativas) ---
# np.random.multinomial(n, pvals) sorteia quantas vezes cada classe saiu em n tentativas.
# Exemplo de retorno possível: [402, 298, 195, 105] (soma sempre = n_tentativas)
amostra_unica = np.random.multinomial(n_tentativas, probabilidades)

# --- 4. SIMULAÇÃO DE MÚLTIPLAS AMOSTRAS (para ver o comportamento estatístico) ---
n_experimentos = 1000  # Quantas vezes vamos repetir o experimento de 1000 tentativas

# Com size=n_experimentos, o numpy retorna uma matriz (1000 linhas x 4 colunas):
# cada linha = 1 experimento, cada coluna = contagem de uma classe
multiplas_amostras = np.random.multinomial(n_tentativas, probabilidades, size=n_experimentos)

# Converte a matriz numpy em DataFrame do pandas para facilitar análise e plotagem.
# columns=classes nomeia as 4 colunas como 'Classe A', 'Classe B', etc.
df = pd.DataFrame(multiplas_amostras, columns=classes)

# --- 5. CÁLCULO DOS VALORES ESPERADOS (TEORIA) ---
# Valor esperado = probabilidade x nº de tentativas. Ex: Classe A = 0.4 * 1000 = 400
valores_esperados = [p * n_tentativas for p in probabilidades]

# Mostra no terminal a comparação entre o que foi simulado e o que a teoria prevê
print("Amostra única (observado):", dict(zip(classes, amostra_unica)))
print("Valores esperados (teoria):", dict(zip(classes, valores_esperados)))
print("\nMédia das 1000 simulações (deve ficar perto do esperado):")
print(df.mean().round(2))  # Média de cada classe ao longo dos 1000 experimentos

# --- 6. VISUALIZAÇÃO GRÁFICA (2 gráficos lado a lado) ---
# Cria uma figura com 1 linha e 2 colunas de eixos (subplots), tamanho 15x6 polegadas
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ----- Gráfico 1 (esquerda): Barras Observado vs Esperado em 1 experimento -----
x = np.arange(len(classes))  # Cria posições [0,1,2,3] no eixo X, uma para cada classe

# Barra da esquerda (deslocada -0.2): valor OBSERVADO vindo da simulação, cor azul-clara
axes[0].bar(x - 0.2, amostra_unica, width=0.4, label='Observado (Simulação)', color='skyblue')

# Barra da direita (deslocada +0.2): valor ESPERADO vindo da teoria, cor salmão transparente
axes[0].bar(x + 0.2, valores_esperados, width=0.4, label='Esperado (Teórico)', color='salmon', alpha=0.7)

axes[0].set_xticks(x)  # Marca as posições [0,1,2,3] no eixo X
axes[0].set_xticklabels(classes)  # Troca os números pelos nomes das classes
axes[0].set_title(f'Resultado de 1 Experimento ({n_tentativas} Tentativas)')  # Título do gráfico 1
axes[0].set_ylabel('Frequência de Ocorrências')  # Rótulo do eixo Y (quantas vezes saiu)
axes[0].legend()  # Exibe a legenda (Observado x Esperado)

# ----- Gráfico 2 (direita): Curvas de densidade (KDE) das 1000 repetições -----
# Para cada classe, desenha a curva de densidade dos 1000 valores simulados.
# Isso mostra a variabilidade: Classe A varia em torno de 400, Classe D em torno de 100.
for classe in classes:  # Loop sobre as 4 classes
    # sns.kdeplot estima e desenha a curva suave de densidade; fill preenche a área
    sns.kdeplot(df[classe], ax=axes[1], label=classe, fill=True, alpha=0.3)

axes[1].set_title(f'Distribuição de Densidade após {n_experimentos} Experimentos')  # Título do gráfico 2
axes[1].set_xlabel('Número de Ocorrências por Experimento')  # Eixo X: contagem por experimento
axes[1].set_ylabel('Densidade')  # Eixo Y: densidade de probabilidade estimada
axes[1].legend()  # Exibe a legenda com as 4 classes

# --- 7. AJUSTE FINAL E EXIBIÇÃO ---
plt.tight_layout()  # Ajusta automaticamente os espaçamentos para nada ficar cortado
plt.show()  # Abre a janela com os 2 gráficos
