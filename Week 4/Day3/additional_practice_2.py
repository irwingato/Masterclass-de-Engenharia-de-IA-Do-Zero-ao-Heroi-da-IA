import numpy as np
import pandas as pd
import scipy.stats as stats

# Objetivo: amostragem ESTRATIFICADA e comparação de ICs entre estratos.
# Matemática por estrato: IC = media ± t * SE, onde SE = s/sqrt(n), t = quantil 97.5% da t com df=n-1.
# Ideia da estratificação: dividir a população heterogênea em grupos homogêneos (estratos)
# e estimar cada um separado. Reduz a variância e evita que um grupo domine a média geral.

# Semente fixa para reprodutibilidade.
np.random.seed(42)

# CRIANDO A POPULAÇÃO (Simulação de uma base de funcionários)
# Cada estrato possui médias e dispersões diferentes
# Júnior: salário baixo e homogêneo. Sênior: alto e bem disperso.
np_por_extrato = 100
dados_jr = np.random.normal(loc=3500, scale=400, size=np_por_extrato)
dados_pl = np.random.normal(loc=6500, scale=800, size=np_por_extrato)
dados_sr = np.random.normal(loc=12000, scale=2000, size=np_por_extrato)

# Junta tudo num DataFrame com coluna de valor + rótulo do estrato.
df = pd.DataFrame({
    'Salario': np.concatenate((dados_jr, dados_pl, dados_sr)),
    'Estrato': ['Júnior'] * np_por_extrato + ['Pleno'] * np_por_extrato + ['Sênior'] * np_por_extrato
})

# 2 AMOSTRAGEM ESTRATIFICADA E CÁLCULO DOS INTERVALOS DE CONFIANÇA
# Aqui cada estrato inteiro (n=100) é tratado como a amostra daquele estrato.
resultados = []
for nome, grupo in df.groupby('Estrato'):
    # Valores de salário só daquele estrato.
    valores = grupo['Salario'].values
    n = len(valores)
    # Média amostral do estrato: centro do IC.
    media = np.mean(valores)
    # Erro-padrão = s/sqrt(n). stats.sem já faz isso (com ddof=1).
    erro_padrao = stats.sem(valores) # Desvio padrão amostral / raiz(n)

    # Calculando o IC de 95% usando a distribuição t-Student
    # t.interval faz media ± t(0.975, df=n-1)*SE. Com n=100, t≈1.98 (≈1.96 da Normal).
    ic = stats.t.interval(0.95, df=n-1, loc=media, scale=erro_padrao)

    resultados.append({
        'Estrato': nome,
        'Média': media,
        'Limite Inferior': ic[0],
        'Limite Superior': ic[1],
    })

# Convertendo em DataFrame para exibição limpa
# Comparação: os 3 ICs não se sobrepõem (Jr ~3386-3530, Pl ~6366-6669, Sr ~11699-12560),
# logo a diferença salarial entre níveis é estatisticamente significativa.
df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))
