"""Teste de hipótese: dois métodos de ensino (A=tradicional vs B=projetos).
CÓDIGO: gera notas simuladas, checa normalidade (Shapiro), checa variâncias (Levene),
  aplica t-test ou Mann-Whitney, decide por p-valor vs alfa e plota boxplot+stripplot.
MATEMÁTICA: H0: mu_A = mu_B vs H1: mu_A != mu_B (bicaudal), alfa=0.05 = P(erro tipo I).
  p-valor = P(dados tão extremos | H0). Se p < alfa, rejeita H0.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Criar um conjunto de dados reais simulados (Notas de Exame)
# CÓDIGO: normal(loc, scale, 50) por grupo + clip(0,100) p/ manter escala de nota.
# MATEMÁTICA: X ~ Normal(mu, sigma²), f(x)=1/(sigma√2π)·exp(-(x-mu)²/2sigma²).
#   Grupo A: mu=72, sigma=12 | Grupo B: mu=78, sigma=10 -> diferença real de +6 pontos.
# Grupo A: Método de ensino tradicional | Grupo B: Método de ensino focado em projetos
np.random.seed(42)
notas_grupo_A = np.random.normal(loc=72, scale=12, size=50) # média=72
notas_grupo_B = np.random.normal(loc=78, scale=10, size=50) # média=78

# Garantir que as notas fiquem no intervalo de 0 a 100
notas_grupo_A = np.clip(notas_grupo_A, 0, 100)
notas_grupo_B = np.clip(notas_grupo_B, 0, 100)

# --- DEFINIÇÃO DAS HIPÓTESES ---
# Hipótese Nula (H0): Não há diferença significativa entre as médias das notas dos dois grupos.
# Hipótese Alternativa (H1): Há uma diferença significativa entre as médias das notas.
alfa = 0.05 # Nível de significância de 5%

print("--- PASSO 1: Verificação de Normalidade (Shapiro-Wilk) ---")
# CÓDIGO: shapiro() retorna (W, p); p > alfa => não rejeita normalidade.
# MATEMÁTICA: W = (Σaᵢ·x_(i))² / Σ(xᵢ-x̄)², onde x_(i) são ordenados e aᵢ os pesos
#   esperados da Normal. H0 do Shapiro: "amostra veio de Normal". É pré-requisito
#   do t-test (que assume médias ~ Normais por TCL + resíduos Normais).
_, p_norm_A = stats.shapiro(notas_grupo_A)
_, p_norm_B = stats.shapiro(notas_grupo_B)

print(f"P-valor Grupo A: {p_norm_A:.4f} | P-valor Grupo B: {p_norm_B:.4f}")
dados_normais = p_norm_A > alfa and p_norm_B > alfa
print(f"Os dados são normais? {'Sim' if dados_normais else 'Não'}\n")

print("--- PASSO 2: Execução do Teste de Hipótese ---")
# CÓDIGO: se normal -> Levene decide Student (equal_var=True) ou Welch (False);
#   senão -> Mann-Whitney (ranks, sem assumir Normal).
# MATEMÁTICA: Levene H0: sigma_A²=sigma_B² (ANOVA dos desvios |x-x̄|).
#   t = (x̄A-x̄B)/EP, EP=sp·√(1/nA+1/nB) [Student, sp² pooled] ou
#   EP=√(sA²/nA+sB²/nB) [Welch, gl≈Satterthwaite]. Sob H0, t~t Student(gl).
#   Mann-Whitney: U = nA·nB + nA(nA+1)/2 − R_A (R_A=soma dos ranks de A);
#   H0: P(A>B)=0.5 (distribuições iguais). Usa ranks -> robusto a outliers.
if dados_normais:
    # Se forem normais, verifica-se a igualdade de variâncias (Levene)
    _, p_levene = stats.levene(notas_grupo_A, notas_grupo_B)
    variancias_iguais = p_levene > alfa
    
    # Executa o Teste t de Student Independente
    t_stat, p_valor = stats.ttest_ind(notas_grupo_A, notas_grupo_B, equal_var=variancias_iguais)
    tipo_teste = f"Teste t de Student (Variâncias iguais={variancias_iguais})"
else:
    # Se não forem normais, utiliza-se a alternativa não-paramétrica
    t_stat, p_valor = stats.mannwhitneyu(notas_grupo_A, notas_grupo_B, alternative='two-sided')
    tipo_teste = "Teste U de Mann-Whitney (Não-paramétrico)"

print(f"Teste Aplicado: {tipo_teste}")
print(f"Estatística de teste: {t_stat:.4f}")
print(f"P-valor obtido: {p_valor:.4f}")

if p_valor < alfa:
    print("Resultado: REJEITA-SE a hipótese nula (H0).")
    print("Conclusão: Há uma diferença estatisticamente significativa entre as notas dos dois grupos.")
else:
    print("Resultado: NÃO REJEITA-SE a hipótese nula (H0).")
    print("Conclusão: Não há evidências de diferença significativa entre os dois grupos.")

# --- PASSO 3: Visualização dos Resultados ---
# CÓDIGO: boxplot (mediana, Q1/Q3, whiskers 1.5·IQR, outliers) + stripplot (pontos reais
#   com jitter) p/ ver forma, sobreposição e outliers além do p-valor.
df = pd.DataFrame({
    'Notas': np.concatenate([notas_grupo_A, notas_grupo_B]),
    'Grupo': ['Grupo A (Tradicional)']*50 + ['Grupo B (Projetos)']*50
})

plt.figure(figsize=(8, 5))
sns.boxplot(x='Grupo', y='Notas', data=df, hue='Grupo', palette='Set2', legend=False)
sns.stripplot(x='Grupo', y='Notas', data=df, color='black', alpha=0.3, jitter=0.2)
plt.title(f'Comparação de Notas entre Grupos\n({tipo_teste} | p-valor: {p_valor:.4f})', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()