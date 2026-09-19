"""
additional_practice_1.py — Teste Z para DUAS PROPORÇÕES (amostras grandes)
============================================================================
Objetivo: testar se a taxa de conversão do Grupo B difere da do Grupo A.

Hipóteses:
    H0: p1 = p2  (as proporções são iguais; diferença = 0)
    H1: p1 != p2 (teste bicaudal; diferença != 0)

Matemática:
    1) Proporções amostrais: p1_hat = x1/n1, p2_hat = x2/n2.
    2) Sob H0 as duas amostras vêm da mesma proporção, então usa-se a
       proporção COMBINADA (pooled):
           p_pool = (x1 + x2) / (n1 + n2)
       Isso é a estimativa de máxima verossimilhança de p sob H0.
    3) Pelo Teorema Central do Limite, para n grande:
           (p2_hat - p1_hat) ~ Normal(0, SE^2) sob H0,
       onde o Erro Padrão sob H0 é:
           SE = sqrt( p_pool * (1 - p_pool) * (1/n1 + 1/n2) )
    4) Estatística de teste (padronização):
           Z = (p2_hat - p1_hat) / SE  ~  N(0,1) sob H0
    5) Valor-p bicaudal:
           p = 2 * (1 - Phi(|Z|)),
       onde Phi é a FDA da Normal padrão.
       Rejeita-se H0 se p < alfa (aqui alfa = 0.05).

Condições de validade (aproximação Normal):
    n1*p_pool >= 5, n1*(1-p_pool) >= 5 (idem para n2).
    Aqui: n1=150, n2=120 -> satisfeito, por isso o Teste Z é adequado
    (e não o teste binomial exato).
"""
import numpy as np
from scipy import stats

# 1. Configuração dos dados (Exemplo: Conversões de um Teste A/B)
# Amostra A (Grupo Controle): n1 ensaios, x1 sucessos
n1 = 150
sucessos1 = 90  # Taxa de conversão de 60%

# Amostra B (Grupo Variante): n2 ensaios, x2 sucessos
n2 = 120
sucessos2 = 84  # Taxa de conversão de 70%

# 2. Cálculo das proporções amostrais: p_hat = sucessos / n
#    São estimadores não-viesados das proporções populacionais p1 e p2.
p1 = sucessos1 / n1
p2 = sucessos2 / n2

# 3. Proporção combinada (pooled) sob a Hipótese Nula (H0: p1 = p2)
#    Fórmula: p_pool = (x1 + x2) / (n1 + n2)
#    Por que pooled? Sob H0 há UM só parâmetro p, então juntamos as
#    duas amostras para estimá-lo com menor variância.
p_pooled = (sucessos1 + sucessos2) / (n1 + n2)

# 4. Cálculo do Erro Padrão (Standard Error) da diferença sob H0
#    Fórmula: SE = sqrt( p_pool*(1-p_pool)*(1/n1 + 1/n2) )
#    Origem: Var(p1_hat)=p(1-p)/n1, Var(p2_hat)=p(1-p)/n2, amostras
#    independentes -> Var(diferença) = soma das variâncias.
se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

# 5. Cálculo da estatística Z (quantos SEs a diferença observada está de 0)
#    Fórmula: Z = (p2_hat - p1_hat) / SE  ~ N(0,1) sob H0
z_stat = (p2 - p1) / se

# 6. Cálculo do Valor-p (Bicaudal)
#    Fórmula: p = 2 * (1 - Phi(|Z|)), onde Phi = stats.norm.cdf
#    Bicaudal porque H1 é "diferente" (não especifica direção).
#    Interpretação: P(observar |Z| tão extremo quanto o obtido, se H0 for verdade).
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Exibição dos resultados
print(f"--- Resultados do Teste Z ---")
print(f"Proporção A: {p1:.2%}")
print(f"Proporção B: {p2:.2%}")
print(f"Estatística Z: {z_stat:.4f}")
print(f"Valor-p: {p_value:.4f}")

# Conclusão estatística (Alfa de 5%)
# Regra de decisão: se p_value < alfa, rejeita H0.
# Caso contrário, não há evidência suficiente contra H0 (não "prova" H0).
alpha = 0.05
if p_value < alpha:
    print("\nResultado: Rejeita-se a hipótese nula (H0). A diferença é estatisticamente significativa.")
else:
    print("\nResultado: Não rejeita-se a hipótese nula (H0). A diferença pode ser fruto do acaso.")
