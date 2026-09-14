"""Teste t para duas amostras independentes (Student, equal_var=True por padrão).
CÓDIGO: compara group1 vs group2 (n=5 cada), obtém t e p, decide com alfa=0.05.
MATEMÁTICA: H0: μ1=μ2 vs H1: μ1≠μ2 (bicaudal). Estatística
  t=(x̄1-x̄2)/EP, EP=sp·√(1/n1+1/n2), sp²=((n1-1)s1²+(n2-1)s2²)/(n1+n2-2) (pooled).
  Sob H0, t~t(gl=n1+n2-2=8). p=P(|T|≥|t||H0). Se p<alfa rejeita H0.
  Pressupostos: independência, normalidade aproximada e variâncias iguais.
  Com n=5 o poder é baixo; não rejeitar ≠ provar igualdade.
"""
from scipy.stats import ttest_ind

# Sample Datasets
# CÓDIGO: duas listas pequenas; médias ~2.72 vs ~2.36 (diferença ~0.36).
# MATEMÁTICA: dispersão intra-grupo (~0.4) é da mesma ordem da diferença,
#   então o sinal/ruído é fraco -> p tende a não ser significativo.
group1 = [2.1, 2.5, 2.8, 3.0, 3.2]
group2 = [1.8, 2.0, 2.4, 2.7, 2.9]

# Perform t-test
# CÓDIGO: ttest_ind assume equal_var=True (Student). Para Welch usar equal_var=False.
# MATEMÁTICA: o p-valor bicaudal já multiplica por 2 as duas caudas da t(8).
t_stat, p_value = ttest_ind(group1, group2)
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Interpretation
# CÓDIGO: regra p<alfa -> rejeita H0 (diferença significativa); senão, falha em rejeitar.
# MATEMÁTICA: alfa=0.05 = tolerância a erro tipo I (falso positivo). "Fail to reject"
#   não confirma H0, só diz que a evidência é insuficiente (pode ser n pequeno).
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: significant difference")
else:
    print("Fail to reject the null hypothesis: no significant difference")