"""
additional_practice_3.py — Teste BINOMIAL EXATO para UMA PROPORÇÃO
===================================================================
Objetivo: testar se a proporção de clientes que preferem o chatbot é
MAIOR que 50% (não usa o dataset Íris; é um exemplo de preferência).

Hipóteses (teste UNILATERAL à direita):
    H0: p = 0.50  (sem preferência; chatbot e alternativa empatam)
    H1: p > 0.50  (chatbot é preferido pela maioria)

Matemática (Distribuição Binomial):
    1) Modelo: cada cliente é um ensaio de Bernoulli(p) independente.
       O número total de sucessos X em n ensaios segue:
           X ~ Binomial(n, p)
       com função de probabilidade (PMF):
           P(X = k) = C(n,k) * p^k * (1-p)^(n-k),
       onde C(n,k) = n! / (k!(n-k)!) é o coeficiente binomial.
    2) Sob H0, X ~ Binomial(n=100, p0=0.50). A média esperada sob H0 é
       n*p0 = 50 sucessos, com desvio padrão sqrt(n*p0*(1-p0)) = 5.
    3) Valor-p unilateral à direita = probabilidade de observar um
       resultado TÃO ou MAIS extremo que k na direção de H1, se H0 for
       verdade:
           p_valor = P(X >= k | H0) = soma_{i=k}^{n} C(n,i)*p0^i*(1-p0)^{n-i}
       Aqui k=62: p_valor = P(X >= 62 | n=100, p=0.5).
    4) Regra de decisão: rejeita H0 se p_valor < alfa (alfa=0.05).
       Diferente do Teste Z (que usa aproximação Normal), aqui o cálculo
       é EXATO — soma direta das probabilidades binomiais, sem TCL.
       Por isso é ideal para n pequeno/moderado ou quando se quer
       exatidão.

    5) Se quisesse testar "diferente" (bicaudal), usaria
       alternative='two-sided', cujo p-valor soma as probabilidades de
       todos os k' com P(X=k') <= P(X=k observado).
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Definição dos parâmetros do teste
n = 100        # Tamanho total da amostra (número de ensaios Bernoulli)
k = 62         # Número de sucessos observados (clientes que preferem chatbot)
p_0 = 0.50     # Proporção sob a hipótese nula (H0): p = 0.5 (indiferença)
alfa = 0.05    # Nível de significância (5%): prob. tolerada de Erro Tipo I

# 2. Execução do Teste Binomial Exato (Unilateral - 'greater')
#    scipy.stats.binomtest(k, n, p=p_0, alternative='greater') calcula
#    p_valor = P(X >= k | H0) de forma exata (soma da cauda binomial).
#    CORREÇÃO: era 'alternativa=' (português), que gerava TypeError;
#    o nome correto do parâmetro em inglês é 'alternative'.
#    Para testar se a proporção é apenas diferente, use alternative='two-sided'.
resultado = stats.binomtest(k, n, p=p_0, alternative='greater')
p_valor = resultado.pvalue

print("--- RESULTADOS DO TESTE BINOMIAL ---")
print(f"Número de ensaios (n): {n}")
print(f"Sucessos observados (k): {k} (Proporção amostral: {k/n:.2%})")
print(f"P-valor calculado: {p_valor:.4f}")

# 3. Regra de Decisão
#    Lógica: se o p-valor < alfa, o observado (62/100) seria raro demais
#    sob H0 (menos de 5% de chance), então rejeitamos H0 em favor de H1.
#    Caso contrário, o resultado é compatível com o acaso sob p=0.5.
if p_valor < alfa:
    print(f"Resultado: Rejeitamos H0 ao nível de significância de {alfa:.0%}.")
    print("Há evidências estatísticas de que proporção é maior que 50%.")
else:
    print(f"Resultado: Não rejeitamos H0 ao nível de significância de {alfa:.0%}.")
    print("Não há evidências suficientes para afirmar que a proporção é maior que 50%.")

# 4. Visualização Gráfica da Distribuição sob H0
#    - x = 0..n: todos os valores possíveis de sucessos.
#    - pmf_valores[i] = P(X=i | n, p0): altura da barra = probabilidade exata.
#    - Barras vermelhas (X >= k): região cuja soma = p-valor unilateral.
#    - Linha tracejada em k: onde caiu nossa observação (62).
x = np.arange(0, n + 1)
pmf_valores = stats.binom.pmf(x, n, p_0)

plt.figure(figsize=(10, 5))
plt.bar(x, pmf_valores, color='lightgray', label='Distribuição sob H0 (p=0.5)')
# Destacando a região a partir do nosso valor observado k
plt.bar(x[x >= k], pmf_valores[x >= k], color='red', label=f'Região do P-valor (X >= {k})')
plt.axvline(k, color='darkred', linestyle='--', linewidth=2, label=f'Dado Observado (k={k})')

plt.title("Teste de Hipótese Binomial Exato")
plt.xlabel("Número de Sucessos")
plt.ylabel("Probabilidade (PMF)")
plt.xlim(30, 70)  # Zoom na região de interesse (caudas além disso têm prob. ~0)
plt.legend()
plt.tight_layout()
plt.show()
