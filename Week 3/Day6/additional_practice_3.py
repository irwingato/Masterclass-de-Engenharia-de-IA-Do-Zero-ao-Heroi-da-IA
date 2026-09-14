"""Intervalo de Confiança (IC) para proporção de aprovação.
CÓDIGO: gera 200 Bernoulli(p=0.65), calcula p_hat=X/n e o IC 95% de Wald
  (statsmodels se disponível, senão fallback manual com scipy).
MATEMÁTICA: cada aluno ~ Bernoulli(p); soma X ~ Binomial(n,p). Estimador
  p_hat=X/n com E[p_hat]=p e Var=p(1-p)/n. Pelo TCL, p_hat≈Normal(p, p(1-p)/n).
  IC 95%: p_hat ± z_{0.975}·EP, EP=√(p_hat(1-p_hat)/n), z_{0.975}≈1.96.
  Interpretação frequentista: 95% dos ICs construídos assim contêm o p verdadeiro.
  Condição de Wald: n·p_hat>=5 e n·(1-p_hat)>=5. Wilson é melhor se n pequeno
  ou p perto de 0/1.
"""
import numpy as np
import pandas as pd

try:
    from statsmodels.stats.proportion import proportion_confint
    _HAS_STATS_MODELS = True
except ImportError:  # Fallback sem statsmodels: aproximação normal (Wald)
    _HAS_STATS_MODELS = False
    from scipy.stats import norm

    def proportion_confint(count, nobs, alpha=0.05, method='normal'):
        p = count / nobs
        z = norm.ppf(1 - alpha / 2)
        se = np.sqrt(p * (1 - p) / nobs)
        return max(0.0, p - z * se), min(1.0, p + z * se)

# 1. Criar um conjunto de dados simulado (Exemplo: Status de aprovação de 200 alunos)
# CÓDIGO: choice([1,0], p=[0.65,0.35]) = 200 ensaios Bernoulli independentes.
# MATEMÁTICA: P(resultado=1)=0.65. Esperado: 200·0.65=130 aprovados, dp=√(200·0.65·0.35)~6.7.
# 1 = Aprovado | 0 = Reprovado
np.random.seed(42)
dados_exame = pd.DataFrame({
    'aluno_id': range(1, 201),
    'resultado': np.random.choice([1, 0], size=200, p=[0.65, 0.35]) # 65% de chance de aprovação teórica
})

# 2. Contar o número de sucessos (aprovados) e o total de observações
# CÓDIGO: sucessos=X=Σresultado; p_hat=X/n = estimativa pontual de p.
# MATEMÁTICA: E[p_hat]=p (não-viesado); EP=√(p(1-p)/n) cai com 1/√n (dobrar precisão=4x amostra).
sucessos = dados_exame['resultado'].sum()
total_amostra = len(dados_exame)
proporcao_amostral = sucessos / total_amostra

print(f"Tamanho da amostra: {total_amostra}")
print(f"Número de aprovados (sucessos): {sucessos}")
print(f"Proporção amostral (p_hat): {proporcao_amostral:.2%} ({proporcao_amostral:.4f})\n")

# 3. Calcular o Intervalo de Confiança (Nível de confiança de 95%)
# CÓDIGO: alpha=0.05; method='normal' = Wald. Fallback manual: z=norm.ppf(0.975)~1.96.
# MATEMÁTICA: IC = p_hat ± 1.96·√(p_hat(1-p_hat)/n). Ex.: p_hat=0.67, n=200 ->
#   EP~0.033, margem~0.065 -> IC~[60.5%, 73.5%]. Margem = z·EP (metade da largura).
nivel_confianca = 0.95
alfa = 1 - nivel_confianca

# O método 'normal' usa a aproximação de Wald (ideal para amostras grandes)
# O método 'wilson' é mais preciso para amostras pequenas ou proporções perto de 0 ou 1
limite_inferior, limite_superior = proportion_confint(
    count=sucessos, 
    nobs=total_amostra, 
    alpha=alfa, 
    method='normal'
)

# 4. Exibir os resultados formatados
print(f"--- Intervalo de Confiança ({nivel_confianca:.0%}) ---")
print(f"Limite Inferior: {limite_inferior:.2%}")
print(f"Limite Superior: {limite_superior:.2%}")
print(f"Margem de Erro: {(limite_superior - proporcao_amostral):.2%}")
print(f"\nInterpretação: Temos 95% de confiança de que a verdadeira proporção "
      f"de aprovação na população está entre {limite_inferior:.2%} e {limite_superior:.2%}.")