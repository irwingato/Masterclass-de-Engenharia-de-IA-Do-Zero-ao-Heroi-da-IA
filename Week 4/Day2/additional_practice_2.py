"""
additional_practice_2.py — VARIÁVEIS ALEATÓRIAS de DISTRIBUIÇÕES PERSONALIZADAS
===============================================================================
OBJETIVO:
    Mostrar 3 formas de criar/amostrar distribuições que NÃO existem prontas
    no numpy/scipy, do mais prático ao mais matemático.

MATEMÁTICA GERAL:
    - PDF f(x): P(a <= X <= b) = integral_a^b f(x) dx, com integral total = 1.
    - CDF F(x) = P(X <= x) = integral_-inf^x f(t) dt (sempre vai de 0 a 1).
    - Amostrar = gerar números que "respeitam" f(x): onde f é alta, saem
      muitos pontos; onde f é baixa, saem poucos.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Semente para reprodutibilidade
np.random.seed(42)
n_samples = 5000

# =====================================================================
# ABORDAGEM 1: Mistura de Distribuições (a mais usada na prática)
# MATEMÁTICA: f_mistura(x) = w1*f1(x) + w2*f2(x), com w1+w2=1.
# Aqui: 70% Normal(15,3) + 30% LogNormal. É como sortear uma moeda viciada:
# com prob. 0.7 pega da Normal, com prob. 0.3 pega da LogNormal.
# RESULTADO: distribuição bimodal / com cauda: corpo normal + cauda de
# "baleias" (ex: 70% compras normais + 30% compras atípicas grandes).
# AMOSTRAGEM: gera n1 pontos de cada componente e concatena.
# =====================================================================
n_comp1 = int(n_samples * 0.7)  # 3500 pontos do comportamento "comum"
n_comp2 = int(n_samples * 0.3)  # 1500 pontos da cauda "atípica"

comp_normal = np.random.normal(loc=15, scale=3, size=n_comp1)
comp_cauda_longa = np.random.lognormal(mean=3, sigma=0.4, size=n_comp2)

dados_mistura = np.concatenate([comp_normal, comp_cauda_longa])

# =====================================================================
# ABORDAGEM 2: Inversão da CDF (Método Analítico Exato)
# TEOREMA DA INVERSÃO: se U ~ Uniforme(0,1), então X = F⁻¹(U) tem CDF F.
# Prova curta: P(X <= x) = P(F⁻¹(U) <= x) = P(U <= F(x)) = F(x).
#
# EXEMPLO AQUI: queremos f(x) = 2x para 0 <= x <= 1 (reta crescente).
#   Passo 1 - CDF: F(x) = integral_0^x 2t dt = x².
#   Passo 2 - Inversa: y = x²  =>  x = sqrt(y), logo F⁻¹(u) = sqrt(u).
#   Passo 3 - Algoritmo: sorteia U ~ Uniforme(0,1), devolve sqrt(U).
# Note que saem mais valores perto de 1 (onde f é maior). Correto!
# Integral de prova: integral_0^1 2x dx = [x²]_0^1 = 1 (é PDF válida).
# =====================================================================
u = np.random.uniform(0, 1, n_samples)
dados_analiticos = np.sqrt(u)  # Transforma a uniforme na distribuição f(x)=2x

# =====================================================================
# ABORDAGEM 3: Distribuição Numérica Contínua Customizada via Scipy
# MATEMÁTICA: queremos f(x) = 0.375*x² para 0 <= x <= 2.
#   Prova que é PDF válida: integral_0^2 0.375*x² dx
#     = 0.375 * [x³/3]_0^2 = 0.375 * 8/3 = 3/8 * 8/3 = 1. OK!
#   É crescente (parábola): valores perto de 2 são mais prováveis.
#
# CORREÇÕES EM RELAÇÃO AO CÓDIGO ANTIGO (que dava
# "RuntimeError: Failed to converge"):
#   1) Informar o SUPORTE a=0, b=2 no construtor. Sem isso o scipy assume
#      suporte (-inf, +inf) e tenta inverter a CDF no infinito -> diverge.
#   2) _pdf deve ser VETORIZADA (recebe array numpy). O antigo
#      "return ... if 0 <= x <= 2 else 0" quebra com arrays.
#      Como o scipy só chama _pdf dentro de [a,b], basta devolver a fórmula.
# =====================================================================
class DistribuicaoCustomizada(stats.rv_continuous):
    def _pdf(self, x):
        # PDF polinomial entre 0 e 2 (já normalizada, integral = 1).
        # x aqui é array numpy -> operação vetorizada, sem "if" escalar.
        return 0.375 * (x ** 2)

# a=0, b=2 diz ao scipy onde a PDF vive. Fora disso, P=0 automaticamente.
dist_custom = DistribuicaoCustomizada(a=0, b=2, name='custom')
dados_scipy = dist_custom.rvs(size=n_samples)

# =====================================================================
# Exibição dos Resultados Estatísticos e Gráficos
# stat="density" normaliza o histograma para área 1 (= estimativa da PDF).
# O título mostra skew (assimetria) e kurtosis (peso das caudas) de cada
# conjunto, para ligar com o additional_practice_1.py.
# =====================================================================
datasets = {
    "1. Mistura (Gaussiana + Lognormal)": dados_mistura,
    "2. Inversão Analítica (f(x)=2x)": dados_analiticos,
    "3. Scipy Custom (f(x)=0.375x²)": dados_scipy
}

plt.figure(figsize=(15, 5))
sns.set_theme(style="whitegrid")

for i, (nome, dados) in enumerate(datasets.items(), 1):
    plt.subplot(1, 3, i)
    sns.histplot(dados, kde=True, color=f"C{i}", stat="density", alpha=0.6)

    sk = stats.skew(dados)
    kt = stats.kurtosis(dados)

    plt.title(f"{nome}\nSkew: {sk:.2f} | Kurtosis: {kt:.2f}", fontsize=11)
    plt.xlabel("Valores Gerados")
    plt.ylabel("Densidade")

plt.tight_layout()
plt.show()
