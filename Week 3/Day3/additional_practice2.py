"""
PROGRAMA: Gradiente Descendente em Duas Variáveis — comparação de taxas de aprendizado

OBJETIVO: Minimizar a função de custo f(x, y) = x² + 5 + y² (um paraboloide
          elíptico) usando Gradiente Descendente, variando a taxa de aprendizado
          para comparar velocidade de convergência e detectar divergência.

CONCEITO MATEMÁTICO:
  - PARABOLOIDE ELÍPTICO: f(x,y) = x² + y² + 5 é uma "tigela" com fundo no
    ponto (0, 0). O +5 apenas desloca a "tigela" para cima (não muda onde fica
    o mínimo). O mínimo global é em (x, y) = (0, 0), onde f = 5.
  - GRADIENTE ∇f: vetor das derivadas parciais que aponta para a direção de
    MAIOR crescimento da função (a direção da subida mais íngreme).
  - GRADIENTE DESCENDENTE: para achar o mínimo, "descemos" indo na direção
    oposta ao gradiente, ou seja, na direção de maior descida.
  - TAXA DE APRENDIZADO (lr): o tamanho do passo dado em cada iteração.
    Se for pequena, converge lento; se for grande demais, pode "pular" por
    cima do mínimo e divergir.
"""

# numpy fornece arrays, operações vetorizadas e numpy.linalg.norm (módulo de vetor).
import numpy as np


# ─── PASSO 1: FUNÇÃO DE CUSTO E SEU GRADIENTE ANALÍTICO ───────────────────────
def f(x, y):
    """Função de custo (paraboloide elíptico)."""
    # f(x, y) = x² + 5 + y²
    #
    # MATEMÁTICA: é um paraboloide simétrico em x e y. As curvas de nível são
    # círculos concêntricos ao redor do mínimo em (0, 0). É uma função convexa,
    # ou seja, tem UMA única solução ótima (mínimo global) — ideal para testar
    # o gradiente descendente porque não há mínimos locais falsos.
    return x**2 + 5 + y**2

def grad_f(x, y):
    """Gradiente analítico da função [df/dx, df/dy]."""
    # O gradiente é o vetor das derivadas parciais de primeira ordem:
    #   ∇f = ( ∂f/∂x , ∂f/∂y )
    #
    #   ∂f/∂x = ∂(x² + 5 + y²)/∂x = 2x    (o +5 e o y² são constantes em relação a x)
    #   ∂f/∂y = ∂(x² + 5 + y²)/∂y = 2y    (o x² e o +5 são constantes em relação a y)
    #
    # ⚠ NOTA PEDAGÓGICA: aqui usamos 10·y no lugar de 2·y apenas para ENFATIZAR
    # a influência da taxa de aprendizado. Multiplicar o gradiente por uma
    # constante positiva (10) NÃO muda a direção da descida (continua apontando
    # para o fundo da tigela), só aumenta a "intensidade" do passo — funciona
    # como se multiplicássemos a taxa de aprendizado por 10 na coordenada y.
    # Em uma implementação real, o valor correto seria np.array([2*x, 2*y]).
    return np.array([2*x, 10 * y])


# ─── PASSO 2: ALGORITMO DO GRADIENTE DESCENDENTE ───────────────────────────────
def gradient_descent(lr, x_init, y_init, max_epochs=1000, tol=1e-4):
    # lr: taxa de aprendizado (tamanho do "passo").
    # x_init, y_init: ponto de partida.
    # max_epochs: número máximo de iterações permitidas.
    # tol: tolerância — quando o gradiente é desprezível, paramos.
    x, y = x_init, y_init
    history = []  # registra o valor de custo f(x, y) em cada iteração

    for epoch in range(max_epochs):
        # 1) Calcula o gradiente no ponto atual (a direção de maior subida)
        grad = grad_f(x, y)

        # 2) Norma (módulo/comprimento) do vetor gradiente:
        #    ||∇f|| = sqrt(g_x² + g_y²)
        #    É a "inclinação local". Próximo de 0 → estamos num ponto plano,
        #    ou seja, num mínimo (ou máximo) local.
        grad_norm = np.linalg.norm(grad)

        # 3) Guarda o custo atual na lista de histórico
        history.append(f(x, y))

        # 4) CRITÉRIO DE PARADA:
        #    Se o gradiente for quase zero, a função está plana nesse ponto
        #    → já convergimos para o mínimo. Não adianta continuar.
        if grad_norm < tol:
            return epoch, history, (x, y)

        # 5) ATUALIZAÇÃO DOS PARÂMETROS (o núcleo do algoritmo):
        #    Nova posição = posição antiga − lr × ∇f
        #      x ← x − lr·2x  = x·(1 − 2·lr)
        #      y ← y − lr·10y = y·(1 − 10·lr)
        #
        #    POR QUE MENOS? ∇f aponta para onde a função MAIS SOBE.
        #    Para minimizar, andamos na direção OPOSTA (−∇f), e lr controla
        #    "o quanto" andamos. Em cada iteração damos um passo em direção
        #    ao fundo da "tigela" (0, 0).
        #
        #    ANÁLISE DE CONVERGÊNCIA DA COORDENADA y:
        #    A cada iteração y se multiplica por (1 − 10·lr).
        #      * |1 − 10·lr| < 1  → y encolhe → CONVERGE (ex.: lr < 0.2)
        #      * |1 − 10·lr| = 1  → y "empaca" oscilando (lr = 0.2 ou 0)
        #      * |1 − 10·lr| > 1  → y cresce a cada passo → DIVERGE (lr > 0.2)
        x = x - lr * grad[0]
        y = y - lr * grad[1]

        # 6) PROTEÇÃO CONTRA DIVERGÊNCIA:
        #    Se lr for grande demais, o passo "passa por cima" do mínimo e
        #    afasta cada vez mais → valores explodem para infinito/NaN.
        #    Detectamos isso e abortamos retornando None.
        if np.isnan(x) or np.isnan(y) or abs(x) > 1e10:
            return epoch, history, None

    # Se o loop terminar sem convergir (estourou max_epochs), retorna o ponto atual
    return epoch, history, (x, y)


# ─── PASSO 3: CONFIGURAÇÃO DO EXPERIMENTO ──────────────────────────────────────
# Ponto de partida fixo para todos os testes (10 unidades do mínimo em (0,0))
x_start, y_start = 4.0, 4.0

# Taxas de aprendizado a comparar:
#   - 0.01 → passo curto: x encolhe 2%/iteração, y encolhe 10%/iteração (lento)
#   - 0.05 → passo intermediário: x encolhe 10%/iteração, y 50%/iteração
#   - 0.1  → y = y·(1−1) = 0 → y chega ao mínimo em UM único passo!
#   - 0.22 → fator de y = (1 − 10·0.22) = −1.2; |−1.2| > 1 → y oscila e explode
learning_rates = [0.01, 0.05, 0.1, 0.22]

print(f"Ponto de partida inicial: ({x_start}, {y_start})")
print("-" * 65)
print(f"{'Taxa (LR)':<12} | {'Status':<12} | {'Iterações':<10} | {'Ponto Final Mínimo'}")
print("-" * 65)

# ─── PASSO 4: RODAR O EXPERIMENTO E COMPARAR OS RESULTADOS ────────────────────
for lr in learning_rates:
    # Roda o gradiente descendente com cada taxa de aprendizado.
    # Retorna: (nº de iterações usadas, histórico de custos, ponto final)
    epochs, _, final_pt = gradient_descent(lr, x_start, y_start)

    # Classifica o resultado de acordo com o comportamento observado:
    if final_pt is None:
        # O algoritmo disparou a proteção contra overflow → lr grande demais
        status = "Divergiu"
        pt_str = "N/A (Explodiu)"
    elif epochs < 1000:
        # Parou antes de esgotar as iterações → atingiu gradiente ≈ 0 (convergiu)
        # O ponto deve estar perto do mínimo teórico (0, 0)
        status = "Convergiu "
        pt_str = f"({final_pt[0]:.4f}, {final_pt[1]:.4f})"
    else:
        # Usou todas as 1000 iterações sem parar → ainda não chegou no mínimo
        # (geralmente por lr muito pequeno, convergindo devagar demais)
        status = "Não convergiu"
        pt_str = f"({final_pt[0]:.4f}, {final_pt[1]:.4f})"

    print(f"{lr:<12} | {status:<12} | {epochs:<10} | {pt_str}")

# ─── LEITURA DE RESULTADOS (esperado) ─────────────────────────────────────────
#   lr=0.01   → convergência lenta (muitas iterações); y converge 5× mais rápido que x
#   lr=0.05   → converge rápido, ponto final ≈ (0, 0)
#   lr=0.1    → y "cai" direto a 0 no 1º passo; x converge depois → rápido
#   lr=0.22   → y oscila com fator −1.2 (cresce em módulo) → "Divergiu"
#
#   ⚠ O fator 10·y foi usado de propósito para deixar a divergência VISÍVEL
#   com lr = 0.22. Com o gradiente matemático exato (2·y), o limiar de
#   divergência em y seria lr = 1, e todos os valores desta tabela convergiriam.