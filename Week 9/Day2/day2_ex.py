import numpy as np

# NumPy é a biblioteca padrão de computação científica em Python.
# Oferece arrays multidimensionais (ndarray) com operações vetorizadas,
# que executam em C por debaixo dos panos → muito mais rápido que loops em Python puro.
# Complexidade: operações vetorizadas são O(n) com baixo fator constante graças à contiguidade de memória e SIMD.

# =============================================================================
# FUNÇÕES DE ATIVAÇÃO
# =============================================================================
# Funções de ativacão introduzem não-linearidade na rede neural.
# Sem elas, a composição de camadas lineares colapsa para uma única transformação linear:
#   W2·(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2)  →  ainda é linear.
# Teorema da aproximação universal (Cybenko, 1989): uma rede com pelo menos uma camada
# escondida e função de ativação não-linear pode aproximar qualquer função contínua
# em um conjunto compacto com erro arbitrário pequeno.

def sigmoid(z):
    # Sigmoid (logística): σ(z) = 1 / (1 + e^(-z))
    # Domínio: ℝ → Imagem: (0, 1). Saúde sempre positiva e limitada.
    # Derivada: σ'(z) = σ(z)·(1 - σ(z)) — útil para backpropagation (regra da cadeia).
    # Problema do gradiente desaparecente: para |z| grande, σ'(z) ≈ 0 → o gradiente some.
    return 1 / (1 + np.exp(-z))

def tanh(z):
    # Tangente hiperbólica: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    # Domínio: ℝ → Imagem: (-1, 1). Saída zero-centrada, o que ajuda na convergência
    # do gradiente descento em comparação com sigmoid (não desloca a média para cima).
    # Relação com sigmoid: tanh(z) = 2·σ(2z) - 1
    # Derivada: tanh'(z) = 1 - tanh(z)^2
    # Também sofre de gradiente desaparecente para |z| grande.
    return np.tanh(z)

def relu(z):
    # ReLU (Unidade Linear Retificada): ReLU(z) = max(0, z)
    # Domínio: ℝ → Imagem: [0, ∞). Não limitada superiormente.
    # Derivada: ReLU'(z) = 1 se z > 0, 0 se z < 0, indefinida em z = 0 (subgradiente = 0 na prática).
    # Vantagens: computação barata (apenas comparação), não saturna para z > 0 → mitiga
    # gradiente desaparecente, esparsidade na ativação (neurônios morrem quando z ≤ 0).
    # Problema: "dying ReLU" — neurônios com z sempre ≤ 0 têm gradiente 0 e nunca aprendem.
    return np.maximum(0, z)

def softmax(z):
    # Softmax: softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
    # Mapeia um vetor de scores (logits) para uma distribuição de probabilidade sobre classes.
    # Propriedades: saída em (0, 1) e soma dos componentes = 1 → interpretável como probabilidade.
    # Usada na camada de saída para classificação multiclasse com entropia cruzada.
    # Estabilidade numérica: subtraímos np.max(z) antes do exp para evitar overflow.
    #   e^(z_i - max(z)) / Σ_j e^(z_j - max(z)) = e^(z_i) / Σ_j e^(z_j)  (matematicamente idêntico)
    #   pois o fator comum e^(-max(z)) cancela entre numerador e denominador.
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)

# =============================================================================
# PASSADA PARA FRENTE (FORWARD PASS)
# =============================================================================
def forward_pass(X, weights, biases, activation_function):
    # Computa a ativação de uma camada: a = φ(W·x + b)
    #   X: vetor de entrada (shape: n_features × 1)
    #   weights: matriz de pesos (shape: n_neurônios × n_features)
    #   biases: vetor de vieses (shape: n_neurônios × 1)
    #   activation_function: φ — qualquer função de ativação acima
    #
    # z = W·x + b é a combinação linear (pré-ativação / logit).
    #   Produto interno escalar: z_i = Σ_j W_ij·x_j + b_i  →  O(n_features · n_neurônios)
    # a = φ(z) aplica a não-linearidade elemento a elemento.
    #
    # Ciência da computação: np.dot usa BLAS (Basic Linear Algebra Subprograms) otimizado,
    # explorando cache locality e instruções SIMD do processador.
    z = np.dot(weights, X) + biases
    a = activation_function(z)
    return a

# =============================================================================
# DADOS DE EXEMPLO
# =============================================================================
# X: entrada com 2 features, 1 amostra (shape: 2 × 1)
X = np.array([[0.5], [0.8]])
# weights: 2 neurônios, cada um com 2 pesos (shape: 2 × 2)
weights = np.array([[0.6, 0.4], [0.6, 0.1]])
# biases: 1 viés por neurônio (shape: 2 × 1)
biases = np.array([[0.1], [0.2]])

# Dicionário que mapeia nomes → funções. Em Python, funções são objetos de primeira classe:
# podem ser armazenadas em estruturas de dados e passadas como argumentos (high-order functions).
activations = {
    "Sigmoid": sigmoid,
    "Tanh": tanh,
    "ReLU": relu,
    "Softmax": softmax
}

# Itera sobre os itens do dicionário e executa a passada para frente com cada ativação.
# Complexidade: O(|ativacoes| · n_neurônios · n_features) — linear no número de ativações testadas.
for name, func in activations.items():
    output = forward_pass(X, weights, biases, func)
    print(f"{name} Activation Output:\n{output}\n")

# =============================================================================
# VISUALIZAÇÃO
# =============================================================================
import matplotlib.pyplot as plt

# Matplotlib é a biblioteca de plotagem padrão do ecossistema científico Python.
# linspace(start, stop, num) gera num pontos igualmente espaçados entre start e stop.
# Usado para amostrar o domínio ℝ e traçar as curvas contínuas das ativações.
z = np.linspace(-10, 10, 100)

plt.figure(figsize=(12, 8))  # Cria figura com 12×8 polegadas

# Plota cada função de ativação sobre o mesmo domínio [-10, 10].
# label é usado para a legenda; o grid auxilia leitura visual dos valores.
plt.plot(z, sigmoid(z), label="Sigmoid")
plt.plot(z, tanh(z), label="Tanh")
plt.plot(z, relu(z), label="ReLU")
plt.plot(z, softmax(z), label="Softmax")

plt.title("Activation")           # Título do gráfico
plt.xlabel('Input (z)')           # Rótulo do eixo x (domínio)
plt.ylabel('Output')              # Rótulo do eixo y (imagem)
plt.legend()                      # Exibe a legenda associando cores aos nomes
plt.grid(True)                    # Grade de fundo para facilitar leitura
plt.show()                        # Renderiza e exibe a janela do gráfico