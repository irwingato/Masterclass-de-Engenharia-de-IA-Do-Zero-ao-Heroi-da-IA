# =============================================================================
# Suporte: suprimir logs ruidosos do TensorFlow (antes de importar o tf)
# =============================================================================
# O TensorFlow e suas dependências (absl, oneDNN) imprimem avisos no STDERR
# antes mesmo de qualquer código seu rodar. Definir estas variáveis de ambiente
# ANTES do `import tensorflow` desliga esses avisos.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0=tudo, 1=INFO, 2=WARNING, 3=ERRO apenas
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # desativa otimizações oneDNN (mensagem de aviso)
import absl.logging
absl.logging.set_verbosity('error')        # reduz verbosidade do logging do absl

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel('ERROR')          # logger do TensorFlow mostra apenas ERROS

# =============================================================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS (regressão linear simples)
# =============================================================================
# np.random.seed(42) fixa a semente do gerador de números pseudo-aleatórios.
# Isso torna o experimento REPRODUTÍVEL: toda execução gera os mesmos X e y.
np.random.seed(42)

# X: matriz 100x1 com valores aleatórios uniformes em [0, 2).
#    np.random.rand(100, 1) produz 100 amostras, 1 feature cada, entre 0 e 1.
#    Multiplicar por 2 escalona para o intervalo [0, 2).
X = 2 * np.random.rand(100, 1)

# y: alvo gerado por um modelo linear verdadeiro com ruído gaussiano:
#    y = 4 + 3*X + epsilon,  epsilon ~ N(0, 1)
#    4   -> intercepto (beta_0) verdadeiro
#    3   -> coeficiente angular (beta_1) verdadeiro
#    np.random.randn(100, 1) amostra de uma Normal Padrão (media 0, desvio 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# (Visualização comentada - descomente para plotar os dados)
# plt.scatter(X, y, color="blue", label="Data")
# plt.title('Generated Dataset')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.grid()
# plt.show()

# =============================================================================
# 2. REGRESSÃO LINEAR MÚLTIPLA VIA DESCIDA DO GRADIENTE (NumPy)
# =============================================================================
# m: número de amostras (linhas) = 100
m = 100

# theta: vetor de parâmetros [theta_0, theta_1]^T inicializado aleatoriamente.
#    theta_0 = intercepto (bias), theta_1 = coeficiente de X.
#    Forma (2, 1) porque vamos usar X_b com coluna de 1s.
theta = np.random.rand(2, 1)

# learning_rate (eta): tamanho do passo na atualização dos pesos.
#    Muito grande -> pode divergir (os saltos ultrapassam o mínimo).
#    Muito pequeno -> convergência lenta.
learning_rate = 0.1

# iterations: número de iterações (épocas) da descida do gradiente.
iterations = 1000

# X_b: matriz de design com coluna de bias (unos) concatenada a X.
#    np.c_ concatena colunas. np.ones((m, 1)) cria um vetor coluna de 1s.
#    X_b = [1, x_i] para cada amostra i, de modo que X_b @ theta =
#          theta_0 + theta_1 * x_i (forma matricial do y = b + w*x).
X_b = np.c_[np.ones((m, 1)), X]

# DESCIDA DO GRADIENTE (Batch Gradient Descent)
# Objetivo: minimizar o Ersto Quadrático Médio (MSE):
#     J(theta) = (1/m) * sum((X_b * theta - y)^2)
# Gradiente do MSE w.r.t. theta:
#     ∇J(theta) = (2/m) * X_b^T * (X_b * theta - y)
# Atualização (regra do gradiente):
#     theta := theta - learning_rate * ∇J(theta)
for iteration in range(iterations):
    # Predição: X_b @ theta tem forma (100, 1)
    # Erro:     (X_b @ theta - y) tem forma (100, 1)
    # Gradiente: (2/m) * X_b.T @ (erro) tem forma (2, 1)
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

print("Optimized Parameters (Theta): \n", theta)

# =============================================================================
# 3. MESMO PROBLEMA COM TENSORFLOW (Gradiente Automático / Autograd)
# =============================================================================
# Converter arrays NumPy em tensores TensorFlow (imutáveis, tipo float32).
# tf.constant cria um tensor constante a partir dos dados.
X_tensor = tf.constant(X, dtype=tf.float32)
y_tensor = tf.constant(y, dtype=tf.float32)

# Definição do modelo como um tf.Module (encapsula variáveis treináveis).
class LinearModel(tf.Module):
    def __init__(self):
        # pesos: variável escalar (matriz 1x1) inicializada com Normal(0,1)
        #      tf.Variable é treinável (será atualizada pelo otimizador)
        self.weights = tf.Variable(tf.random.normal([1, 1]))
        # bias: variável escalar (vetor 1x1) inicializada com Normal(0,1)
        self.bias = tf.Variable(tf.random.normal([1]))

    # __call__ permite chamar a instância como uma função: model(X)
    # Retorna: w * X + b (regressão linear simples)
    def __call__(self, X):
        return self.weights * X + self.bias

# Função de perda: Erro Quadrático Médio (MSE / Mean Squared Error)
#     MSE = (1/n) * sum((y_true - y_pred)^2)
# tf.reduce_mean calcula a média de todos os elementos do tensor.
# tf.square eleva cada elemento ao quadrado.
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Instancia o modelo (pesos e bias inicializados aleatoriamente)
model = LinearModel()

# Otimizador SGD (Stochastic Gradient Descent) com taxa de aprendizado 0.1.
# Aplica a regra: variavel := variavel - learning_rate * gradiente
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# LOOP DE TREINAMENTO COM GRADIENTE AUTOMÁTICO (Autograd)
for epoch in range(100):
    # tf.GradientTape() grava todas as operações para calcular gradientes depois.
    # Dentro do bloco, o TensorFlow constrói um grafo de computação.
    with tf.GradientTape() as tape:
        y_pred = model(X_tensor)           # passagem forward (predição)
        loss = mse_loss(y_tensor, y_pred)  # calcula a perda (MSE)

    # Calcula o gradiente da perda w.r.t. as variáveis treináveis (pesos, bias)
    # usando regra da cadeia (backpropagation automático).
    gradients = tape.gradient(loss, [model.weights, model.bias])

    # Atualiza os parâmetros aplicando os gradientes (regra do SGD)
    optimizer.apply_gradients(zip(gradients, [model.weights, model.bias]))

    # Imprime o progresso a cada 10 épocas
    if epoch % 10 == 0:
        # .numpy() converte o tensor TF de volta para array NumPy para impressão
        print(f"Epoch {epoch}: Loss = {loss.numpy():.4f}")

# Imprime o peso otimizado (o coeficiente angular w aproxima o valor verdadeiro 3)
print("Optimized Parameters (Theta): \n", model.weights.numpy())

# =============================================================================
# RESUMO MATEMÁTICO E DE CIÊNCIA DA COMPUTAÇÃO
# =============================================================================
#
# 1. REGRESSÃO LINEAR
#    Modelo: y = theta_0 + theta_1 * x + epsilon
#    Objetivo: encontrar theta que minimize o MSE (função custo convex).
#
# 2. DESCIDA DO GRADIENTE
#    - É um algoritmo de otimização de primeira ordem (usa apenas gradientes).
#    - Complexidade por iteração: O(m * d) onde m = amostras, d = features.
#    - Converge para o mínimo global porque o MSE é convexo (sem mínimos locais).
#
# 3. TENSORFLOW / AUTOGRAF
#    - GradientTape implementa diferenciação automática (modo reverso/backprop).
#    - Grava operações no forward pass e computa gradientes no backward pass.
#    - tf.Variable são os nós treináveis; tf.constant são dados fixos.
#
# 4. DIFERENÇA ENTRE AS DUAS ABORDAGENS
#    - NumPy: gradiente calculado à mão (fórmula analítica) e atualização manual.
#    - TensorFlow: gradiente obtido por autodiferenciação; otimizador encapsula a regra
#      de atualização (pode ser trocado por Adam, RMSprop, etc. sem mudar o loop).
# =============================================================================