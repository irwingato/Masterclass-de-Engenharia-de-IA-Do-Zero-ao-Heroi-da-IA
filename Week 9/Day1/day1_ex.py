import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      # Suprime logs INFO/WARNING do TensorFlow (nível 2 = apenas erros)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     # Desativa otimizações oneDNN para evitar avisos de precisão float

from tensorflow.keras.datasets import mnist, cifar10
# tensorflow.keras.datasets: módulo do Keras (API do TensorFlow) que contém datasets prontos para uso.
# mnist: banco de dados de dígitos manuscritos (0 a 9) em escala de cinza.
# cifar10: banco de imagens coloridas (RGB) de 10 classes (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão).

import tensorflow as tf
# TensorFlow: framework de computação numérica e aprendizado de máquina desenvolvido pelo Google.
# tf.keras: API de alto nível para construção e treinamento de redes neurais.

import torch.nn as nn
# PyTorch: framework de aprendizado de máquina desenvolvido pelo Meta (Facebook).
# torch.nn: módulo que contém camadas (layers) de redes neurais (ex: Linear, Conv2d, ReLU).

import matplotlib.pyplot as plt
# Matplotlib: biblioteca de visualização de dados (gráficos, imagens, histogramas) para Python.

# =====================================================================
# CARREGAMENTO DO DATASET MNIST
# =====================================================================

# Ciência da Computação: um "dataset" é um conjunto de dados organizado.
# Em aprendizado supervisionado, dividimos os dados em:
#   - Treino (train): usado para o modelo aprender padrões (ajustar pesos).
#   - Teste (test): usado para avaliar o desempenho em dados nunca vistos (generalização).
# Matemática: cada amostra é um vetor/matriz e cada label é um inteiro representando a classe.

(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
# mnist.load_data(): baixa (se necessário) e carrega o dataset MNIST do servidor do Keras.
# Retorna duas tuplas: ((X_train, y_train), (X_test, y_test)).
# X = "features" (entradas/dados): matrizes de pixels das imagens.
# y = "labels" (alvos/classes): números inteiros de 0 a 9 indicando qual dígito está na imagem.
#
# Matemática das dimensões (shape) do MNIST:
#   X_train_mnist.shape = (60000, 28, 28)
#     - 60000: número de amostras de treino (n_samples)
#     - 28: altura da imagem em pixels (n_rows)
#     - 28: largura da imagem em pixels (n_cols)
#   Cada pixel é um valor inteiro de 0 (preto) a 255 (branco) em escala de cinza.
#   Matematicamente, cada imagem é uma matriz M ∈ ℝ^(28×28) com entradas M_ij ∈ [0, 255].
#
#   y_train_mnist.shape = (60000,)
#     - 60000: número de labels, um inteiro por amostra.
#
#   X_test_mnist.shape = (10000, 28, 28)  # 10.000 amostras de teste
#   y_test_mnist.shape = (10000,)

print(f"MNIST Dataset: Train - {X_train_mnist.shape}, Test - {X_test_mnist.shape}")
# shape exibe as dimensões do array NumPy subjacente (TensorFlow/Keras usa arrays NumPy por baixo).

# =====================================================================
# CARREGAMENTO DO DATASET CIFAR-10
# =====================================================================

(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
# cifar10.load_data(): baixa e carrega o dataset CIFAR-10 (60.000 imagens coloridas 32×32).
#
# Matemática das dimensões do CIFAR-10:
#   X_train_cifar.shape = (50000, 32, 32, 3)
#     - 50000: número de amostras de treino
#     - 32: altura em pixels
#     - 32: largura em pixels
#     - 3: número de canais de cor (Red, Green, Blue) -> formato "channels last"
#   Cada imagem é um tensor T ∈ ℝ^(32×32×3), onde cada voxel T_ijk é um valor 0-255.
#
#   y_train_cifar.shape = (50000, 1)  # Note: CIFAR-10 retorna labels com shape (n, 1)
#     - 50000: número de labels
#     - 1: cada label é um único inteiro (classe de 0 a 9)
#
#   X_test_cifar.shape = (10000, 32, 32, 3)
#   y_test_cifar.shape = (10000, 1)

print(f"CIFAR-10 Dataset: Train - {X_train_cifar.shape}, Test - {X_test_cifar.shape}")

# =====================================================================
# CAMADAS DE REDE NEURAL (TensorFlow e PyTorch)
# =====================================================================

# Define a basic dense layer
layers = tf.keras.layers.Dense(units=10, activation='relu')
# Dense (camada totalmente conectada / fully connected layer):
#   - units=10: número de neurônios de saída nesta camada.
#   - activation='relu': função de ativação ReLU (Rectified Linear Unit).
#
# Matemática da camada Dense:
#   Para entrada x ∈ ℝ^n e pesos W ∈ ℝ^(m×n), viés b ∈ ℝ^m:
#     z = W·x + b          (combinação linear / produto escalar matricial)
#     a = ReLU(z) = max(0, z)   (ativação não-linear elemento a elemento)
#   A não-linearidade é essencial: sem ela, pilhas de camadas lineares colapsariam para uma única transformação linear.
#
# Ciência da Computação: camadas densas são o bloco básico de MLPs (Multi-Layer Perceptrons).
# Elas aprendem mapeamentos arbitrários entre espaços de entrada e saída via ajuste iterativo de W e b (backpropagation + descida do gradiente).

print(f"TensorFlow Layer: {layers}")

# Define a basic dense layer
layer = nn.Linear(in_features=10, out_features=5)
# PyTorch nn.Linear: equivalente à camada Dense do Keras.
#   - in_features=10: dimensão da entrada (n)
#   - out_features=5: dimensão da saída (m)
#
# Matemática idêntica à do Keras Dense:
#   y = x·W^T + b   (PyTorch armazena W transposto em relação ao Keras)
#   W ∈ ℝ^(5×10), b ∈ ℝ^5

print(f"PyTorch Layer: {layer}")

# =====================================================================
# VISUALIZAÇÃO DE AMOSTRAS DAS IMAGENS
# =====================================================================

# Visualize MNIST sample
plt.imshow(X_train_mnist[0], cmap='gray')
# plt.imshow(): renderiza uma matriz numérica como imagem na figura.
#   X_train_mnist[0]: primeira imagem do treino (matriz 28×28 de pixels).
#   cmap='gray': colormap escala de cinza (0 = preto, 255 = branco).
#
# Matemática: imshow mapeia cada entrada M_ij da matriz para um nível de intensidade de cor.
# Se a matriz tiver valores 0-255, o matplotlib normaliza para exibição.

plt.title(f"MNIST Label: {y_train_mnist[0]}")
# y_train_mnist[0]: label (classe correta) da primeira imagem (ex: 7).

plt.show()
# Exibe a figura gerada.

# Visualize CIFAR-10 sample
plt.imshow(X_train_cifar[0])
# X_train_cifar[0]: primeira imagem do CIFAR-10 (tensor 32×32×3 com canais R, G, B).
# Sem cmap, o matplotlib interpreta os 3 canais como imagem RGB colorida.

plt.title(f"CIFAR-10 Label: {y_train_cifar[0]}")
# y_train_cifar[0]: label da primeira imagem (inteiro de 0 a 9, mapeado para uma das 10 classes).

plt.show()
