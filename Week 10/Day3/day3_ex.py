import os
# Suprime logs do TensorFlow (AVX, oneDNN, etc.) antes de importar bibliotecas de ML.
# TF_CPP_MIN_LOG_LEVEL='3' mostra apenas erros fatais (0=all, 1=INFO, 2=WARNING, 3=ERROR).
# TF_ENABLE_ONEDNN_OPTS='0' desativa otimizações oneDNN que geram mensagens de aviso.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, uniform_filter

# =============================================================================
# PARTE 1: Pooling com SciPy / NumPy (implementação manual)
# =============================================================================

# Cria uma matriz (feature map) 4x4 que representa ativações de uma camada convolucional.
# Cada posição (i, j) contém um valor de ativação extraído por filtros convolucionais.
feature_map = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [0, 1, 2, 3]
])

# -----------------------------------------------------------------------------
# Max Pooling com scipy.ndimage.maximum_filter
# -----------------------------------------------------------------------------
# maximum_filter aplica um filtro de máxima sobre uma janela deslizante (kernel).
# size=2 significa janela 2x2. mode='constant' preenche bordas com valor constante (0).
#
# MATEMÁTICA:
# Para cada posição (i, j) da saída, o resultado é:
#   Y[i, j] = max{ X[i+di, j+dj] | 0 <= di < size, 0 <= dj < size }
#
# CIÊNCIA DA COMPUTAÇÃO:
# Complexidade: O(H * W * k^2) onde H,W são dimensões da entrada e k é o tamanho do kernel.
# Reduz a resolução espacial, retendo apenas o valor mais ativo em cada região.
# Útil para detectar as características mais fortes (bordas, texturas) independentemente
# de sua posição exata dentro da janela (invariância translocal limitada).
max_pooled = maximum_filter(feature_map, size=2, mode='constant')

# -----------------------------------------------------------------------------
# Average Pooling com scipy.ndimage.uniform_filter
# -----------------------------------------------------------------------------
# uniform_filter calcula a média aritmética dos valores dentro de uma janela deslizante.
#
# MATEMÁTICA:
# Para cada posição (i, j) da saída, o resultado é:
#   Y[i, j] = (1 / (k * k)) * sum{ X[i+di, j+dj] | 0 <= di < k, 0 <= dj < k }
# onde k = tamanho do kernel (2 neste caso).
#
# CIÊNCIA DA COMPUTAÇÃO:
# Suaviza a feature map, reduzindo ruído e preservando informações de baixo contraste.
# Diminui a dimensionalidade espacial e o número de parâmetros nas camadas seguintes,
# combatendo overfitting e reduzindo custo computacional.
avg_pooled = uniform_filter(feature_map, size=2, mode='constant')

# -----------------------------------------------------------------------------
# Plotagem (comentada) - visualizaria as matrizes originais e processadas
# -----------------------------------------------------------------------------
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(feature_map, cmap='viridis')
# axes[0].set_title("Original Feature Map")
# axes[1].imshow(max_pooled, cmap='viridis')
# axes[1].set_title("Max Pooled")
# axes[2].imshow(avg_pooled, cmap='viridis')
# axes[2].set_title("Average Pooled")
# plt.show()

# =============================================================================
# PARTE 2: Pooling com TensorFlow (Keras)
# =============================================================================

import tensorflow as tf

# Redimensiona a feature_map para o formato de tensor esperado pelo TensorFlow:
# (batch_size, height, width, channels) = (1, 4, 4, 1)
# - batch_size=1: uma única amostra sendo processada de cada vez.
# - height=4, width=4: dimensões espaciais da imagem/feature map.
# - channels=1: imagem em escala de cinza (1 canal).
input_tensor = tf.constant(feature_map.reshape(1, 4, 4, 1), dtype=tf.float32)

# -----------------------------------------------------------------------------
# Max Pooling 2D com TensorFlow/Keras
# -----------------------------------------------------------------------------
# MaxPooling2D aplica pooling máximo sobre janelas 2x2 com passo (stride) de 2.
# padding='valid' significa que não há preenchimento (zero-padding); a janela
# só se move onde cabe completamente dentro da entrada.
#
# MATEMÁTICA:
# Se entrada tem dimensão H x W e kernel k x k com stride s:
#   Output_H = floor((H - k) / s) + 1   (quando padding='valid')
#   Output_W = floor((W - k) / s) + 1
# Para H=4, W=4, k=2, s=2: Output = floor((4-2)/2) + 1 = 2
# Resultando em uma saída 2x2.
#
# CIÊNCIA DA COMPUTAÇÃO:
# Operação de sub-amostragem (downsampling) que reduz o tamanho espacial em 4x
# (metade na altura, metade na largura). Diminui memória e computação nas camadas
# posteriores, além de fornecer certa invariância a pequenas translações.
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
max_pooled_tensor = max_pool(input_tensor)

# -----------------------------------------------------------------------------
# Average Pooling 2D com TensorFlow/Keras
# -----------------------------------------------------------------------------
# AveragePooling2D calcula a média dos valores em cada janela 2x2.
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')
avg_pooled_tensor = avg_pool(input_tensor)

# remove dimensões de tamanho 1 (batch e channel) para exibição limpa.
print(f"Max Pooled Tensor:\n{tf.squeeze(max_pooled_tensor).numpy()}")
print(f"Average Pooled Tensor:\n{tf.squeeze(avg_pooled_tensor).numpy()}")
print("\n\n\n")

# =============================================================================
# PARTE 3: Pooling com PyTorch
# =============================================================================

import torch
import torch.nn as nn

# Cria tensor no formato (batch_size, channels, height, width) = (1, 1, 4, 4)
# PyTorch usa o formato NCHW (Batch, Channel, Height, Width), diferente do
# TensorFlow que usa NHWC (Batch, Height, Width, Channel).
# unsqueeze(0) adiciona dimensão de batch no início; outro unsqueeze(0) adiciona
# dimensão de canal.
input_tensor = torch.tensor(feature_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# -----------------------------------------------------------------------------
# Max Pooling 2D com PyTorch
# -----------------------------------------------------------------------------
# nn.MaxPool2d(kernel_size=2, stride=2) execulta pooling máximo sobre janelas 2x2.
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
max_pooled_tensor = max_pool(input_tensor)

# -----------------------------------------------------------------------------
# Average Pooling 2D com PyTorch
# -----------------------------------------------------------------------------
# nn.AvgPool2d(kernel_size=2, stride=2) execulta pooling médio sobre janelas 2x2.
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_pooled_tensor = avg_pool(input_tensor)

# .squeeze() remove dimensões de tamanho 1; .numpy() converte tensor PyTorch para array NumPy.
print(f"Max Pooled Tensor:\n{max_pooled_tensor.squeeze().numpy()}")
print(f"Average Pooled Tensor:\n{avg_pooled_tensor.squeeze().numpy()}")

# =============================================================================
# PARTE 4: Exemplos de CNNs completas (definição de arquitetura, sem treinamento)
# =============================================================================

# -----------------------------------------------------------------------------
# Exemplo de CNN com TensorFlow/Keras (Sequential API)
# -----------------------------------------------------------------------------
# Arquitetura: Conv2D -> MaxPool -> Conv2D -> AvgPool
# Cada camada Conv2D aprende filtros (kernels) que detectam padrões espaciais.
# As camadas de pooling reduzem a resolução entre os blocos convolucionais.
model_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),  # Entrada: imagem 32x32 com 3 canais (RGB)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 32 filtros 3x3, ativação ReLU
    tf.keras.layers.MaxPooling2D((2, 2)),                    # Reduz 32x32 -> 16x16
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 64 filtros 3x3
    tf.keras.layers.AveragePooling2D((2, 2))                 # Reduz 16x16 -> 8x8 (após conv)
])

# -----------------------------------------------------------------------------
# Exemplo de CNN com PyTorch (classe Module)
# -----------------------------------------------------------------------------
# nn.Module é a classe base para todas as redes neurais no PyTorch.
# __init__ define as camadas; forward define o fluxo de dados (passagem à frente).
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)   # 3 canais de entrada -> 32 saida, kernel 3x3
        self.pool1 = nn.MaxPool2d(2, 2)                 # Max pooling 2x2, stride 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 32 -> 64 canais, kernel 3x3
        self.pool2 = nn.AvgPool2d(2, 2)                 # Average pooling 2x2, stride 2
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))   # Convoluciona + ReLU (max(0, x))
        x = self.pool1(x)               # Downsample por max pooling
        x = torch.relu(self.conv2(x))   # Segunda convolucao + ReLU
        x = self.pool2(x)               # Downsample por average pooling
        return x
