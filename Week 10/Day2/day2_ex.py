import os
# Suprime mensagens de log do TensorFlow/C++ ANTES de importar o tf.
# TF_CPP_MIN_LOG_LEVEL controla o nível de verbosidade do backend C++:
#   0 = todos os logs, 1 = filtra INFO, 2 = filtra INFO+WARNING,
#   3 = filtra INFO+WARNING+ERROR (mantém apenas FATAL).
# Definir como '3' remove os avisos de inicialização (absl::InitializeLog)
# e as mensagens INFO/WARNING do C++ que escrevem no STDERR.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Desativa as operações customizadas oneDNN do TensorFlow.
# oneDNN pode reordenar operações de ponto flutuante, gerando resultados
# numéricos ligeiramente diferentes entre execuções e imprimindo um aviso
# em port.cc:153. Definir como '0' desliga essa otimização e remove o aviso.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

# =============================================================================
# PARTE 1: Convolução 2D com SciPy (processamento clássico de imagens)
# =============================================================================

# Gera uma imagem sintética em escala de cinza 10x10 com valores aleatórios
# uniformemente distribuídos no intervalo [0, 1).
# Em ciência da computação, isso simula uma grade de pixels; em matemática,
# é uma matriz I ∈ R^(10×10).
image = np.random.rand(10, 10)

# Define um kernel (máscara/filtro) para detecção de bordas (Laplaciano discreto).
# Matriz 3x3 com -1 na periferia e 8 no centro: realça regiões de transição
# brusca de intensidade entre pixels vizinhos.
edge_detection_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Define um kernel de blur (média) 3x3. Cada elemento vale 1/9.
# Matriz que suaviza a imagem calculando a média móvel dos 9 pixels da
# vizinhança de cada posição (filtro passa-baixa).
blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9

# Aplica convolução discreta 2D entre a imagem e cada kernel.
# Para cada posição de saída (i, j):
#   (f * g)[i, j] = Σ_m Σ_n f[i + m, j + n] · g[m, n]
# onde f é a imagem de entrada e g é o kernel.
# Nota: em processamento de imagens costuma-se usar correlação (sem espelhar
# o kernel), que é exatamente o que scipy.ndimage.convolve implementa na prática.
edge_detected_image = convolve(image, edge_detection_kernel)
blurred_image = convolve(image, blur_kernel)

# (Opcional) Visualização com matplotlib: plota original, detecção de bordas
# e blur lado a lado. Comentado para evitar abrir janelas gráficas durante
# execução em ambientes sem display.
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(image, cmap="gray")
# axes[0].set_title("Original Image")
# axes[1].imshow(edge_detected_image, cmap="gray")
# axes[1].set_title("Edge Detected")
# axes[2].imshow(blurred_image, cmap="gray")
# axes[2].set_title("Blurred")
# plt.show()

# =============================================================================
# PARTE 2: Convolução 2D com TensorFlow (aprendizado profundo / CNN)
# =============================================================================

import tensorflow as tf

# Cria um tensor de entrada com forma (batch_size, height, width, channels)
# = (1, 10, 10, 1): um lote com uma única imagem 10x10 de um canal (cinza),
# cujos valores vêm de uma distribuição normal padrão N(0, 1).
image_tensor = tf.random.normal([1, 10, 10, 1])

# Define uma camada convolucional (Conv2D) do Keras.
# filters=1: número de filtros (mapas de características) de saída.
# kernel_size=(3,3): tamanho do campo receptivo (janela) do filtro.
# strides=(1,1): passo de deslocamento do filtro sobre a imagem (1 pixel).
# padding='same': adiciona bordas de zeros ao redor da entrada para que a
# saída tenha as mesmas dimensões espaciais (10x10) da entrada.
conv_layer = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same'
)

# Aplica a convolução ao tensor de entrada.
# Matematicamente, cada posição do canal de saída k é calculada como:
#   O[b, i, j, k] = Σ_di Σ_dj Σ_c I[b, i·s + di, j·s + dj, c] · W[di, dj, c, k] + bias[k]
# onde:
#   I = tensor de entrada,
#   W = pesos do kernel aprendidos pela camada,
#   bias[k] = viés do filtro k,
#   s = stride,
#   c = índice do canal de entrada,
#   k = índice do filtro de saída.
# Essa operação é o bloco fundamental de Redes Neurais Convolucionais (CNNs).
output_tensor = conv_layer(image_tensor)

# Imprime as formas (shapes) dos tensores para conferir que a convolução
# preservou as dimensões espaciais graças ao padding='same'.
print(f"Original Shape: {image_tensor.shape}")
print(f"Output Shape: {output_tensor.shape}")