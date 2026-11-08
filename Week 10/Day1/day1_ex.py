import os
# Suprime mensagens de log do TensorFlow (INFO/WARNING) antes de qualquer outra importação
# TF_CPP_MIN_LOG_LEVEL: 0=todos os logs, 1=INFO, 2=WARNING, 3=ERROS apenas
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Desativa otimizações oneDNN do TensorFlow para evitar avisos de diferenças numéricas
# por ordem diferente de operações de ponto flutuante (round-off errors)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
# Suprime UserWarnings (ex: aviso sobre input_shape em camadas Keras)
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

# -------------------------------------------------------------------------
# PARTE 1: Carregamento e visualização do dataset CIFAR-10 (está comentado)
# -------------------------------------------------------------------------
# O CIFAR-10 contém 60.000 imagens coloridas 32x32 pixels divididas em 10 classes
# (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão)
# Cada imagem é representada como um tensor de forma (3, 32, 32):
#   3 canais de cor (RGB), 32 linhas (altura), 32 colunas (largura)
# Matematicamente, uma imagem é uma função I: {0..31}×{0..31} → ℝ³ que atribui
# a cada pixel um vetor de 3 componentes (vermelho, verde, azul) no intervalo [0, 1]

# transform = transforms.ToTensor()
# Converte uma imagem PIL (0-255, inteiros) em tensor PyTorch de floats no intervalo [0, 1]
# Matematicamente: para cada pixel (x, y) e canal c, divide o valor inteiro por 255:
#   T[c, y, x] = PIL_image[x, y][c] / 255.0
# Isso normaliza os dados para facilitar a convergência do treinamento de redes neurais

# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# Baixa (se necessário) e carrega o conjunto de treino do CIFAR-10
# root='./data': pasta onde os dados são armazenados localmente
# train=True: usa a partição de treino (50.000 imagens); train=False usaria os 10.000 de teste
# download=True: faz o download automático caso os arquivos não existam localmente
# transform=transform: aplica a conversão ToTensor a cada imagem ao acessá-la

# -------------------------------------------------------------------------
# PARTE 2: Visualização de 5 imagens de amostra (está comentado)
# -------------------------------------------------------------------------
# fig, axes = plt.subplots(1, 5, figsize=(12, 3))
# Cria uma figura com 1 linha e 5 colunas de subplots (eixos)
# figsize=(12, 3): largura 12 polegadas, altura 3 polegadas
# Retorna fig (objeto Figure) e axes (array 1D com 5 objetos Axes)
# Cada Axes é uma área de desenho independente onde uma imagem será plotada

# for i in range(5):  # itera exatamente 5 vezes (uma por imagem)
#     image, label = train_dataset[i]  # acessa a i-ésima amostra (tupla: imagem, rótulo)
#     # image: tensor (3, 32, 32) → canal primeiro (CHW)
#     # label: inteiro de 0 a 9 indicando a classe
#     # image.permute(1, 2, 0): rearranja as dimensões de (C, H, W) para (H, W, C)
#     # O matplotlib imshow espera imagens no formato (altura, largura, canais)
#     # Matematicamente, permute é uma transposição de eixos (reordenação de dimensões)
#     axes.flat[i].imshow(image.permute(1, 2, 0))  # exibe a imagem na posição i
#     axes.flat[i].axis('off')  # remove os eixos (coordenadas x, y) da visualização
#     axes.flat[i].set_title(f"Label {label}")  # coloca o número da classe como título
# plt.show()  # renderiza e exibe a figura na tela (bloqueia até a janela ser fechada)

# -------------------------------------------------------------------------
# PARTE 3: Exibição dos valores de pixel da primeira imagem (está comentado)
# -------------------------------------------------------------------------
# image, label = train_dataset[0]
# image: tensor (3, 32, 32) com valores de ponto flutuante em [0, 1]
# print(image): imprime todos os valores de pixel no console (útil para inspecionar dados brutos)

# -------------------------------------------------------------------------
# PARTE 4: Definição de um CNN (Rede Neural Convolucional) simples com TensorFlow/Keras
# -------------------------------------------------------------------------
import tensorflow as tf

# tf.keras.Sequential([...])
# Modelo sequencial: camadas são empilhadas linearmente, a saída de uma é a entrada da próxima
# Representa uma função composta: f = f_n ◦ f_{n-1} ◦ ... ◦ f_1
# onde cada f_i é uma camada (transformação) aprendível

model = tf.keras.Sequential([
    # Camada 1: Convolução 2D
    # tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3))
    #   - 32 filtros (núcleos) de tamanho 3×3, cada um com 3 canais de entrada (RGB)
    #   - input_shape=(32, 32, 3): entrada esperada é (altura, largura, canais) = formato HWC do TensorFlow
    # Operação matemática da convolução:
    #   S[i, j, k] = (I * K)[i, j, k] = Σ_m Σ_n Σ_c I[i+m, j+n, c] × K[m, n, c, k] + b_k
    #   onde I é a imagem de entrada, K é o núcleo (kernel/filtro), b é o viés (bias)
    #   O resultado é um mapa de características (feature map) de tamanho (30, 30, 32)
    #   (32-3+1 = 30, sem padding e com stride=1)
    #   A função ReLU aplica: f(x) = max(0, x) elemento a elemento, introduzindo não-linearidade
    # Sem ReLU, a composição de camadas lineares resultaria em uma única transformação linear,
    # perdendo a capacidade de aproximar funções complexas (Teorema de Aproximação Universal)
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),

    # Camada 2: Max Pooling 2×2
    # tf.keras.layers.MaxPooling2D((2, 2))
    #   - Reduz a resolução espacial pela metade: (30, 30) → (15, 15)
    #   - Para cada janela 2×2, seleciona apenas o valor máximo
    #   - Matematicamente: P[i, j] = max(I[2i:2i+2, 2j:2j+2])
    #   - Propriedades: invariância a pequenas translações, redução de parâmetros,
    #     controle de sobreajuste (overfitting) e redução do custo computacional
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Camada 3: Achata (Flatten) o tensor multidimensional em um vetor unidimensional
    # tf.keras.layers.Flatten()
    #   - Converte o tensor (15, 15, 32) em um vetor de 15×15×32 = 7.200 elementos
    #   - Necessário porque as camadas Dense (fully connected) esperam entrada 1D
    #   - Não há parâmetros aprendíveis; é apenas uma rearranjo de memória (reshape)
    tf.keras.layers.Flatten(),

    # Camada 4: Densa (fully connected) com 128 neurônios e ReLU
    # tf.keras.layers.Dense(128, activation='relu')
    #   - Cada neurônio calcula: y = ReLU(Σ_j w_j × x_j + b)
    #   - Onde x é o vetor de entrada (7.200 elementos), w_j são os pesos aprendíveis,
    #     b é o viés, e ReLU introduz não-linearidade
    #   - Total de parâmetros nesta camada: 7.200 × 128 + 128 = 921.728 + 128 = 921.856
    tf.keras.layers.Dense(128, activation="relu"),

    # Camada 5: Densa de saída com 10 neurônios e softmax
    # tf.keras.layers.Dense(10, activation='softmax')
    #   - 10 neurônios, um para cada classe do CIFAR-10
    #   - Softmax: σ(z)_i = e^{z_i} / Σ_j e^{z_j}
    #   - Transforma os logits (saídas brutas) em uma distribuição de probabilidade
    #     sobre as 10 classes, onde Σ p_i = 1 e 0 ≤ p_i ≤ 1
    #   - A classe prevista é: argmax_i p_i (índice da maior probabilidade)
    tf.keras.layers.Dense(10, activation="softmax")
])

# -------------------------------------------------------------------------
# PARTE 5: Compilação do modelo
# -------------------------------------------------------------------------
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#   - optimizer='adam': algoritmo Adaptive Moment Estimation
    #   - Combina descida do gradiente estocástica com momento (médias móveis dos gradientes)
    #   - Adam mantém duas médias móveis por parâmetro:
    #     m_t = β₁·m_{t-1} + (1-β₁)·g_t      (primeiro momento / média)
    #     v_t = β₂·v_{t-1} + (1-β₂)·g_t²    (segundo momento / variância não centrada)
    #   - Corrige viés e atualiza: θ_{t+1} = θ_t - η·m̂_t / (√v̂_t + ε)
    #   - Vantagem: taxas de aprendizado adaptativas por parâmetro
    #   - loss='sparse_categorical_crossentropy': entropia cruzada categórica para rótulos inteiros
    #     - Para uma amostra com rótulo verdadeiro y (inteiro 0-9) e predição p (vetor de probabilidades):
    #       L = -log(p_y)  (logaritmo natural da probabilidade atribuída à classe correta)
    #     - Minimizar isso equivale a maximizar a probabilidade da classe correta (MLE)
    #     - 'sparse' significa que os rótulos são inteiros, não one-hot encoded
    #   - metrics=['accuracy']: acurácia = (número de predições corretas) / (total de amostras)
    #     - Métrica de avaliação, não usada para atualizar os pesos durante o treino
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Mensagem de confirmação de que o modelo foi construído e compilado com sucesso
print("Tensorflow CNN Model is ready")
