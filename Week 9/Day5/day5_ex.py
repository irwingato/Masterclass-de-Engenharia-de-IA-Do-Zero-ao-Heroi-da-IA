# =============================================================================
# CONFIGURAÇÃO DO AMBIENTE E IMPORTS
# =============================================================================

import os
# Desabilita logs de baixo nível do TensorFlow/C++ antes de qualquer importação
# TF_CPP_MIN_LOG_LEVEL: 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
# '3' suprime mensagens de INFO e WARNING, exibindo apenas ERROS críticos.
# Isso evita poluir o stdout com logs internos do TensorFlow (ex: oneDNN, CUDA).
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Desabilita otimizações oneDNN (one Deep Neural Network Library) do Intel.
# oneDNN acelera operações de tensores em CPUs Intel, mas pode introduzir
# pequenas diferenças numéricas devido à ordem diferente de operações de
# ponto flutuante (associatividade não exata em aritmética IEEE 754).
# Definir como '0' garante resultados determinísticos e reproduzíveis.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# TensorFlow: framework de machine learning profundo (deep learning) desenvolvido pelo Google.
# Baseia-se em grafos de computação de tensores (arrays multidimensionales).

from tensorflow.keras.datasets import mnist
# Keras é a API de alto nível do TensorFlow para construção de redes neurais.
# mnist é um módulo que fornece o dataset MNIST (Modified National Institute of
# Standards and Technology database): 70.000 imagens de dígitos manuscritos (0-9),
# cada uma em escala de cinza, 28x28 pixels.

from tensorflow.keras.utils import to_categorical
# Utilitário do Keras para converter vetores de rótulos inteiros em representação
# "one-hot encoding" (vetores binários com um único 1 na posição da classe).

from tensorflow.keras.models import Sequential
# Sequential: modelo de rede neural empilhada (camadas lineares, uma após a outra).
# Adequado para arquiteturas feedforward onde cada camada tem exatamente uma
# entrada e uma saída tensor.

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
# Camadas (layers) básicas de redes neurais:
# - Dense: camada totalmente conectada (fully connected / perceptron multicamadas).
# - Flatten: transforma tensores multidimensionais em vetor 1D.
# - Conv2D: camada convolucional 2D (extração de características espaciais).
# - MaxPooling2D: amostragem por máxima (redução de dimensionalidade espacial).
# - Dropout: regularização que desativa aleatoriamente neurônios durante treino.

# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================

# Load MNIST dataset
# Carrega o dataset MNIST dividido em conjunto de treino e teste.
# Retorna tuplas ((X_train, y_train), (X_test, y_test)) onde:
#   X: imagens (matrizes 28x28 de pixels, valores inteiros 0-255 de intensidade).
#   y: rótulos inteiros (0-9) indicando qual dígito está na imagem.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
# Normalização de características (feature scaling): mapeia valores de pixels
# do intervalo [0, 255] para [0.0, 1.0] dividindo por 255.0.
#
# Motivos matemáticos e de computação:
# 1. Gradientes estáveis: redes neurais usam descida do gradiente. Valores grandes
#    (ex: 255) podem causar explosão de gradientes ou atualizações muito agressivas.
# 2. Convergência mais rápida: otimizadores (como Adam) convergem mais rápido com
#    entradas em escala similar e próxima de zero.
# 3. Ativações não saturam: funções como sigmoid/tanh saturam para |x| grande,
#    tornando gradientes próximos de zero (problema do gradiente desvanecente).
#
# Operações realizadas:
#   reshape(-1, 28, 28, 1): remodela o tensor de (n_amostras, 28, 28) para
#       (n_amostras, 28, 28, 1). O '-1' infere automaticamente o número de amostras.
#       A última dimensão '1' representa 1 canal de cor (escala de cinza).
#       CNNs esperam entrada no formato (batch, altura, largura, canais).
#   astype('float32): converte inteiros (uint8) para ponto flutuante de 32 bits,
#       necessário para operações matemáticas com precisão adequada em GPUs/TPUs.
#   / 255.0: divisão elemento a elemento (broadcasting) por escalar 255.0.
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode labels
# Converte rótulos inteiros (ex: 3) em vetores binários (ex: [0,0,0,1,0,0,0,0,0,0]).
#
# Matemática:
#   Se y_i é a classe verdadeira de amostra i (em {0,...,9}), o vetor one-hot é:
#   y_onehot[i, j] = 1 se j == y_i, caso contrário 0.
#
# Motivo: a camada de saída usa softmax, que produz uma distribuição de
# probabilidade sobre as 10 classes. A perda categorical_crossentropy espera
# rótulos na mesma distribuição (one-hot) para calcular:
#   L = -sum_{j=1}^{10} y_j * log(p_j)
# onde y_j é o target one-hot e p_j é a probabilidade prevista pela rede.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")
# Exibe as dimensões dos tensores para verificar o pré-processamento.
# Esperado: X_train (60000, 28, 28, 1), X_test (10000, 28, 28, 1)

# =============================================================================
# CONSTRUÇÃO DA ARQUITETURA DA REDE NEURAL (CNN)
# =============================================================================

# Build the model
# Cria uma Rede Neural Convolucional (CNN - Convolutional Neural Network)
# sequencial para classificação de dígitos MNIST.

model = Sequential([
    # Camada Convolucional 2D (Conv2D)
    # Aplica 32 filtros (kernels) de tamanho 3x3 sobre a imagem de entrada.
    #
    # Matemática da convolução 2D:
    #   Para cada filtro F (3x3) e cada posição (i, j) da imagem de entrada X:
    #       (X * F)[i, j] = sum_{m=-1}^{1} sum_{n=-1}^{1} X[i+m, j+n] * F[m+1, n+1]
    # O resultado é um feature map (mapa de ativação) que destaca padrões locais
    # (bordas, cantos, texturas) independentes de sua posição na imagem.
    #
    # Propriedades:
    #   - Peso compartilhado (weight sharing): o mesmo filtro varre toda a imagem,
    #     reduzindo drasticamente o número de parâmetros vs camadas totalmente conectadas.
    #   - Equivariância à translação: deslocar o padrão na entrada desloca a ativação
    #     na saída de forma correspondente.
    #   - input_shape=(28, 28, 1): define o formato de entrada (altura, largura, canais).
    #   - activation='relu': função ReLU (Rectified Linear Unit):
    #       f(x) = max(0, x)
    #     Introduz não-linearidade, permite que a rede aprenda funções complexas,
    #     e alivia o problema do gradiente desvanecente em redes profundas.
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),

    # Camada MaxPooling2D (pooling por máximo) de tamanho 2x2 com passo 2.
    # Reduz as dimensões espaciais (altura e largura) pelo fator 2.
    #
    # Matemática:
    #   Para cada janela 2x2 do feature map, seleciona o valor máximo:
    #       Y[i, j] = max(X[2i:2i+2, 2j:2j+2])
    #
    # Propriedades de ciência da computação:
    #   - Redução de dimensionalidade: diminui computação e memória nas camadas seguintes.
    #   - Invariância (aproximada) a pequenas translações e distorções locais.
    #   - Controle de sobreajuste (overfitting): ao reduzir resolução, força a rede
    #     a reter apenas características mais relevantes.
    MaxPooling2D((2, 2)),

    # Flatten: converte o tensor 3D de saída (altura, largura, canais) em um vetor 1D.
    # Necessário porque camadas Dense esperam entrada vetorial, não tensores espaciais.
    # Exemplo: (7, 7, 32) -> (1568,) pois 7 * 7 * 32 = 1568.
    Flatten(),

    # Camada Densa (fully connected) com 128 neurônios e ativação ReLU.
    # Cada neurônio calcula:
    #       z = sum_{k} w_k * x_k + b      (produto interno + viés)
    #       a = ReLU(z) = max(0, z)
    # onde x é o vetor de entrada, w são os pesos treináveis e b é o viés.
    # Esta camada combina as características extraídas pelas camadas convolucionais
    # para tomar decisões de classificação.
    Dense(128, activation="relu"),

    # Dropout com taxa 0.5 (50%): durante o treino, desativa aleatoriamente 50%
    # dos neurônios desta camada a cada passo (batch).
    #
    # Ciência da computação / estatística:
    #   - Regularização: impede co-adaptação de neurônios (nenhum neurônio depende
    #     excessivamente de poucos outros), forçando representações mais robustas.
    #   - Equivalente a treinar uma coleção exponencial de redes "finas" e fazer
    #     média de suas previsões (ensemble implícito).
    #   - Durante avaliação/teste, todos os neurônios estão ativos, mas as saídas
    #     são escaladas (ou os pesos ajustados) para manter a expectativa.
    Dropout(0.5),

    # Camada de saída Densa com 10 neurônios (um para cada dígito 0-9)
    # e ativação softmax.
    #
    # Softmax: transforma logits (saídas brutas z_j) em probabilidades p_j:
    #       p_j = exp(z_j) / sum_{k=1}^{10} exp(z_k)
    #
    # Propriedades:
    #   - p_j em (0, 1) para todo j.
    #   - soma_j p_j = 1 (distribuição de probabilidade válida).
    #   - A classe prevista é argmax_j p_j.
    #   - Differenciável: permite propagação reversa do erro (backpropagation).
    Dense(10, activation="softmax")
])

# Exibe a arquitetura completa da rede (número de camadas, output shapes,
# e total de parâmetros treináveis e não-treináveis).
model.summary()

# =============================================================================
# COMPILAÇÃO DO MODELO (CONFIGURAÇÃO DO APRENDIZADO)
# =============================================================================

# Compile the model
# Configura o processo de treinamento: otimizador, função de perda e métricas.
model.compile(
    # Otimizador Adam (Adaptive Moment Estimation):
    #   Algoritmo de descida do gradiente estocástico com taxas de aprendizado
    #   adaptativas por parâmetro, baseadas em momentos de primeira e segunda
    #   ordem dos gradientes (média móvel e média móvel dos quadrados).
    #   Vantagens: converge rapidamente, lida bem com dados esparsos e não estacionários,
    #   requer pouca ajustagem de hiperparâmetros.
    optimizer="adam",

    # Função de perda: categorical cross-entropy (entropia cruzada categórica).
    #   Para uma amostra com target one-hot y e previsão p:
    #       L = - sum_{j=1}^{10} y_j * log(p_j)
    #   Como y é one-hot, apenas o termo da classe verdadeira contribui:
    #       L = -log(p_classe_verdadeira)
    #   Minimizar isso equivale a maximizar a probabilidade da classe correta
    #   (princípio da verossimilhança máxima).
    loss="categorical_crossentropy",

    # Métrica monitorada durante treino e avaliação: acurácia (fração de previsões corretas).
    metrics=["accuracy"]
)

# =============================================================================
# TREINAMENTO DO MODELO
# =============================================================================

# Train the model
# Executa o treinamento por 10 épocas usando descida do gradiente em mini-batches.
history = model.fit(
    X_train, y_train,
    epochs=10,
    # Número de épocas: 1 passagem completa sobre todo o conjunto de treino.
    # Em cada época, o modelo vê todas as 60.000 amostras (divididas em batches).
    #
    # Mais épocas = melhor aprendizado, mas risco de overfitting (memorização
    # do treino em vez de generalização).

    batch_size=32,
    # Tamanho do mini-batch: 32 amostras por atualização de peso.
    # Ciência da computação:
    #   - Compromisso entre precisão do gradiente e eficiência computacional.
    #   - Batch muito grande: gradiente mais preciso, mas mais memória e menos atualizações por época.
    #   - Batch muito pequeno: gradiente ruído, mas mais atualizações e possível efeito regularizador.
    #   - 32 é um tamanho padrão que cabe facilmente em memória de GPU/CPU.

    validation_split=0.2
    # Separa 20% dos dados de treino (12.000 amostras) como conjunto de validação.
    # Usado para monitorar overfitting: se acurácia de treino sobe mas validação
    # estagna ou cai, o modelo está memorizando em vez de generalizar.
)

# =============================================================================
# AVALIAÇÃO DO MODELO
# =============================================================================

# Evaluate the model
# Avalia o desempenho no conjunto de teste (dados nunca vistos durante treino).
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
# test_accuracy: fração de imagens de teste classificadas corretamente.
# Exemplo: 0.9873 significa 98,73% de acurácia no conjunto de teste.

# =============================================================================
# SERIALIZAÇÃO E CARREGAMENTO DO MODELO
# =============================================================================

# Save the model
# Salva o modelo treinado (arquitetura + pesos treináveis + estado do otimizador)
# no formato HDF5 (.h5). Permite reutilizar o modelo sem retreinar.
model.save('mnist_classifier.h5')

# Load the model
from tensorflow.keras.models import load_model
# Carrega o modelo salvo do disco, reconstruindo arquitetura e restaurando pesos.
loaded_model = load_model('mnist_classifier.h5')

# Verifica que o modelo carregado reproduz o mesmo desempenho (sanity check).
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print(f"Loaded model Accuracy: {accuracy:.4f}")
# Se accuracy carregado == test_accuracy, a serialização funcionou corretamente.
