import os
import sys

# -----------------------------------------------------------------------------
# Supressão de logs do TensorFlow / oneDNN / absl
# -----------------------------------------------------------------------------
# Definir variáveis de ambiente ANTES de qualquer import do TensorFlow. Elas
# devem estar no topo absoluto do arquivo para surtir efeito na inicialização
# do runtime C++.
#
# TF_CPP_MIN_LOG_LEVEL: controla o nível de log do runtime C++ do TensorFlow.
#   0 = todos os logs (default)
#   1 = filtra INFO
#   2 = filtra INFO + WARNING
#   3 = filtra INFO + WARNING + ERROR (apenas FATAL permanece)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TF_ENABLE_ONEDNN_OPTS: desativa otimizações oneDNN (Intel oneAPI Deep Neural
# Network Library). Remove o aviso do port.cc:153 sobre diferenças de ponto
# flutuante por ordem de operações.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# AUTOGRAPH_VERBOSITY: '0' suprime mensagens de debug do AutoGraph.
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

# -----------------------------------------------------------------------------
# Redirecionamento de STDERR de baixo nível (Windows)
# -----------------------------------------------------------------------------
# As mensagens "All log messages before absl::InitializeLog()..." e
# "oneDNN custom operations are on..." são escritas pelo C++ do TensorFlow
# diretamente no file descriptor 2 (STDERR do SO), ANTES de qualquer código
# Python poder interceptá-las via contextlib.redirect_stderr.
#
# No Windows, usamos os.dup2 para redirecionar o descriptor de stderr para
# o dispositivo nulo (nul), silenciando mensagens de C++ de baixo nível.
try:
    import msvcrt
    stderr_fd = sys.stderr.fileno()          # descriptor do stderr (geralmente 2)
    nul_fd = os.open(os.devnull, os.O_RDWR)  # abre 'nul' do Windows
    os.dup2(nul_fd, stderr_fd)               # forja stderr -> nul
    os.close(nul_fd)
except Exception:
    # Se falhar (ex: não Windows), segue sem redirecionar de baixo nível.
    pass

import logging
logging.disable(logging.WARNING)

import contextlib

with contextlib.redirect_stderr(open(os.devnull, 'w')):
    from tensorflow.keras.datasets import imdb

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# -----------------------------------------------------------------------------
# Hiperparâmetros do problema
# -----------------------------------------------------------------------------
# vocab_size: tamanho do vocabulário considerado. Apenas as 10.000 palavras mais
# frequentes do dataset IMDB são mantidas; palavras mais raras são mapeadas para
# o índice 0 (padding/unknown). Isso controla a dimensão da camada Embedding.
vocab_size = 10000

# max_len: comprimento fixo de cada sequência (review). Reviews mais curtos são
# preenchidos com zeros (padding); reviews mais longos são truncados. Isso é
# necessário porque redes neurais exigem entradas de dimensão fixa (tensor
# retangular de shape [batch_size, max_len]).
max_len = 200

# -----------------------------------------------------------------------------
# Carregamento e pré-processamento dos dados
# -----------------------------------------------------------------------------
# imdb.load_data carrega o dataset IMDB de avaliações de filmes, já pré-processado
# como sequências de inteiros (índices de palavras no vocabulário).
# num_words=vocab_size restringe ao vocabulário das N palavras mais frequentes.
# Retorna: ((X_train, y_train), (X_test, y_test))
#   X_train / X_test: listas de listas de inteiros (sequências de índices)
#   y_train / y_test: rótulos binários (0 = negativa, 1 = positiva)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# pad_sequences ajusta todas as sequences para o mesmo comprimento (max_len).
# padding="post": adiciona zeros no FINAL da sequência.
#   Exemplo: [1, 5, 3] com max_len=5 vira [1, 5, 3, 0, 0]
# Ciência da computação: transforma listas de comprimento variável em um tensor
# 2D de shape (n_amostras, max_len), compatível com operações de matriz (dot product).
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

print(f"Training DataShape: {X_train.shape}")
print(f"Test Data Shape: {X_train.shape}")

# -----------------------------------------------------------------------------
# Construção do modelo (Sequencial)
# -----------------------------------------------------------------------------
# Sequential empilha camadas linearmente; a saída de uma é a entrada da próxima.
model = Sequential([
    # -------------------------------------------------------------------------
    # Camada Embedding
    # -------------------------------------------------------------------------
    # input_dim=vocab_size: tamanho do vocabulário (número de palavras únicas + 1).
    # output_dim=128: dimensão do denso vetor de embedding (espaço latente).
    #
    # MATEMÁTICA:
    #   O Embedding é uma tabela de busca (lookup table) — uma matriz de pesos
    #   W_emb de shape (vocab_size, output_dim). Cada palavra i é representada
    #   por um vetor denso w_i = W_emb[i, :] ∈ R^128.
    #   Durante o treino, W_emb é aprendido via backpropagation para que palavras
    #   com significado similar fiquem próximas no espaço vetorial (similaridade
    #   do cosseno / distância euclidiana).
    #
    # CIÊNCIA DA COMPUTAÇÃO:
    #   Substitui a representação one-hot (esparsa, de tamanho vocabulário) por
    #   um vetor denso de baixa dimensão, reduzindo drasticamente a dimensionalidade
    #   e a esparsidade. Complexidade de consulta: O(1) por índice.
    Embedding(input_dim=vocab_size, output_dim=128),

    # -------------------------------------------------------------------------
    # Camada SimpleRNN
    # -------------------------------------------------------------------------
    # 128: número de unidades ocultas (hidden units) da RNN.
    # activation='tanh': função de ativação hiperbólica tangente.
    # return_sequences=False: retorna apenas o último estado oculto h_T (não a
    # sequência completa de estados), adequado para classificação de sequência inteira.
    #
    # MATEMÁTICA (célula SimpleRNN em cada passo de tempo t):
    #   h_t = tanh( W_xh · x_t + W_hh · h_{t-1} + b_h )
    #   onde:
    #     x_t ∈ R^128   : vetor de embedding da palavra no passo t
    #     h_{t-1} ∈ R^128: estado oculto anterior (inicializado em zeros)
    #     h_t   ∈ R^128 : novo estado oculto
    #     W_xh ∈ R^(128×128): pesos de entrada-oculto
    #     W_hh ∈ R^(128×128): pesos oculto-oculto (recorrentes)
    #     b_h  ∈ R^128      : bias
    #     tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z)) ∈ (-1, 1)
    #
    # Após processar todos os max_len passos de tempo, o estado final h_T carrega
    # uma representação resumida (encoding) de toda a sequência (review).
    #
    # CIÊNCIA DA COMPUTAÇÃO:
    #   RNNs processam dados sequenciais mantendo um estado oculto que é atualizado
    #   a cada passo. Isso permite capturar dependências temporais/contextuais entre
    #   palavras. A complexidade é O(T · H^2) por sequência, onde T = max_len e
    #   H = unidades ocultas. O problema do gradiente desvanecente (vanishing
    #   gradient) pode limitar a captura de dependências de longo prazo.
    SimpleRNN(128, activation='tanh', return_sequences=False),

    # -------------------------------------------------------------------------
    # Camada Dense (saída)
    # -------------------------------------------------------------------------
    # 1: um único neurônio de saída.
    # activation='sigmoid': mapeia o logit para (0, 1), interpretável como
    # probabilidade da classe positiva (avaliação positiva).
    #
    # MATEMÁTICA:
    #   z = W_dense · h_T + b_dense          (logit, escalar)
    #   y_hat = sigmoid(z) = 1 / (1 + e^(-z)) ∈ (0, 1)
    #   onde W_dense ∈ R^(1×128), b_dense ∈ R.
    #
    # CIÊNCIA DA COMPUTAÇÃO:
    #   Camada totalmente conectada (fully connected / linear) que projeta o estado
    #   oculto final para um único valor. Com sigmoid, a saída é uma probabilidade
    #   estimada P(classe=1 | entrada).
    Dense(1, activation='sigmoid')
])

# -----------------------------------------------------------------------------
# Compilação do modelo
# -----------------------------------------------------------------------------
# optimizer='adam': algoritmo de otimização adaptativo (Adaptive Moment
# Estimation) que combina momento (SGD com momentum) e RMSProp, ajustando a
# taxa de aprendizado por parâmetro com base em médias móveis dos gradientes
# (primeiro momento) e gradientes ao quadrado (segundo momento).
#
# loss='binary_crossentropy': entropia cruzada binária. Para um exemplo com
# rótulo y ∈ {0,1} e predição p = y_hat:
#   L = -[ y · log(p) + (1 - y) · log(1 - p) ]
#   MATEMÁTICA: penaliza predições confiantes e erradas. Quando y=1, L = -log(p);
#   quando y=0, L = -log(1-p). A derivada ∂L/∂z = p - y é simples e estável.
#
# metrics=['accuracy']: acurácia = (acertos) / (total), métrica de classificação
# padrão (não é usada para atualizar pesos, apenas para monitoramento).
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------------------------------------------------------
# Resumo da arquitetura
# -----------------------------------------------------------------------------
# model.summary() imprime a tabela de camadas, output shape e número de
# parâmetros treináveis de cada camada. Útil para verificar shapes e capacidade
# do modelo (quantidade de graus de liberdade).
model.summary()

# -----------------------------------------------------------------------------
# Treinamento
# -----------------------------------------------------------------------------
# epochs=5: número de passagens completas pelo conjunto de treino.
# batch_size=32: número de amostras por atualização de peso (mini-batch).
#   Ciência da computação: mini-batch oferece trade-off entre estabilidade do
#   gradiente (batch grande) e ruído benéfico/eficiência (batch pequeno).
# validation_split=0.2: 20% dos dados de treino são separados para validação
# (não usados para atualizar pesos), servindo para detectar overfitting.
#
# MATEMÁTICA do treino (descida do estocástico de gradiente):
#   Para cada batch: calcula-se L(θ) = mean(binary_crossentropy) sobre o batch,
#   calcula-se ∇_θ L via backpropagation (regra da cadeia) e atualiza-se
#   θ ← θ - lr · ∇_θ L (modificado pelo Adam).
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# -----------------------------------------------------------------------------
# Avaliação
# -----------------------------------------------------------------------------
# model.evaluate roda o modelo nos dados de teste (vistos nunca durante treino)
# e retorna a perda e as métricas compiladas. Fornece estimativa não tendenciosa
# do desempenho em dados novos (generalização).
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
