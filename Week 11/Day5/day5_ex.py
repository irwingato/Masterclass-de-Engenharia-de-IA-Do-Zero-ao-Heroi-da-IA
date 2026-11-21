import os

# Desativa as mensagens de log do TensorFlow/absl antes de inicializá-lo:
# - TF_CPP_MIN_LOG_LEVEL=3: apenas mensagens de severidade FATAL são registradas no stderr,
#   suprimindo as warnings "All log messages before absl::InitializeLog()...".
# - TF_ENABLE_ONEDNN_OPTS=0: desativa as operações customizadas oneDNN (os avisos de INFO do
#   port.cc "oneDNN custom operations are on" só aparecem quando elas estão ligadas).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Modelo de classificação de sentimento de filmes (IMDB): sequências de ids de palavras
# -> Embedding (vetores GloVe) -> LSTM -> classificador sigmóide binário.
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer  # importado, mas não usado neste script
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# vocab_size: tamanho do vocabulário - imdb.load_data(num_words=vocab_size) mantém as
# palavras de maior frequência (id 1 = 'the'); palavras com id ≥ vocab_size são removidas
# das sequências.
vocab_size = 10000

# max_len: comprimento fixo (número de timesteps) para o qual todas as sequências serão
# ajustadas: sequências longas são truncadas, curtas são preenchidas.
max_len = 200

# imdb.load_data: devolve (X_train, y_train) e (X_test, y_test) - 25.000 reviews de treino
# e 25.000 de teste, já pré-processados pelo Keras:
# CS/NLP: cada review é representada como uma lista de ints (um id por palavra, ids por
# frequência); y: 0 = review negativa, 1 = review positiva (classificação binária).
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# word_index: dicionário {palavra: id inteiro} do vocabulário imdb - ids 1..88584,
# 'the'=1, 'and'=2, 'a'=3, 'of'=4, 'to'=5, 'is'=6, 'br'=7.
word_index = imdb.get_word_index()

# reverse_word_index: inverte o dicionário para {id: palavra} - usado para decodificar os ids.
reverse_word_index = {value: key for key, value in word_index.items()}

# Decodifica os primeiros 5 reviews de treino de volta para texto (para visualização -
# variável não é usada depois neste script).
# CS: os id armazenados nas sequências do imdb estão deslocados em relação aos id do
# vocabulário; `i - 3` realiza esse deslocamento para recuperar o id da palavra; ids sem
# correspondência mapeiam para '?' (ex.: os primeiros tokens da primeira review decodificam
# como '? this film was just brilliant casting location scenery story').
decoded_reviews = [" ".join([reverse_word_index.get(i - 3, "?") for i in review]) for review in X_train[:5]]

# CS/ML: tensores precisam de comprimento fixo: X_train/X_test têm forma (n_reviews, 200) -
# sequences longas são truncadas para max_len; curtas são preenchidas com zeros
# (padding="post": preenche no fim da sequência; value=0.0 default - id 0 reservado para padding).
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# GloVe (Penning et al., 2014): vetores de embedding pré-treinados em Common Crawl
# (vocabulário de 400.000 palavras - cabeçalho do arquivo glove.6B.100d: "400000 100").
# embeddings_index: dicionário {palavra: vetor float32 de 100 dimensões} lido do arquivo.
embeddings_index = {}
glove_file = None
for name in ("glove.6B.100D.txt", "glove.6B.100d"):
    if os.path.exists(name):
        glove_file = name
        break
    if "__file__" in globals():
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if os.path.exists(candidate):
            glove_file = candidate
            break
if glove_file is None:
    glove_file = "glove.6B.100D.txt"
with open(glove_file, "r", encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

print(f"Loaded {len(embeddings_index)} word vectors")

# CS/ML: tabela de embedding usada pela camada Embedding:
# embedding_matrix[i] = vetor GloVe da palavra com id i no vocabulário (linha i). Palavras
# do vocabulário imdb que não têm vetor GloVe (embeddings_index.get → None) mantêm a linha
# inicial np.zeros((vocab_size, 100)).
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if i < vocab_size:
        embeddings_vector = embeddings_index.get(word)
        if embeddings_vector is not None:
            embedding_matrix[i] = embeddings_vector

# LSTM com vetores GloVe (transfer learning):
# - Embedding(input_dim=vocab_size=10000, output_dim=embedding_dim=100, weights=[embedding_matrix],
#   trainable=False): busca por id - cada timestep retorna o vetor denso (representação
#   densa em vez de one-hot). trainable=False congela os vetores: GloVe já é pré-treinado.
# - LSTM(128, activation='tanh', return_sequences=False): camada recorrente que processa a
#   sequência e devolve o estado oculto final h_T (vetor de dimensão 128).
#   Math (Hochreiter & Schmidhuber, 1997):
#     i_t = σ(W_i x_t + U_i h_{t-1} + b_i)      # porta de entrada
#     f_t = σ(W_f x_t + U_f h_{t-1} + b_f)      # porta de esquecimento
#     o_t = σ(W_o x_t + U_o h_{t-1} + b_o)      # porta de saída
#     g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)   # candidato a estado
#     C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t           # estado da célula (memória longa)
#     h_t = o_t ⊙ tanh(C_t)                     # estado oculto (saída do timestep t)
#   σ = sigmoide; ⊙ = produto elementwise. tanh: (-1, 1); sigmoide: (0, 1).
# - Dense(1, activation='sigmoid'): z = W·h_T + b (transformação linear) e
#   σ(z) = 1/(1 + e^(-z)) ∈ (0, 1) - a probabilidade estimada de a review ser positiva.
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Compilação - define como o modelo será treinado:
# - optimizer='adam': otimizador com estimativas adaptativas de momentos - Math (Kingma & Ba, 2015):
#     m_t = β1·m_{t-1} + (1-β1)·g_t                        # primeiro momento
#     v_t = β2·v_{t-1} + (1-β2)·g_t²                       # segundo momento
#     m̂_t = m_t/(1 - β1^t), v̂_t = v_t/(1 - β2^t)          # correção de viés (m_0 = v_0 = 0)
#     θ_t = θ_{t-1} - η·m̂_t/(√v̂_t + ε)
#   (β1=0.9, β2=0.999, ε=1e-7, η=0.001 - padrões do Keras).
# - loss='binary_crossentropy': para y ∈ {0, 1}, por amostra:
#   L = -(y·log(ŷ) + (1-y)·log(1-ŷ)); perda média do lote. Quando ŷ ≈ y, L ≈ 0; erros
#   grandes geram L grande.
# - metrics=['accuracy']: fração das previsões (ŷ > 0.5) == y.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Treinamento - Math/CS: gradiente descendente: a cada lote de 64 amostras (batch_size), os
# parâmetros θ são atualizados pelo Adam (gradientes computados de volta pela rede -
# backpropagation). validation_split=0.2 reserva 20% dos dados de treino para medir o
# desempenho em dados não vistos (evita overfitting); históricos de treino/validação em `history`.
# epochs=5: 5 passadas completas sobre os dados de treino. verbose=1: barra de progresso.
history = model.fit(
    X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1
)

# Avaliação em dados de teste - loss e accuracy via model.evaluate.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"LSTM modle with GloVe Test Accuracy: {accuracy:.4f}")

# Modelo baselin: LSTM sem GloVe - Embedding(input_dim=vocab_size, output_dim=128): tabela
# de embedding nova, inicializada aleatoriamente e treinável (sem weights= e sem
# trainable=False): linha i inicializada com um vetor aleatório (glorot uniform) que a
# rede aprende durante o treinamento, em vez de vetores GloVe pré-treinados.
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

lstm_history = lstm_model.fit(
    X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)
print(f"LSTM without GloVe Test Accurracy: {lstm_loss:.4f}")
print(f"LSTM modle with GloVe Test Accuracy: {accuracy:.4f}")

import matplotlib.pyplot as plt

# Comparação gráfica: 'LSTM' = acurácia do modelo baselin sem GloVe (lstm_accuracy);
# 'LSTM GloVe' = acurácia do modelo com embeddings GloVe (accuracy). plt.show() exibe a janela.
models = ['LSTM', 'LSTM GloVe']
accuracies = [lstm_accuracy, accuracy]
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('Comparison of LSTM with and without word embeddings')
plt.ylabel('Accuracy')
plt.show()
