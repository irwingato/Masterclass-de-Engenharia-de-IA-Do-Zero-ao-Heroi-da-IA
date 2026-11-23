from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU

# Imports do Keras/TensorFlow usados no projeto:
# imdb -> dataset de reviews de filmes; pad_sequences -> padronizacao de sequencias;
# Sequential -> modelo com camadas empilhadas; Embedding/RNN/Dense/LSTM/GRU -> camadas do modelo

# Load the dataset
vocab_size = 10000  # Tamanho do vocabulario: mantem apenas as 10.000 palavras mais frequentes do corpus IMDB
max_len = 100  # Tamanho maximo de sequencia: limita cada review a 100 tokens (palavras)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)
# X_train/X_test -> sequencias de indices de palavras (features)
# y_train/y_test -> rotulos binarios (0 = negativa, 1 = positiva)

# Pad Sequences
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_train, maxlen=max_len, padding="post")
# pad_sequences padroniza o tamanho de todas as sequencias para max_len (largura fixa)
# padding="post" adiciona tokens de zero (padding) ao final -> o RNN processa as palavras reais primeiro

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")
# Exibe as formas: X_train -> (amostras, 100), y_train -> (amostras,)

# Define the RNN model
rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    # Embedding: converte cada indice de palavra em um vetor denso de 128 dimensoes (word embedding).
    # Na pratica, comporta-se como uma matriz de shape (vocab_size, 128) com pesos aprendidos.
    SimpleRNN(128, activation='tanh'),
    # SimpleRNN: calcula estados ocultos repetidos h_t = tanh(W h_{t-1} + U x_t + b).
    # tanh mantem os valores em [-1, 1], evitando explosoes de gradiente.
    Dense(1, activation='sigmoid')
])

# Compile the model
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Adam: otimizador com taxa de aprendizado adaptativa (momentos m_t e v_t).
# binary_crossentropy: -(y*log(p) + (1-y)*log(1-p)) -> perda para classificacao binaria.
# metrics=['accuracy']: accuracy = (VP + VN) / total -> proporcao de acertos.

# Display the model summary
rnn_model.summary()

# Define the LSTM Model
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    # LSTM (Long Short-Term Memory): usa gates de input, forget e output para controlar o estado celular,
    # permitindo aprender dependencias de longo prazo em sequencias.
    LSTM(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
lstm_model.summary()

# Define the GRU Model
gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    # GRU (Gated Recurrent Unit): unifica os gates de update e reset em uma porta,
    # funcionando como uma LSTM simplificada com menos parametros.
    GRU(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Compile the model
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
gru_model.summary()

# Train RNN model
history_rnn = rnn_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)
# Treina com otimizacao (Adam) e backpropagation through time (BPTT) para minimizar a loss.

# Train LSTM model
history_lstm = lstm_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)

# Train GRU model
history_gru = gru_model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)

# Evalaute models
loss_rnn, accuracy_rnn = rnn_model.evaluate(X_test, y_test, verbose=0)
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)
loss_gru, accuracy_gru = gru_model.evaluate(X_test, y_test, verbose=0)
# evaluate calcula loss e metricas nos dados de teste (holdout).

print(f"RNN Test Accuracy: {accuracy_rnn:.4f}")
print(f"LSTM Test Accuracy: {accuracy_lstm:.4f}")
print(f"GRU Test Accuracy: {accuracy_gru:.4f}")
# :.4f formata o valor com 4 casas decimais.

import matplotlib.pyplot as plt

# Plot training accuracy
plt.plot(history_rnn.history['accuracy'], label="RNN Traning Accuracy")
plt.plot(history_lstm.history['accuracy'], label="LSTM Traning Accuracy")
plt.plot(history_gru.history['accuracy'], label="GRU Traning Accuracy")
plt.title("Training Accuracy Comparsion")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Plota a evolucao da accuracy por epoca; history_* armazena as metricas de treino por epoca.