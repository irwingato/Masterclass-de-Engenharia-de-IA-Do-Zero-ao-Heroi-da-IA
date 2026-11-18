import os

# Suprime logs do TensorFlow/C++ (INFO e WARNING) para não poluir o console.
# TF_CPP_MIN_LOG_LEVEL: 0=tudo, 1=INFO desligado, 2=INFO+WARNING desligado, 3=erro apenas.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Desativa otimizações oneDNN do TensorFlow para evitar mensagens de aviso sobre
# diferenças de precisão ponto-flutuante entre ordens de computação.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

# ===================== Hiperparâmetros =====================
# vocab_size: tamanho do vocabulário (número de palavras mais frequentes do dataset IMDB).
# Cada palavra é representada por um inteiro (índice). Palavras fora desse conjunto são ignoradas.
vocab_size = 10000

# max_len: comprimento fixo de cada sequência (número de tokens por revisão).
# Sequências maiores são cortadas, menores são preenchidas com zeros (padding).
max_len = 200

# ===================== Carregamento do dataset =====================
# IMDB: dataset de análise de sentimento (binário: positivo/negativo).
# Retorna listas de inteiros (índices de palavras) e rótulos 0/1.
# split padrão: 25.000 treino, 25.000 teste.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# ===================== Padding (pré-processamento) =====================
# pad_sequences: iguala o comprimento de todas as sequências para max_len.
# padding='post': adiciona zeros no FINAL da sequência curta.
# Máquina de aprendizado exige entradas com shape fixo (matriz/tensor retangular).
# Matematicamente: mapeia sequências de comprimento variável em R^(max_len).
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

# ===================== Modelo RNN Simples =====================
# Sequential: empilha camadas linearmente; saída de uma é entrada da próxima.
rnn_model = Sequential([
    # Embedding: converte cada índice de palavra em um denso vetor de tamanho 128.
    # Matriz de pesos W_emb ∈ R^(vocab_size × 128). Look-up: w_i = W_emb[i].
    # Reduz esparsidade (one-hot) para representação densa e contínua no espaço semântico.
    Embedding(input_dim=vocab_size, output_dim=128),

    # SimpleRNN: rede neural recorrente vanilla.
    # Equação por passo de tempo t:
    #   h_t = tanh(W_xh · x_t + W_hh · h_(t-1) + b_h)
    # onde h_t é o estado oculto, x_t a entrada (embedding da palavra t),
    # W_xh, W_hh matrizes de pesos, b_h viés, tanh ativação.
    # return_sequences=False: retorna apenas o último estado oculto h_T (vetor fixo).
    SimpleRNN(128, activation='tanh', return_sequences=False),

    # Dense(1, sigmoid): camada totalmente conectada com 1 neurônio e ativação sigmóide.
    # σ(z) = 1 / (1 + e^(-z)), mapeia qualquer real para (0, 1) — interpretação como probabilidade.
    # Saída: p(y=1 | x) (probabilidade de ser revisão positiva).
    Dense(1, activation='sigmoid')
])

# ===================== Compilação =====================
# optimizer='adam': algoritmo de descida do gradiente estocástico adaptativo.
# Combina momento (primeiro momento do gradiente) e RMSprop (segundo momento),
# com taxas de aprendizado individuais por parâmetro. Atualização:
#   θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
# loss='binary_crossentropy': entropia cruzada binária.
#   L = -[ y·log(p̂) + (1-y)·log(1-p̂) ]
# Mede diferença entre rótulo real y ∈ {0,1} e probabilidade prevista p̂.
# metrics=['accuracy']: fração de classificações corretas (limiar 0.5).
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.summary()  # Imprime arquitetura e número de parâmetros treináveis.

# ===================== Treinamento =====================
# fit: ajusta pesos via retropropagação (backpropagation) + otimizador.
# epochs=5: 5 passadas completas sobre o treino.
# batch_size=32: atualiza pesos a cada 32 amostras (descida do gradiente em mini-batches).
# validation_split=0.2: 20% do treino reservado para validação (não usado nos pesos).
rnn_history = rnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# ===================== Avaliação =====================
# evaluate: calcula loss e métricas nos dados de teste (sem atualizar pesos).
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test)
print(f"RNN Test Loss: {rnn_loss:.4f}, Test Accuracy: {rnn_accuracy:.4f}")

# ===================== Modelo LSTM =====================
# LSTM (Long Short-Term Memory): RNN com portões para mitigar gradiente desvanecente/explodindo.
# Estrutura (por passo t):
#   i_t = σ(W_i x_t + U_i h_(t-1) + b_i)      portão de entrada
#   f_t = σ(W_f x_t + U_f h_(t-1) + b_f)      portão de esquecimento
#   o_t = σ(W_o x_t + U_o h_(t-1) + b_o)      portão de saída
#   ĉ_t = tanh(W_c x_t + U_c h_(t-1) + b_c)   candidato à memória
#   c_t = f_t · c_(t-1) + i_t · ĉ_t           estado celular atualizado
#   h_t = o_t · tanh(c_t)                     estado oculto
# O produto elemento-a-elemento permite reter/esquecer informação por longas sequências.
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

# Treinamento do LSTM (mesmo procedimento do RNN).
lstm_history = lstm_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Avaliação do LSTM no teste.
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Test Loss: {lstm_loss:.4f}, Test Accuracy: {lstm_accuracy:.4f}")