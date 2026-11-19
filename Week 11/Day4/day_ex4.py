import os

# =====================================================================
# 1. VARIÁVEIS DE AMBIENTE DO TENSORFLOW (configuração antes de importar)
# =====================================================================
# TF_CPP_MIN_LOG_LEVEL controla o nível de log do TensorFlow/C++.
#   0 = imprime tudo (default)
#   1 = desliga INFO
#   2 = desliga INFO e WARNING (recomendado para não poluir o console)
#   3 = exibe apenas erros críticos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# oneDNN (Intel oneAPI Deep Neural Network Library) acelera operações de
# ponto flutuante na CPU, mas pode reordenar operações e gerar pequenos
# desvios de precisão entre execuções. Desativar garante reprodutibilidade.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# =====================================================================
# 2. IMPORTAÇÕES
# =====================================================================
from tensorflow.keras.datasets import imdb          # Dataset IMDB (sentimento binário)
from tensorflow.keras.preprocessing.sequence import pad_sequences  # padding de sequências
import tensorflow as tf                             # framework de deep learning
from tensorflow.keras.models import Sequential      # modelo linear (pilha de camadas)
from tensorflow.keras.layers import (             # camadas de rede neural
    Embedding, SimpleRNN, LSTM, GRU, Dense
)

# =====================================================================
# 3. HIPERPARÂMETROS
# =====================================================================
# vocab_size: tamanho do vocabulário considerado (top 10.000 palavras mais
# frequentes do IMDB). Cada palavra vira um inteiro (índice). Palavras fora
# desse conjunto são mapeadas para 0 (descartadas/desconhecidas).
# Motivo: controlar a dimensionalidade da matriz de Embedding (vocab_size × dim).
vocab_size = 10000

# max_len: comprimento fixo de cada sequência (número de tokens por revisão).
# Sequências mais longas são truncadas; mais curtas recebem zeros no final.
# Deep learning exige entradas com shape fixo (tensor retangular), não listas
# de tamanhos variáveis.
max_len = 200

# =====================================================================
# 4. CARREGAMENTO DO DATASET IMDB
# =====================================================================
# O IMDB contém 50.000 revisões de filmes rotuladas:
#   - 25.000 para treino, 25.000 para teste
# Cada revisão já está pré-processada como lista de inteiros (índices de
# palavras no vocabulário). O rótulo é binário: 0 = negativo, 1 = positivo.
# load_data(num_words=vocab_size) mantém apenas as `vocab_size` palavras mais
# frequentes; o restante é substituído por 0.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# =====================================================================
# 5. PRÉ-PROCESSAMENTO: PADDING (IGUALAR COMPRIMENTO DAS SEQUÊNCIAS)
# =====================================================================
# pad_sequences transforma listas de tamanhos variáveis em uma matriz 2D
# de shape (n_amostras, max_len).
# padding='post': adiciona zeros no FINAL das sequências curtas.
# Ciência da computação: redes neurais operam sobre tensores (arrays
# multidimensionais) com dimensões fixas; não aceitam "ragged arrays"
# (linhas de comprimento diferente) sem tratamento especial.
# Matematicamente: mapeia o conjunto de sequências de comprimento variável
# para o espaço vetorial R^(max_len), onde cada posição é um token (inteiro).
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test  = pad_sequences(X_test,  maxlen=max_len, padding='post')

print(f"Training Data Shape: {X_train.shape}")  # (25000, 200)
print(f"Test Data Shape: {X_test.shape}")       # (25000, 200)

# =====================================================================
# 6. MODELO RNN SIMPLES (SimpleRNN / Vanilla RNN)
# =====================================================================
# Sequential: empilha camadas linearmente; a saída de uma camada é a entrada
# da próxima. Cada camada é um bloco computacional com pesos treináveis.
rnn_model = Sequential([
    # -----------------------------------------------------------------
    # Camada Embedding
    # -----------------------------------------------------------------
    # Converte cada índice de palavra (inteiro) em um vetor denso de 128
    # dimensões. Internamente é uma matriz de pesos W_emb ∈ R^(vocab_size × 128).
    # Para entrada i (índice da palavra), o vetor de saída é a i-ésima linha
    # de W_emb (operação de lookup/indexação).
    #
    # Motivo matemático: one-hot encoding gera vetores esparsos e de altíssima
    # dimensão (vocab_size = 10.000). O embedding projeta isso para um espaço
    # de baixa dimensão (128) e contínuo, onde palavras com significados
    # similares tendem a ficar próximas (mesotopia semântica aprendida).
    # Reduz a esparsidade e permite que a rede aprenda relações entre palavras.
    Embedding(input_dim=vocab_size, output_dim=128),

    # -----------------------------------------------------------------
    # Camada SimpleRNN
    # -----------------------------------------------------------------
    # Rede neural recorrente vanilla. Processa a sequência um token por vez,
    # mantendo um "estado oculto" h_t que carrega memória do passado.
    #
    # Equação por passo de tempo t:
    #     h_t = tanh( W_xh · x_t + W_hh · h_(t-1) + b_h )
    #
    #   x_t        : vetor de entrada no passo t (embedding da palavra t,
    #                dimensão 128)
    #   h_(t-1)    : estado oculto do passo anterior (dimensão 128)
    #   h_t        : novo estado oculto (dimensão 128)
    #   W_xh ∈ R^(128×128) : pesos que mapeiam entrada -> estado oculto
    #   W_hh ∈ R^(128×128) : pesos recorrentes (estado -> estado)
    #   b_h  ∈ R^128       : vetor de viés
    #   tanh     : ativação que comprime valores para o intervalo (-1, 1),
    #              evitando explosão de valores e introduzindo não-linearidade.
    #
    # return_sequences=False: retorna APENAS o último estado oculto h_T
    # (vetor fixo de 128 elementos), descartando os estados intermediários.
    # Isso transforma a sequência de comprimento variável em um vetor fixo,
    # adequado para classificação.
    #
    # Problema da SimpleRNN: gradiente desvanecente (vanishing gradient).
    # Ao aplicar a regra da cadeia através de muitos passos de tempo, o
    # gradiente se multiplica por W_hh repetidamente. Se |eigenvalues(W_hh)| < 1,
    # o gradiente decai exponencialmente e camadas iniciais deixam de aprender
    # dependências de longo prazo.
    SimpleRNN(128, activation='tanh', return_sequences=False),

    # -----------------------------------------------------------------
    # Camada Dense (saída)
    # -----------------------------------------------------------------
    # Camada totalmente conectada com 1 neurônio e ativação sigmóide.
    # Operação: z = W · h_T + b, onde W ∈ R^(1×128), b ∈ R^1.
    # Saída: p̂ = σ(z) = 1 / (1 + e^(-z))
    #
    # A sigmóide mapeia qualquer número real para o intervalo (0, 1), que é
    # interpretado como a probabilidade prevista de a revisão ser POSITIVA:
    #     p̂ = P(y = 1 | x)
    # Se p̂ >= 0.5, prevemos classe 1 (positivo); senão, classe 0 (negativo).
    Dense(1, activation='sigmoid')
])

# =====================================================================
# 7. COMPILAÇÃO (configuração do treinamento)
# =====================================================================
# optimizer='adam': Adaptive Moment Estimation.
#   Combina descida do gradiente com momento (primeiro momento m_t) e RMSprop
#   (segundo momento v_t), mantendo taxas de aprendizado individuais por
#   parâmetro.
#
#   Atualização por parâmetro θ:
#       m_t      = β1·m_(t-1) + (1-β1)·g_t          (média móvel do gradiente)
#       v_t      = β2·v_(t-1) + (1-β2)·g_t²         (média móvel do gradiente²)
#       m̂_t      = m_t / (1 - β1^t)                 (correção de viés)
#       v̂_t      = v_t / (1 - β2^t)
#       θ_(t+1)  = θ_t - η · m̂_t / (√v̂_t + ε)
#
#   onde g_t = ∇L(θ_t) é o gradiente da perda, η a taxa de aprendizado,
#   ε um pequeno constante para estabilidade numérica.
#
# loss='binary_crossentropy': entropia cruzada binária.
#   L(y, p̂) = -[ y·log(p̂) + (1 - y)·log(1 - p̂) ]
#
#   y ∈ {0, 1} é o rótulo verdadeiro; p̂ ∈ (0,1) a probabilidade prevista.
#   Se y=1 e p̂→1, perda → 0 (acertou com confiança). Se y=1 e p̂→0, perda → ∞.
#   Mede a "surpresa" do modelo: quão diferente a previsão está da verdade.
#   Do ponto de vista da teoria da informação, é a divergência de KL entre
#   a distribuição real e a prevista (mais a entropia da real).
#
# metrics=['accuracy']: fração de predições corretas, usando limiar 0.5.
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.summary()  # Imprime arquitetura e total de parâmetros treináveis.

# =====================================================================
# 8. TREINAMENTO (aprendizado dos pesos via backpropagation)
# =====================================================================
# fit(): ajusta os pesos do modelo para minimizar a perda.
#   epochs=5: o modelo passa 5 vezes por todo o conjunto de treino.
#   batch_size=32: descida do gradiente em mini-batches. A cada 32 amostras,
#                  calcula-se o gradiente médio e atualiza-se os pesos.
#                  Ciência da computação: trade-off entre precisão do gradiente
#                  (batch grande) e velocidade/regularização (batch pequeno).
#   validation_split=0.2: 20% dos dados de treino são separados para validação.
#                         São usados apenas para monitorar overfitting, NÃO
#                         afetam os pesos.
#
# Ajuste dos pesos usa retropropagação (backpropagation):
#   1. Forward pass: calcula predição e perda.
#   2. Backward pass: pela regra da cadeia, calcula ∂L/∂w para cada peso w,
#      propagando o erro da camada de saída de volta para a entrada.
#   3. O otimizador (Adam) atualiza os pesos na direção oposta ao gradiente.
rnn_history = rnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# =====================================================================
# 9. AVALIAÇÃO (dados de teste, sem atualizar pesos)
# =====================================================================
# evaluate(): roda forward pass nos dados de teste e reporta perda e métrica.
#   Não há backpropagation nem atualização de pesos. Mede a capacidade de
#   generalização do modelo (desempenho em dados nunca vistos).
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test)
print(f"RNN Test Loss: {rnn_loss:.4f}, Test Accuracy: {rnn_accuracy:.4f}")

# =====================================================================
# 10. MODELO LSTM (Long Short-Term Memory)
# =====================================================================
# LSTM é um tipo de RNN projetado para mitigar o gradiente desvanecente e
# o gradiente explodindo, permitindo aprender dependências de longo prazo.
# Em vez de um único estado oculto, mantém um "estado celular" c_t e um
# "estado oculto" h_t, regulados por três portões (gates) com ativação sigmóide:
#
#   i_t = σ(W_i x_t + U_i h_(t-1) + b_i)      portão de entrada (o que gravar)
#   f_t = σ(W_f x_t + U_f h_(t-1) + b_f)      portão de esquecimento (o que apagar)
#   o_t = σ(W_o x_t + U_o h_(t-1) + b_o)      portão de saída (o que expor)
#   ĉ_t = tanh(W_c x_t + U_c h_(t-1) + b_c)   candidato a novo conteúdo da célula
#
#   c_t = f_t · c_(t-1) + i_t · ĉ_t           atualização do estado celular
#   h_t = o_t · tanh(c_t)                     estado oculto de saída
#
#   · denota produto elemento-a-elemento (Hadamard product).
#
# Matemática/CS: o portão de esquecimento f_t decide quanto do estado celular
# anterior c_(t-1) é preservado. Se f_t ≈ 1, a informação flui quase inalterada
# através de muitos passos de tempo (caminho de gradiente "em rotação" curta),
# evitando que o gradiente desapareça. Isso permite que a rede lembre de fatos
# ocorridos há centenas de tokens atrás, ao contrário da SimpleRNN.
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

# =====================================================================
# 11. MODELO GRU (Gated Recurrent Unit)
# =====================================================================
# GRU é uma variante simplificada do LSTM. Combina os portões de esquecimento
# e entrada em um único "portão de atualização" z_t, e usa um "portão de
# reset" r_t para controlar quanto do estado anterior é esquecido ao calcular
# o candidato. Tem menos parâmetros que o LSTM (não tem estado celular c_t
# separado nem portão de saída o_t), o que costuma treinar mais rápido com
# desempenho similar em muitas tarefas.
gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    GRU(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_model.summary()

# Treinamento do GRU.
gru_history = gru_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Avaliação do GRU no teste.
gru_loss, gru_accuracy = gru_model.evaluate(X_test, y_test)
print(f"GRU Test Loss: {gru_loss:.4f}, Test Accuracy: {gru_accuracy:.4f}")