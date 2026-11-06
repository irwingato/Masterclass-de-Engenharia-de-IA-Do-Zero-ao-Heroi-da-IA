# =============================================================================
# IMPORTS E CONFIGURAÇÃO DO AMBIENTE
# =============================================================================

import os
# Suprime logs de informação/warning do TensorFlow (nível 3 = apenas erros).
# Isso remove mensagens como "oneDNN custom operations are on" e logs de
# inicialização da biblioteca absl, que são ruído visual e não afetam o treino.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Desativa otimizações oneDNN do TensorFlow para evitar pequenas diferenças
# numéricas devido à ordem de operações de ponto flutuante (reprodutibilidade).
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
# Nota: o professor usa "from tensorflow.keras.models import to_categorical"
# (funciona na versão dele). Estou usando Python 3.12 e 3.13, onde essa
# importação está desatualizada e gera ImportError. No Python 3.12/3.13,
# to_categorical está em keras.utils (mudança de localização entre versões
# do Keras/TensorFlow, quebra de compatibilidade de API).
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# =============================================================================
# CARREGAMENTO DO DATASET CIFAR-10
# =============================================================================

# CIFAR-10: conjunto de 60.000 imagens coloridas 32x32 pixels em 10 classes
# (avião, automóvel, pássaro, gato, veado, cão, sapo, cavalo, navio, caminhão).
# 50.000 para treino, 10.000 para teste. Cada imagem é um tensor (32, 32, 3)
# com canais RGB (3 canais: vermelho, verde, azul), valores inteiros 0-255.
# X = features (pixels das imagens), y = rótulos (classe inteira 0-9).
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# =============================================================================
# PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================

# Normalização de pixels para o intervalo [0, 1].
# Motivo matemático: redes neurais convergem mais rápido e de forma mais estável
# quando as entradas estão em uma escala pequena e semelhante. Valores brutos
# 0-255 causam gradientes grandes e instáveis na retropropagação.
# Divisão por 255.0 (máximo valor possível de um byte de cor) mapeia cada pixel
# para x' = x / 255, de modo que preto = 0.0 e branco = 1.0.
# astype('float32') converte inteiros para ponto flutuante de 32 bits, necessário
# para as operações de álgebra linear do TensorFlow (evita overflow e perde precisão
# desnecessária com float64).
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encoding dos rótulos.
# Transforma um inteiro (ex: classe 3) em um vetor binário de tamanho 10 com 1
# na posição da classe e 0 nas demais (ex: [0,0,0,1,0,0,0,0,0,0]).
# Motivo matemático/CS: a saída da rede será um vetor de probabilidades (softmax)
# com 10 neurônios. A perda categorical_crossentropy espera vetores de probabilidade
# alvo (distribuição de probabilidade one-hot), não índices inteiros.
# Se y = 3 e num_classes = 10, to_categorical produz e_3 (vetor base canônico).
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# =============================================================================
# ARQUITETURA DO MODELO (BASELINE)
# =============================================================================

# Sequential: modelo pilha de camadas (linear stack). Cada camada recebe a saída
# da anterior como entrada. Topologia feed-forward (sem conexões recorrentes).
model = Sequential([
    # Conv2D: camada convolucional 2D.
    # Aplica 32 filtros (núcleos) de tamanho 3x3 sobre a imagem de entrada.
    # Operação matemática: correlação (convolução disfarçada) entre o filtro W
    # e uma janela da imagem X. Para cada posição (i,j):
    #   S(i,j) = sum_{m,n} X(i+m, j+n) * W(m,n) + b
    # Produz um feature map (mapa de ativação) que destaca padrões locais
    # (bordas, texturas, cores). input_shape=(32,32,3): altura, largura, canais.
    # activation='relu': função de ativação ReLU (Rectified Linear Unit):
    #   f(x) = max(0, x). Introduz não-linearidade (sem ela, a rede seria apenas
    #   uma combinação linear de camadas, equivalente a uma única camada linear).
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    # MaxPooling2D: pooling máximo 2x2. Reduz a resolução espacial pela metade
    # (downsampling). Para cada janela 2x2, mantém apenas o valor máximo.
    # Motivo CS: redução de dimensionalidade, diminuição do número de parâmetros,
    # controle de overfitting e invariância a pequenas translações do objeto na imagem.
    MaxPooling2D((2, 2)),
    # Segunda camada convolucional com 64 filtros 3x3. Detecta padrões mais
    # complexos (combinações de bordas -> formas) nos feature maps da camada anterior.
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # Flatten: transforma o tensor 3D (altura, largura, canais) em um vetor 1D.
    # Necessário para conectar camadas convolucionais a camadas densas (fully connected),
    # que esperam entrada vetorial. Não tem parâmetros, apenas rearranja os dados.
    Flatten(),
    # Dense: camada totalmente conectada (fully connected / linear layer).
    # Cada neurônio está conectado a todas as saídas da camada anterior.
    # Operação: y = relu(W @ x + b), onde W é matriz de pesos (128 x n_entradas).
    # Aprende combinações não-lineares de features globais para classificação.
    Dense(128, activation='relu'),
    # Dropout(0.5): durante o treino, desativa aleatoriamente 50% dos neurônios
    # desta camada a cada passo (batch). Motivo CS: regularização que impede
    # co-adaptação de neurônios (overfitting), forçando a rede a aprender
    # representações robustas e redundantes. No teste, todos os neurônios estão ativos
    # e as saídas são escaladas (inverted dropout).
    Dropout(0.5),
    # Camada de saída com 10 neurônios (uma por classe) e ativação softmax.
    # Softmax: σ(z)_i = exp(z_i) / sum_j exp(z_j). Transforma logits (saídas brutas)
    # em uma distribuição de probabilidade sobre as 10 classes (soma = 1, cada termo >= 0).
    # A classe predita é argmax_i σ(z)_i.
    Dense(10, activation='softmax')
])

# =============================================================================
# COMPILAÇÃO DO MODELO
# =============================================================================

# Configura o processo de aprendizado:
# - optimizer='adam': Adaptive Moment Estimation. Otimizador de primeira ordem que
#   adapta a taxa de aprendizado para cada parâmetro usando médias móveis dos
#   gradientes (m) e dos gradientes ao quadrado (v):
#     m_t = beta1*m_{t-1} + (1-beta1)*g_t
#     v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
#     theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
#   Combina vantagens de RMSProp e momentum; bom padrão para redes profundas.
# - loss='categorical_crossentropy': entropia cruzada categórica.
#   Para uma amostra com alvo one-hot y e predição p (softmax):
#     L = - sum_i y_i * log(p_i)
#   Como y é one-hot, reduz a -log(p_classe_correta). Penaliza fortemente predições
#   confiantes e erradas. Minimizar isso equivale à estimativa de máxima verossimilhança.
# - metrics=['accuracy']: fração de amostras classificadas corretamente
#   (predição == classe verdadeira). Métrica de avaliação, não usada no gradiente.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Exibe resumo da arquitetura: camadas, formatos de saída e número de parâmetros
# treináveis (pesos + bias) de cada camada. Útil para debug e dimensionamento.
model.summary()

# =============================================================================
# TREINAMENTO DO MODELO BASELINE
# =============================================================================

# model.fit executa o loop de treinamento (descida do gradiente estocástica em mini-batches).
# Para cada época:
#   1. Embaralha os dados de treino.
#   2. Divide em batches de 64 amostras.
#   3. Para cada batch: forward pass -> calcula perda -> backward pass (retropropagação)
#      calcula gradientes dL/dtheta via regra da cadeia -> atualiza pesos com Adam.
# - validation_split=0.2: reserva 20% dos dados de treino como validação (não usados
#   para atualizar pesos, apenas para monitorar generalização a cada época).
# - epochs=10: número de passos completos sobre o dataset de treino.
# - batch_size=64: tamanho do mini-batch. Trade-off: batches pequenos = ruído de gradiente
#   (pode ajudar a escapar de mínimos locais) e menos memória; batches grandes = gradiente
#   mais estável e aproveitamento de paralelismo (GPU/TPU), mas mais memória.
# - verbose=1: barra de progresso por época.
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    verbose=1
)

# =============================================================================
# AVALIAÇÃO DO MODELO BASELINE
# =============================================================================

# model.evaluate roda forward pass nos dados de teste (sem atualizar pesos)
# e retorna a perda e a acurácia. verbose=0 suprime o progress bar.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Baseline Model Test Accuracy: {accuracy:.4f}")

# =============================================================================
# MODELO MELHORADO (ARQUITETURA MAIS PROFUNDA / MAIS PARÂMETROS)
# =============================================================================

# Mesmo padrão Sequential, mas com mais filtros (64, 128) e kernels maiores (5x5).
# Mais filtros = maior capacidade de representar features complexos (maior capacidade
# do modelo, risco maior de overfitting se dados forem insuficientes).
# Kernel 5x5 = campo receptivo maior por camada (enxerga região maior da imagem de uma vez).
improved_model = Sequential([
    Conv2D(64, (5, 5), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Otimizador Adam instanciado explicitamente com learning_rate=0.001 (taxa de aprendizado).
# learning rate controla o tamanho do passo na atualização dos pesos:
#   theta = theta - lr * gradiente_normalizado
# Muito alto -> pode divergir (pular o mínimo); muito baixo -> treino lento e pode
# ficar preso em mínimos locais/plateaus.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
improved_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Treina por 20 épocas (o dobro do baseline), permitindo que o modelo maior converja.
improved_history = improved_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    verbose=1
)

# =============================================================================
# AVALIAÇÃO DO MODELO MELHORADO
# =============================================================================

improved_loss, improved_accuracy = improved_model.evaluate(X_test, y_test, verbose=0)
print(f"Improved Model Test Acurracy: {improved_accuracy:.4f}")

# =============================================================================
# VISUALIZAÇÃO DOS RESULTADOS (MATPLOTLIB)
# =============================================================================

# Plot: acurácia de treino e validação por época.
# improved_history.history é um dicionário com listas de métricas por época.
# Chaves: 'accuracy' (treino), 'val_accuracy' (validação), 'loss', 'val_loss'.
# Motivo do erro anterior (KeyError: 'value'): a chave correta do Keras é
# 'val_accuracy', não 'value'. 'value' não existe no dicionário history,
# então o Python levanta KeyError ao tentar acessá-lo.
plt.plot(improved_history.history['accuracy'], label="Training Accuracy")
plt.plot(improved_history.history['val_accuracy'], label="Validation Accuracy")
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot: perda (loss) de treino e validação por época.
# Análise de overfitting: se loss de treino cai mas loss de validação sobe,
# o modelo está memorizando os dados de treino (overfitting).
plt.plot(improved_history.history['loss'], label="Training Loss")
plt.plot(improved_history.history['val_loss'], label="Validation Loss")
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')  # correto:Loss, não Accuracy (erro de cópia no rótulo do eixo Y)
plt.legend()
plt.grid()
plt.show()