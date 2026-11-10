# =============================================================================
# IMPORTS E CONFIGURAÇÃO DO AMBIENTE
# =============================================================================
import os
# Desabilita logs informativos do TensorFlow (apenas erros críticos serão mostrados)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Desativa otimizações oneDNN para evitar avisos de compatibilidade em algumas CPUs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS (CIFAR-10)
# =============================================================================

# Carrega o dataset CIFAR-10.
# Ciência da computação: divide automaticamente em treino (50.000 imagens) e teste (10.000 imagens).
# Cada imagem é um tensor 3D (32, 32, 3): 32 pixels de altura, 32 de largura, 3 canais de cor (RGB).
# Os valores dos pixels estão no intervalo inteiro [0, 255].
# As labels são inteiros de 0 a 9 indicando a classe de cada imagem (avião, carro, pássaro, etc.).
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalização Min-Max dos pixels.
# Matemática: x' = x / 255.0, mapeando o intervalo [0, 255] para [0, 1].
# Ciência da computação: reduz a magnitude dos valores de entrada, evitando que os gradientes
# explodam durante a retropropagação e ajudando o otimizador a convergir mais rápido e estável.
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encoding dos rótulos.
# Matemática: converte um inteiro c em um vetor de tamanho 10 com 1 na posição c e 0 nas demais.
# Exemplo: classe 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
# Ciência da computação: necessário porque a camada de saída usa softmax, que produz uma
# distribuição de probabilidade sobre as classes (precisa de alvos no mesmo formato).
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Training Data Shape: {X_train.shape}, Label Shapes: {y_train.shape}")
print(f"Test Data Shape: {X_test.shape}, Label Shapes: {y_test.shape}")

# =============================================================================
# CONSTRUÇÃO DA ARQUITETURA CNN (REDE NEURAL CONVOLUCIONAL)
# =============================================================================

# Sequential: empilha camadas linearmente (grafo acíclico direto simples).
# A saída de cada camada é a entrada da próxima.
model = Sequential([
    # -------------------------------------------------------------------------
    # CAMADA 1: Conv2D (Convolução 2D)
    # -------------------------------------------------------------------------
    # Matemática: operação de correlação cruzada entre um filtro (kernel) K e a entrada X:
    #     (X * K)(i, j) = soma_m soma_n X(i+m, j+n) * K(m, n)
    # 32 filtros de tamanho 3x3, cada um varrendo toda a imagem de entrada (32x32x3).
    # Cada filtro aprende a detectar um padrão local (borda, textura, cor, etc.).
    # Parâmetros desta camada: 32 filtros * (3*3*3 pesos + 1 bias) = 896 parâmetros.
    # Ciência da computação: compartilhamento de parâmetros (mesmo filtro usado em toda a imagem)
    # e esparsidade de conexões (cada neurônio vê apenas um patch local), o que reduz drasticamente
    # o número de parâmetros em comparação com uma camada totalmente conectada.
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    
    # -------------------------------------------------------------------------
    # CAMADA 2: MaxPooling2D (Amostragem para baixo / Pooling)
    # -------------------------------------------------------------------------
    # Matemática: em janelas de 2x2, retém apenas o valor máximo.
    # Reduz a resolução espacial de 32x32 para 16x16.
    # Ciência da computação: diminui a quantidade de dados e computação nas camadas seguintes,
    # e fornece certa invariância à translação (pequenos deslocamentos do objeto não alteram
    # drasticamente o resultado).
    MaxPooling2D((2, 2)),
    
    # -------------------------------------------------------------------------
    # CAMADA 3: Conv2D (segunda convolução)
    # -------------------------------------------------------------------------
    # 64 filtros de 3x3 aplicados sobre os 32 feature maps da camada anterior.
    # Cada filtro tem 3*3*32 + 1 = 289 parâmetros. Total: 64 * 289 = 18.496 parâmetros.
    # Aprende padrões mais complexos compostos pelos padrões simples da camada anterior.
    Conv2D(64, (3,3), activation='relu'),
    
    # -------------------------------------------------------------------------
    # CAMADA 4: MaxPooling2D
    # -------------------------------------------------------------------------
    # Reduz novamente a resolução espacial de 16x16 para 8x8.
    MaxPooling2D(2, 2),
    
    # -------------------------------------------------------------------------
    # CAMADA 5: Flatten
    # -------------------------------------------------------------------------
    # Ciência da computação: achata o tensor tridimensional (8, 8, 64) em um vetor unidimensional
    # de tamanho 8*8*64 = 4096, para que possa ser alimentado nas camadas Dense (fully connected).
    Flatten(),
    
    # -------------------------------------------------------------------------
    # CAMADA 6: Dense (camada totalmente conectada / camada densa)
    # -------------------------------------------------------------------------
    # Matemática: z = W*x + b, seguido de ativação ReLU.
    # Cada um dos 128 neurônios está conectado a todos os 4096 entradas do flatten.
    # Parâmetros: 4096 * 128 + 128 = 524.416 parâmetros.
    # Ciência da computação: combina características globais extraídas pelas camadas convolucionais.
    Dense(128, activation='relu'),
    
    # -------------------------------------------------------------------------
    # CAMADA 7: Dropout
    # -------------------------------------------------------------------------
    # Matemática: durante o treino, cada neurônio é "desligado" (saída forçada a zero) com
    # probabilidade 0.5, independente a cada batch.
    # Ciência da computação: técnica de regularização que impede co-adaptação de neurônios
    # (overfitting), forçando a rede a aprender representações mais robustas. Approxima um
    # ensemble de muitas redes menores. Na inferência (teste), todos os neurônios são usados
    # com seus pesos escalados pelo fator de retenção.
    Dropout(0.5),
    
    # -------------------------------------------------------------------------
    # CAMADA 8: Dense de saída + softmax
    # -------------------------------------------------------------------------
    # Matemática: z = W*x + b produz 10 logits (um por classe).
    # Aplica softmax: softmax(z_i) = exp(z_i) / soma_j(exp(z_j)), para i = 1..10.
    # Resultado: um vetor de 10 probabilidades que somam 1, representando a confiança
    # do modelo em cada classe. A classe prevista é argmax_i(softmax(z_i)).
    Dense(10, activation='softmax')
])

# NOTA: o modelo abaixo redefine exatamente o mesmo modelo, sobrescrevendo o anterior.
# Isso é redundância desnecessária; apenas a segunda definição é usada no treino.
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)), 
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Exibe um resumo da arquitetura: camadas, formatos de saída e número de parâmetros.
model.summary()

# =============================================================================
# COMPILAÇÃO DO MODELO
# =============================================================================

# Configura o processo de aprendizado.
model.compile(
    # Otimizador Adam (Adaptive Moment Estimation).
    # Matemática: variante do Descida do Gradiente Estocástico que calcula médias móveis
    # dos gradientes (primeiro momento m) e dos gradientes ao quadrado (segundo momento v),
    # ajustando o learning rate individualmente para cada parâmetro:
    #     theta_{t+1} = theta_t - eta * m_hat_t / (sqrt(v_hat_t) + epsilon)
    # Ciência da computação: converge rapidamente e requer pouco ajuste manual de learning rate.
    optimizer='adam',
    
    # Função de perda: Entropia Cruzada Categórica.
    # Matemática: L = - soma_i (y_i * log(y_hat_i)), onde y é one-hot (apenas a classe verdadeira
    # tem y_i = 1). Penaliza fortemente quando o modelo atribui baixa probabilidade à classe correta.
    # Ciência da computação: função padrão para problemas de classificação multi-classe com
    # saída softmax.
    loss='categorical_crossentropy',
    
    # Métrica monitorada durante o treino (não afeta os pesos, apenas informa ao usuário).
    metrics=['accuracy']
)

# =============================================================================
# TREINAMENTO (APRENDIZAGEM SUPERVISIONADA)
# =============================================================================

# history guarda as métricas de treino e validação por época.
# Ciência da computação: Backpropagation + otimizador ajustam os pesos para minimizar a perda.
history = model.fit(
    X_train, y_train,
    # epochs: número de passadas completas sobre todo o conjunto de treino.
    epochs=10,
    # batch_size: número de amostras processadas antes de atualizar os pesos (mini-batch SGD).
    # Compensa entre ruído do gradiente (batch pequeno) e eficiência de memória/GPU (batch grande).
    batch_size=64,
    # validation_split: 20% dos dados de treino são separados para validação (não usados nos
    # pesos, servem para detectar overfitting sem tocar no conjunto de teste).
    validation_split=0.2
)

# =============================================================================
# AVALIAÇÃO FINAL EM DADOS DE TESTE (GENERALIZAÇÃO)
# =============================================================================

# Avalia o modelo em X_test (dados nunca vistos durante o treino).
# test_loss: perda média no teste. test_accuracy: fração de previsões corretas.
test_loss, test_accuracy= model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# =============================================================================
# VISUALIZAÇÃO DE CURVAS DE APRENDIZAGEM
# =============================================================================
import matplotlib.pyplot as plt

# Gráfico de acurácia.
plt.plot(history.history['accuracy'], label="")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Gráfico de perda (loss).
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Logs")
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()