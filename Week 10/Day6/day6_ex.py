import os
# Desativa logs de baixo nível do TensorFlow (INFO e WARNING) para não poluir o console.
# TF_CPP_MIN_LOG_LEVEL: 0=todos os logs, 1=filtra INFO, 2=filtra INFO+WARNING, 3=filtra tudo exceto ERROS.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Desativa otimizações oneDNN (oneAPI Deep Neural Network Library) do TensorFlow.
# oneDNN pode causar pequenas diferenças nos resultados devido à ordem diferente de operações de ponto flutuante.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Carrega o dataset CIFAR-10: 60.000 imagens 32x32 RGB divididas em 10 classes (aviao, carro, pássaro, gato, veado, cachorro, sapo, cavalo, navio, caminhão).
# Retorna ((x_train, y_train), (x_test, y_test)) onde x são imagens e y são rótulos inteiros (0-9).
# Ciência da Computação: divisão treino/teste evita overfitting permitindo avaliar o modelo em dados nunca vistos.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normaliza os valores de pixels para o intervalo [0, 1].
# Imagens originais estão em uint8 [0, 255]. Divisão por 255.0 escalona para float32 [0.0, 1.0].
# Matemática: x_normalizado = x_original / 255. Isso estabiliza o gradiente durante o treino (evita explosão/desaparecimento de gradiente).
# Ciência da Computação: redes neurais convergem mais rápido quando as entradas estão em escala pequena e parecida.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Converte rótulos inteiros (ex: 3) para one-hot encoding (vetor com 1 na posição da classe, 0 nas outras).
# Exemplo: classe 3 em 10 classes vira [0,0,0,1,0,0,0,0,0,0].
# Matemática: representa a classe como vetor base canônico e_i em R^10. Necessário para perda categorical_crossentropy.
# Ciência da Computação: transforma problema de classificação multiclasse em formato adequado para funções de perda baseadas em entropia cruzada.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Cria gerador de data augmentation (aumento de dados sintéticos).
# rotation_range=15: rotaciona imagem aleatoriamente até 15 graus.
# width_shift_range=0.1: desloca horizontalmente até 10% da largura.
# height_shift_range=0.1: desloca verticalmente até 10% da altura.
# horizontal_flip=True: espelha horizontalmente com 50% de probabilidade.
# Matemática: transformações afins (rotação, translação, reflexão) preservam a semântica da classe enquanto aumentam a variedade do treino.
# Ciência da Computação: data augmentation atua como regularizador, reduz overfitting ao expor o modelo a versões variadas das mesmas imagens, aumentando o tamanho efetivo do dataset.
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Ajusta estatísticas internas do gerador (média e desvio padrão) com base nos dados de treino.
# Necessário se usar featurewise_center ou featurewise_std_normalization; aqui serve para inicializar o gerador.
datagen.fit(x_train)

# Constrói a arquitetura CNN (Convolutional Neural Network) usando API Sequencial.
# Sequencial empilha camadas linearmente; a saída de uma é a entrada da próxima.
def create_model():
    model = models.Sequential()
    
    # --- Bloco Convolucional 1 ---
    # Input(shape=(32, 32, 3)): define formato do tensor de entrada: altura=32, largura=32, canais=3 (RGB).
    # Tensor de entrada tem shape (batch_size, 32, 32, 3).
    model.add(layers.Input(shape=(32, 32, 3)))
    
    # Conv2D(32, (3, 3), activation='relu'): camada convolucional com 32 filtros de tamanho 3x3 e ativação ReLU.
    # Matemática da convolução: para cada filtro F (3x3x3) e posição (i,j), saída(i,j) = sum(F * patch) + b, seguido de ReLU(z)=max(0,z).
    # ReLU introduz não-linearidade, permitindo que a rede aprenda funções complexas (teorema de aproximação universal).
    # 32 filtros aprendem 32 detectores de características diferentes (bordas, texturas, cores).
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    
    # BatchNormalization(): normaliza ativações de cada lote (mini-batch).
    # Matemática: para cada característica, x_hat = (x - mean_batch) / sqrt(var_batch + epsilon), depois escala y = gamma * x_hat + beta.
    # Estabiliza o treino, reduz internal covariate shift, permite taxas de aprendizado maiores e atua como regularizador leve.
    model.add(layers.BatchNormalization())
    
    # Segunda camada convolucional do bloco 1: detecta características mais complexas compostas pelas do primeiro bloco.
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    
    # MaxPooling2D(2, 2): redução de dimensionalidade por subamostragem. Retém o valor máximo em janelas 2x2 com passo 2.
    # Matemática: reduz mapa de ativação de (H, W) para (H/2, W/2), diminuindo parâmetros e computação.
    # Torna a representação invariante a pequenas translações (deslocamentos) da imagem de entrada.
    model.add(layers.MaxPooling2D(2, 2))        
    
    # Dropout(0.25): desativa aleatoriamente 25% dos neurônios a cada passo de treino.
    # Matemática/Ciência da Computação: regularização que evita co-adaptação de neurônios (overfitting). No teste, todos os neurônios são usados com pesos escalados.
    
    model.add(layers.Dropout(0.25))
    
    # --- Bloco Convolucional 2 ---
    # Dobramos o número de filtros (64) pois mapas de características ficam menores (maior campo receptivo relativo).
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))        
    model.add(layers.Dropout(0.25))
    
    # --- Camadas Totalmente Conectadas (Dense) ---
    # Flatten(): transforma tensor 3D (altura, largura, canais) em vetor 1D.
    # Ex: (4, 4, 64) vira vetor de 1024 elementos. Necessário para conectar a camadas Dense.
    model.add(layers.Flatten())
    
    # Dense(512, activation='relu'): camada densa (fully connected) com 512 neurônios e ReLU.
    # Matemática: y = ReLU(W * x + b), onde W é matriz (512, entrada). Cada neurônio calcula combinação linear de todas as entradas.
    # Aprende combinações não lineares das características extraídas pelas camadas convolucionais.
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    
    # Dropout(0.5): taxa maior nas camadas densas porque possuem muitos parâmetros e tendem a overfit.
    model.add(layers.Dropout(0.5))
    
    # Dense(10, activation='softmax'): camada de saída com 10 neurônios (um por classe) e ativação softmax.
    # Matemática softmax: softmax(z_i) = exp(z_i) / sum(exp(z_j)) para j=1..10. Resulta em vetor de probabilidades que soma 1.
    # Interpretação: saída é uma distribuição de probabilidade sobre as 10 classes.
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Instancia o modelo com a arquitetura definida acima.
model = create_model()

# Compila o modelo configurando otimizador, função de perda e métricas.
# optimizer='adam': Adaptive Moment Estimation. Combina descida do estocástico com momento e escalonamento adaptativo de taxa de aprendizado por parâmetro.
#   Matemática: mantém médias móveis do gradiente (m) e do quadrado do gradiente (v), corrige viés e atualiza theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon).
# loss='categorical_crossentropy': entropia cruzada categórica.
#   Matemática: L = - sum(y_true * log(y_pred)). Mede diferença entre distribuição verdadeira (one-hot) e predita (softmax). Quanto menor, melhor.
# metrics=['accuracy']: fração de amostras classificadas corretamente.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treina o modelo usando gerador de dados aumentados.
# datagen.flow(x_train, y_train, batch_size=64): gera lotes de 64 imagens aumentadas em tempo real.
# epochs=20: número de passos completos sobre o dataset de treino.
# validation_data=(x_test, y_test): avalia perda e acurácia no teste a cada época (sem aplicar aumento).
# steps_per_epoch: número de lotes (batches) por época = 50000 // 64.
# Ciência da Computação: treino por mini-batches (stochastic gradient descent) equilibra precisão do gradiente e eficiência computacional/memória.
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_test, y_test),
    steps_per_epoch=x_train.shape[0] // 64
)

# Avalia o modelo no conjunto de teste (dados nunca vistos durante o treino).
# verbose=2 exibe barra de progresso resumida.
# Retorna perda e acurácia no teste.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.2f}")

# --- Visualização com Matplotlib ---
# Plota curvas de acurácia de treino e validação ao longo das épocas.
# Permite diagnosticar overfitting: se acurácia de treino sobe e validação estagna/cai, o modelo está memorizando os dados de treino.
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')          # eixo x: número de épocas
plt.ylabel('Accuracy')        # eixo y: acurácia (0 a 1)
plt.title("Training and Validation Accuracy")
plt.legend()                  # exibe legenda identificando cada curva
plt.show()                    # renderiza a figura na tela

# Plota curvas de perda (loss) de treino e validação.
# Perda deve diminuir com o treino. Se val_loss aumenta enquanto train_loss diminui, confirma overfitting.
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Training and Validation Losss")
plt.legend()
plt.show()