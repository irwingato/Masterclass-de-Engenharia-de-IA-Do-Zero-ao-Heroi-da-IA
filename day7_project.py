#!/usr/bin/env python3
"""Day 7 Project - Engenharia de IA Do Zero ao Herói da IA"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print(f"TensorFlow version: {tf.__version__}")

# Load IMDB dataset
print("\nCarregando dataset IMDB...")
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=100)
print(f"Dados de treinamento: {x_train.shape}, {y_train.shape}")
print(f"Dados de teste: {x_test.shape}, {y_test.shape}")

# Pad sequences to maxlen=100
print("\nPadronizando sequences...")
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)
print(f"Após padronização - Treinamento: {x_train.shape}")
print(f"Após padronização - Teste: {x_test.shape}")

# Build the model
print("\nConstruindo modelo Sequencial...")

model = keras.Sequential([
    # Embedding layer: converts word indices to dense vectors
    keras.layers.Embedding(input_dim=100, output_dim=32, input_length=100),
    # GlobalAveragePooling1D: computes mean across sequence dimension
    keras.layers.GlobalAveragePooling1D(),
    # Dense hidden layer with ReLU activation
    keras.layers.Dense(16, activation='relu'),
    # Dropout for regularization (50% dropout)
    keras.layers.Dropout(0.5),
    # Output layer with sigmoid for binary classification
    keras.layers.Dense(1, activation='sigmoid')
])

print("Modelo definido com sucesso!")
print(model.summary())

# Compile the model
print("\nCompilando modelo...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("Modelo compilado com sucesso!")

# Train the model
print("\nTreinando modelo...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=512,
    validation_data=(x_test, y_test),
    verbose=1
)
print("Treinamento concluído!")

# Scientific explanations section
print("\n" + "="*60)
print("EXPLICAÇÕES CIENTÍFICAS E MATEMÁTICAS")
print("="*60)

explanations = """
1. EMBEDDING LAYER:
   - Mapeia índices de palavras para vetores densos de dimensão 32
   - Cada palavra i → vetor e_i ∈ R^32
   - Palavras semanticamente semelhantes têm vetores próximos (propriedade de "distributional hypothesis")
   - Permite que a rede neural processe texto transformando-o em números

2. GLOBAL AVERAGE POOLING:
   - Computa a média: z = (1/n) · Σ(x_i) ao longo da dimensão da sequência
   - Reduz sequência T×D → vetor D (apenas as características médias)
   - Equivalente a um pooling que captura a "essência" geral do texto

3. DENSE (FULLY CONNECTED) LAYERS:
   - Operação matricial: y = σ(Wx + b) onde W ∈ R^(m×n)
   - Cada neurônio conecta-se a todas as entradas anteriores
   - Camadas ocultas aprendem características compostas em hierarquias

4. RELU ATIVATION (f(x) = max(0, x)):
   - Resolve gradiente vanishing em redes profundas
   - Non-linearity essencial para aprendizado de funções complexas
   - Computacionalmente eficiente (comparação em vez de exponenciação)

5. DROPOUT (taxa 0.5):
   - Durante treinamento: desativa cada neurônio com probabilidade 0.5
   - Força aprendizado de representações redundantes
   - Reduz overfitting ao impedir co-adaptação de neurônios
   - Na inferência: outputs multiplicados por taxa de retenção

6. SIGMOID ATIVATION (saída):
   - σ(x) = 1/(1 + e^(-x)), mapeia R → (0,1)
   - Interpretado como probabilidade de classe positiva
   - Função de perda binary_crossentropy assume entradas sigmoideadas

7. BINARY CROSS-ENTROPY:
   - L(y, p) = -[y·log(p) + (1-y)·log(1-p)]
   - Mede "surpresa" da previsão em relação ao rótulo verdadeiro
   - Gradiente: ∂L/∂p = -(y/p) + ((1-y)/(1-p))
   - Função convexa garantindo otimização estável

8. OTIMIZADOR ADAM:
   - Combinação de Momentum e RMSprop
   - m_t = β_1·m_{t-1} + (1-β_1)·g_t  (primeiro momento)
   - v_t = β_2·v_{t-1} + (1-β_2)·g_t²  (segundo momento)
   - θ_t = θ_{t-1} - α·m_t/(√v_t + ε)  (atualização de parâmetros)
   - β_1=0.9, β_2=0.999, ε=10^-8 (parâmetros padrão)
   - Adapta taxa de aprendizado individualmente por parâmetro

9. TF_CPP_MIN_LOG_LEVEL=3:
   - Nível 0: todas as mensagens (debug + info + warning)
   - Nível 1: oculta debug (warning + info apenas)
   - Nível 2: oculta info (warning apenas)
   - Nível 3: oculta tudo (apenas erros críticos)
   - Evita poluição do stderr no Windows CMD

10. TF_ENABLE_ONEDNN_OPTS=0:
    - oneDNN = biblioteca de operações otimizadas da TensorFlow
    - Desativar pode mudar ordem de computação → ligeiras diferenças numéricas
    - Útil para reproducibilidade ou debugging de operações customizadas

11. Windows GPU limitation:
    - TensorFlow >= 2.11: suporte nativo a GPU não disponível no Windows
    - Mensagem é informativa, não erro crítico
    - Alternativas: WSL2, plugin Direct-ML, ou TensorFlow 2.10 anterior
    - CPU otimizado com instruções SSE3, SSE4.1, AVX, AVX2, FMA
"""

print(explanations.strip())

# Evaluate the model
print("\nAvaliando modelo no teste...")
results = model.evaluate(x_test, y_test, verbose=0)
print(f"Loss: {results[0]:.4f}")
print(f"Acurácia: {results[1]:.4f}")

# Make predictions
print("\nExemplos de previsão:")
predictions = model.predict(x_test[:5], verbose=0)
for i in range(5):
    prob = float(predictions[i][0])
    predicted_class = 1 if prob > 0.5 else 0
    print(f"  Amostra {i}: probabilidade={prob:.4f}, classe prevista={predicted_class}, real={int(y_test[i])}")

# Evaluate the model
print("\nAvaliando modelo no teste...")
results = model.evaluate(x_test, y_test, verbose=0)
print(f"Loss: {results[0]:.4f}")
print(f"Acurácia: {results[1]:.4f}")

# Make predictions
print("\nExemplos de previsão:")
predictions = model.predict(x_test[:5], verbose=0)
for i in range(5):
    prob = float(predictions[i][0])
    predicted_class = 1 if prob > 0.5 else 0
    print(f"  Amostra {i}: probabilidade={prob:.4f}, classe prevista={predicted_class}, real={int(y_test[i])}")