"""
visualization_day2.py - Simplified Transformer Encoder block (Keras).

Explicações matemáticas:
- A multi-head attention divide a dimensão d_model em num_heads cabeças e cada
  cabeça computa a atenção por produto escalar: A = softmax(QK^T / sqrt(d_k)) V.
- A saída da atenção é combinada com a entrada via conexão residual,
  seguida de Layer Normalization.
- A feed-forward aplica duas camadas densas com ativação ReLU.
- A saída final é combinada com conexão residual e normalizada.

- A Layer Normalization aplica normalização por feature:
  LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta.

Notas de ciência da computação:
- Usa a API funcional do Keras para construir o encoder block.
- O `plot_model` gera um diagrama PNG da arquitetura.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONDNN_OPTS'] = '0'

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Add, MultiHeadAttention

# Definir um Transformer Encoder block simplificado
def transformer_encoder(input_dim, num_heads, ff_dim):
    inputs = Input(shape=(None, input_dim))
    # Multi-Head Self-Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization()(attention_output)
    
    # Feed-Forward Neural Network
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dense(input_dim)(ff_output)
    outputs = Add()([attention_output, ff_output])
    outputs = LayerNormalization()(outputs)
    return Model(inputs, outputs)

# Criar e visualizar um sample Transformer Encoder block
encoder_block = transformer_encoder(input_dim=64, num_heads=8, ff_dim=128)
plot_model(encoder_block, show_shapes=True, to_file="transformer_encoder.png")
