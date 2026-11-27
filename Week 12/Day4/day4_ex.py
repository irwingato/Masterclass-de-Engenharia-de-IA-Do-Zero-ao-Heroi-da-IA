import numpy as np
import matplotlib.pyplot as plt

# Exemplo: Positional Encoding + Transformer

# Função de codificação posicional (positional encoding)
def positional_encoding(seq_len, embed_dim):
    # pos: matriz (seq_len, 1) com as posições dos tokens: [0, 1, 2, ..., seq_len-1]
    pos = np.arange(seq_len)[:, np.newaxis]
    # i: matriz (1, embed_dim) com os índices das dimensões: [0, 1, 2, ..., embed_dim-1]
    i = np.arange(embed_dim)[np.newaxis, :]
    # angle_rates: taxa de variação angular por par de dimensões: 10000^(-2*(i//2)/embed_dim)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
    # angle_rads: radianos = posição × taxa de variação -> matriz (seq_len, embed_dim)
    angle_rads = pos * angle_rates
    
    # Aplicar seno nos índices pares e cosseno nos ímpares
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_encoding

# Gera o positional encoding
seq_len = 50
embed_dim = 16
pos_encoding = positional_encoding(seq_len, embed_dim)

# Visualiza o positional encoding como um heatmap
plt.figure(figsize=(10,6))
plt.pcolormesh(pos_encoding, cmap='viridis')
plt.colorbar()
plt.title("Positional Encoding")
plt.xlabel("Embedding Dimension")
plt.show()

import torch
import torch.nn as nn

class TransformerWithPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads, ff_dim):
        super(TransformerWithPositionalEncoding, self).__init__()
        # Embedding de tokens: tabela de busca com shape (seq_len, embed_dim)
        self.embedding = nn.Embedding(seq_len, embed_dim)
        # Parâmetro de positional encoding fixo (não aprendível)
        self.positional_encoding = nn.Parameter(torch.tensor(positional_encoding(seq_len, embed_dim), dtype=torch.float32))
        # Atenção multi-cabeça
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        # Rede feed-forward (feed-forward)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(), 
            nn.Linear(ff_dim, embed_dim)
        )
        # Normalização em camadas (LayerNorm) 1
        self.norm1 = nn.LayerNorm(embed_dim)
        # Normalização em camadas (LayerNorm) 2
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Adicionar o positional encoding ao embedding
        x = self.embedding(x) + self.positional_encoding
        # Atenção auto (self-attention)
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.norm1(x + attn_output)
        # Rede feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

# Define os parâmetros do modelo
embed_dim = 16
seq_len = 50
num_heads = 4
ff_dim = 64

model = TransformerWithPositionalEncoding(embed_dim, seq_len, num_heads, ff_dim)
print(model)

# Positional encoding aprendível
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super(LearnablePositionalEncoding, self).__init__()
        # Parâmetro do positional encoding aprendível
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, embed_dim))
    
    def forward(self, x):
        return x + self.positional_encoding

learnable_pe = LearnablePositionalEncoding(seq_len, embed_dim)
print(learnable_pe)