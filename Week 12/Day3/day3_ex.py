import numpy as np  # Biblioteca NumPy: operações numéricas e de matrizes (vetores e tensores)

# Gera matrizes aleatórias de Query, Key e Value (Q, K, V) para a atenção
def generate_data(seq_len, embed_dim):
    np.random.seed(42)  # Fixa a semente (seed) para garantir reprodutibilidade
    # Retorna uma matriz aleatória de shape (seq_len, embed_dim)
    # Cada linha é uma posição da sequência e cada coluna a dimensão do embedding
    return np.random.rand(seq_len, embed_dim)

# Hiperparâmetros: comprimento da sequência e dimensão do embedding
sequence_length = 4  # Número de tokens (posições) na sequência de entrada
embedding_dim = 3   # Dimensão do vetor de embedding (d)

# Matrizes de Query, Key e Value, cada uma com shape (seq_len, embed_dim)
query = generate_data(sequence_length, embedding_dim)  # Q: consulta (o que cada token "pergunta")
key = generate_data(sequence_length, embedding_dim)   # K: chave (opção de comparação)
value = generate_data(sequence_length, embedding_dim) # V: valor (atributo a ser combinado)

# Calcula os scores de atenção: score[i, j] = produto escalar entre o query i e o key j
# Escala por 1/sqrt(embedding_dim) para evitar saturação na softmax
scores = np.dot(query, key.T) / np.sqrt(embedding_dim)
# Matematicamente: score[i, j] = (Q_i · K_j) / sqrt(d)

# Aplica softmax para normalizar os scores em pesos de atenção (cada linha soma 1)
def softmax(x):
    # Subtrai o máximo para garantir estabilidade numérica, depois exponencia e normaliza
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # Normaliza dividindo pelo total para que cada linha (eixo -1) soma 1
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)  # shape (seq_len, seq_len)

# Calcula o vetor de contexto: soma ponderada dos valores
# context[i] = soma_j attention_weights[i, j] * value[j]
context = np.dot(attention_weights, value)

# print("Attention Weights: \n", attention_weights)
# print("Context Vector:\n", context)

import torch  # PyTorch: framework de deep learning com operações sobre tensores
import torch.nn as nn  # Módulo nn: contém as camadas (layers) de redes neurais, como a Linear
import torch.nn.functional as F  # Módulo F: funções funcionais (como softmax) sem estado persistente

# Módulo Multi-Head Attention: executa múltiplos heads (subespaços de atenção) em paralelo

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 'embed_dim' é a dimensão de entrada do modelo; 'num_heads' divide esse espaço em heads paralelos
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Cada head trabalha com uma dimensão de embedding dividida pelo número de heads
        self.num_head_dim = embed_dim // num_heads
        
        # Garante que embed_dim seja divisível por num_heads para que cada head tenha dimensão válida
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        
        # Camadas lineares para projeções de Q, K, V e para a saída final (equação: x @ W.T + b)
        self.query = nn.Linear(embed_dim, embed_dim)
        # Camada linear para a projeção de key (x @ W.T + b)
        self.key = nn.Linear(embed_dim, embed_dim)
        # Camada linear para a projeção de value (x @ W.T + b)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size = x.size(0)  # Número de sequências no batch

        # Projeções lineares: compute Q, K, V a partir do input x
        # Reescreve (batch, seq_len, embed_dim) para (batch, seq_len, num_heads, head_dim) e transposta para (batch, num_heads, seq_len, head_dim)
        q = self.query(x).view(batch_size, -1, self.num_heads, self.num_head_dim).transpose(1, 2)
        # Mesma reestruturação aplicada às keys
        k = self.key(x).view(batch_size, -1, self.num_heads, self.num_head_dim).transpose(1, 2)
        # Mesma reestruturação aplicada aos values
        v = self.value(x).view(batch_size, -1, self.num_heads, self.num_head_dim).transpose(1, 2)
        
        # Calcula os scores de atenção: scores = Q · K^T / sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.num_head_dim)
        attention_weights= F.softmax(scores, dim=-1)
        
        # Calcula o contexto: combina os values com os pesos de atenção
        context = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out(context), attention_weights

# Entrada de exemplo (sample input): tensor de shape (batch_size=1, seq_len=4, embed_dim=8)
seq_len, embed_dim = 4, 8
x = torch.rand(1, seq_len, embed_dim)  # Dados simulados para teste

# Instancia e testa o módulo
mha = MultiHeadAttention(embed_dim, num_heads=2)  # Dois heads, cada um com num_head_dim = embed_dim / 2
context, attn_weights = mha(x)
print("Attention Weights:\n", attn_weights)
print("Context Vector:\n", context)

import matplotlib.pyplot as plt  # Biblioteca para visualização de gráficos

# Visualiza os pesos de atenção de um único head
plt.matshow(attn_weights[0, 0].detach().numpy(), cmap='viridis')
plt.colorbar()
plt.title("Attention Weights (Head 1)")
plt.show()