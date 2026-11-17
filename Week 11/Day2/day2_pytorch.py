import os
# Suprime logs do TensorFlow/oneDNN antes de qualquer import do tensorflow.
# TF_CPP_MIN_LOG_LEVEL=3 exibe apenas erros (0=tudo, 1=INFO, 2=WARNING, 3=ERRO).
# TF_ENABLE_ONEDNN_OPTS=0 desativa otimizações oneDNN que geram mensagens de aviso.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch                      # Biblioteca principal de deep learning (tensores, autograd, otimizadores).
import torch.nn as nn             # Módulos de redes neurais (camadas, funções de perda, containers).
import torch.optim as optim       # Algoritmos de otimização (Adam, SGD, etc.).
from torch.utils.data import DataLoader, TensorDataset  # Carregador de dados em lotes (batches) e dataset a partir de tensores.
from tensorflow.keras.datasets import imdb               # Dataset IMDB de críticas de cinema (positivas/negativas).
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Ajusta tamanho de sequências com padding/truncamento.

# Hiperparâmetros do modelo e do vocabulário.
vocab_size = 10000   # Número de palavras mais frequentes do dataset a serem consideradas (índices 0..vocab_size-1).
max_len = 200        # Tamanho fixo de cada sequência (review). Sequências maiores são truncadas; menores, preenchidas com zeros.

# ------------------------------------------------------------------
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ------------------------------------------------------------------

# imdb.load_data devolve tuplas ((treino), (teste)).
# Cada amostra é uma lista de inteiros (índices de palavras no vocabulário).
# y são rótulos binários: 0 = negativa, 1 = positiva.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# pad_sequences iguala o comprimento de todas as sequências para max_len.
# padding="post" adiciona zeros no final das sequências curtas.
# Sequências longas são cortadas no final (truncamento pós).
# Matemática: mapeia sequências de comprimento variável para um tensor de shape (n_amostras, max_len),
# necessário porque redes neurais exigem entradas de dimensão fixa para formar batches (matrizes).
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test  = pad_sequences(X_test,  maxlen=max_len, padding="post")

# TensorDataset empacota features (X) e rótulos (y) como tensores do PyTorch.
# DataLoader divide o dataset em mini-batches de tamanho batch_size e embaralha (shuffle) a cada época.
# Ciência da computação: o uso de batches reduz o uso de memória e permite atualizações de pesos mais frequentes
# (estocásticas), ajudando a escapar de mínimos locais e a generalizar melhor.
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ------------------------------------------------------------------
# 2. DEFINIÇÃO DO MODELO (RNN)
# ------------------------------------------------------------------

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()  # Inicializa a classe base nn.Module (registra parâmetros/submódulos).

        # Camada de Embedding: converte cada índice de palavra em um vetor denso de tamanho embedding_dim.
        # Matemática: W_emb é uma matriz de shape (vocab_size, embedding_dim). Para um índice i, a saída é a linha i de W_emb.
        # Isso aprende representações semânticas (palavras com significados similares ficam próximas no espaço vetorial).
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Camada RNN (recorrente simples Elman).
        # Processa a sequência um passo de cada vez, mantendo um estado oculto h_t.
        # Equação (para cada passo t):
        #   h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
        # Onde x_t é o embedding da palavra no passo t, h_{t-1} é o estado oculto anterior.
        # batch_first=True significa que a entrada tem shape (batch, seq_len, features).
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # Camada linear (fully connected) que mapeia o estado oculto final para a saída (1 neurônio).
        # y = W_fc @ h_T + b_fc, onde h_T é o último estado oculto da sequência.
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: tensor de índices de palavras, shape (batch, seq_len).
        embedded = self.embedding(x)  # shape (batch, seq_len, embedding_dim).

        # output: saídas da RNN em todos os passos (batch, seq_len, hidden_dim).
        # hidden: estado oculto final (1, batch, hidden_dim) para RNN de uma camada unidirecional.
        output, hidden = self.rnn(embedded)

        # Pega o último estado oculto (hidden[0]) e remove a primeira dimensão (n_layers=1).
        # hidden.squeeze(0): shape (batch, hidden_dim).
        # self.fc(...): shape (batch, 1).
        # torch.sigmoid(...): aplica função sigmoid elemento a elemento, mapeando para (0, 1).
        # Matemática do sigmoid: σ(z) = 1 / (1 + exp(-z)). Interpreta-se como probabilidade da classe positiva.
        return torch.sigmoid(self.fc(hidden.squeeze(0)))

# Instancia o modelo com embedding_dim=128, hidden_dim=128, saída binária (1 neurônio).
model = RNNModel(vocab_size=10000, embedding_dim=128, hidden_dim=128, output_dim=1)

# ------------------------------------------------------------------
# 3. FUNÇÃO DE PERDA E OTIMIZADOR
# ------------------------------------------------------------------

# BCELoss = Binary Cross Entropy Loss.
# Matemática: L = -[ y * log(p) + (1 - y) * log(1 - p) ], onde p = sigmoid(saída do modelo).
# Penaliza fortemente previsões confiantes e erradas. Adequada para classificação binária.
criterion = nn.BCELoss()

# Adam = Adaptive Moment Estimation. Combina momento (média dos gradientes) e RMSProp (média dos gradientes ao quadrado).
# lr=0.001 é a taxa de aprendizado (tamanho do passo na atualização dos pesos).
# Ciência da computação: Adam ajusta a taxa de aprendizado individualmente para cada parâmetro, acelerando a convergência.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------------
# 4. LOOP DE TREINAMENTO
# ------------------------------------------------------------------

def train_rnn(model, train_loader, criterion, optimizer, epochs=5):
    model.train()  # Coloca o modelo em modo treino (ativa dropout/batch norm se houver; aqui não afeta o RNN simples).

    for epoch in range(epochs):  # Uma época = uma passagem completa pelo dataset de treino.
        epoch_loss = 0
        for X_batch, y_batch in train_loader:  # Itera sobre mini-batches de tamanho 32.
            optimizer.zero_grad()              # Zera os gradientes acumulados (evita acumulação entre batches).

            # Forward pass: predição do modelo para o lote atual.
            # squeeze(1) remove a dimensão extra de saída (shape de (batch,1) para (batch,)) para coincidir com o target.
            predictions = model(X_batch).squeeze(1)

            # Calcula a perda entre predições e rótulos verdadeiros (convertidos para float).
            loss = criterion(predictions, y_batch.float())

            # Backward pass (retropropagação): calcula gradientes de perda w.r.t. todos os parâmetros via autograd.
            # Matemática: usa regra da cadeia para computar dL/dW para cada peso W da rede.
            loss.backward()

            # Atualiza os pesos: W = W - lr * gradiente (versão adaptativa do Adam).
            optimizer.step()

            epoch_loss += loss.item()  # Acumula o valor escalar da perda (desconecta do grafo de computação).

        # Média da perda por batch nesta época.
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

# Executa o treinamento por 5 épocas.
train_rnn(model, train_loader, criterion, optimizer)

# ------------------------------------------------------------------
# 5. AVALIAÇÃO NO CONJUNTO DE TESTE
# ------------------------------------------------------------------

def evaluate_rnn(model, X_test, y_test):
    model.eval()  # Modo avaliação (desativa comportamentos específicos de treino).

    with torch.no_grad():  # Desativa o cálculo de gradientes (economiza memória e tempo; não vamos treinar aqui).
        # Predições para todo o teste de uma vez.
        predictions = model(torch.tensor(X_test)).squeeze(1)  # shape (n_testes,).

        # Perda no conjunto de teste.
        loss = criterion(predictions, torch.tensor(y_test).float())

        # Acurácia: fração de predições corretas.
        # Limiar 0.5: se p > 0.5 prevê classe 1 (positiva), senão classe 0 (negativa).
        # Compara com o rótulo verdadeiro e calcula a média (proporção de acertos).
        accuracy = ((predictions > 0) == torch.tensor(y_test).float()).float().mean().item()

    print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

# Executa avaliação final.
evaluate_rnn(model, X_test, y_test)
