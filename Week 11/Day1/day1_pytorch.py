import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# Hiperparâmetros do problema
# -----------------------------------------------------------------------------
# vocab_size: tamanho do vocabulário considerado. Apenas as 10.000 palavras mais
# frequentes do dataset IMDB são mantidas. Palavras mais raras são mapeadas para
# o índice 0 (padding/unknown).
# Ciência da computação: controla a dimensão da matriz de Embedding
# (vocab_size x embedding_dim) e, portanto, a quantidade de parâmetros treináveis.
vocab_size = 10000

# max_len: comprimento fixo de cada sequência (review). Reviews mais curtos são
# preenchidos com zeros (padding); reviews mais longos são truncados.
# Necessário porque redes neurais exigem entradas de dimensão fixa (tensor
# retangular de shape [batch_size, max_len]) para operações de matriz.
max_len = 200

# -----------------------------------------------------------------------------
# Carregamento e pré-processamento dos dados
# -----------------------------------------------------------------------------
# imdb.load_data carrega o dataset IMDB de avaliações de filmes, já pré-processado
# como sequências de inteiros (índices de palavras no vocabulário).
# num_words=vocab_size restringe ao vocabulário das N palavras mais frequentes.
# Retorna: ((X_train, y_train), (X_test, y_test))
#   X_train / X_test: listas de listas de inteiros (sequências de índices)
#   y_train / y_test: rótulos binários (0 = negativa, 1 = positiva)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# pad_sequences ajusta todas as sequências para o mesmo comprimento (max_len).
# padding="post": adiciona zeros no FINAL da sequência.
#   Exemplo: [1, 5, 3] com max_len=5 vira [1, 5, 3, 0, 0]
# Ciência da computação: transforma listas de comprimento variável em um tensor
# 2D de shape (n_amostras, max_len), compatível com operações de matriz (dot product).
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# -----------------------------------------------------------------------------
# Preparação dos DataLoaders (PyTorch)
# -----------------------------------------------------------------------------
# TensorDataset empacota tensores de entrada e rótulos em um único Dataset.
# torch.tensor converte as listas/arrays NumPy em tensores do PyTorch.
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))

# DataLoader envolve o Dataset e fornece mini-batches.
#   batch_size=32: número de amostras por atualização de peso.
#   shuffle=True: embaralha as amostras a cada época, evitando que o modelo
#   aprenda a ordem dos dados e ajudando a convergência do gradiente estocástico.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -----------------------------------------------------------------------------
# Definição da arquitetura do modelo (RNN)
# -----------------------------------------------------------------------------
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        # nn.Module é a classe base de todos os módulos de rede neural no PyTorch.
        # super().__init__() inicializa a classe pai para registrar parâmetros.
        super(RNNModel, self).__init__()

        # -----------------------------------------------------------------
        # Camada Embedding
        # -----------------------------------------------------------------
        # nn.Embedding(vocab_size, embedding_dim) cria uma matriz de pesos
        # W_emb de shape (vocab_size, embedding_dim).
        #
        # MATEMÁTICA:
        #   O Embedding é uma tabela de busca (lookup table). Cada palavra i
        #   é representada por um vetor denso w_i = W_emb[i, :] ∈ R^embedding_dim.
        #   Durante o treino, W_emb é aprendido via backpropagation para que
        #   palavras com significado similar fiquem próximas no espaço vetorial
        #   (similaridade do cosseno / distância euclidiana).
        #
        # CIÊNCIA DA COMPUTAÇÃO:
        #   Substitui a representação one-hot (esparsa, tamanho vocabulário) por
        #   um vetor denso de baixa dimensão (128), reduzindo drasticamente a
        #   dimensionalidade e a esparsidade. Complexidade de consulta: O(1).
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # -----------------------------------------------------------------
        # Camada RNN (recorrente simples)
        # -----------------------------------------------------------------
        # nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        #   embedding_dim: dimensão da entrada em cada passo de tempo (128).
        #   hidden_dim: número de unidades ocultas (128).
        #   batch_first=True: o tensor de entrada tem shape (batch, seq, feature).
        #
        # MATEMÁTICA (célula RNN em cada passo de tempo t):
        #   h_t = tanh( W_ih · x_t + b_ih + W_hh · h_{t-1} + b_hh )
        #   onde:
        #     x_t ∈ R^128        : vetor de embedding da palavra no passo t
        #     h_{t-1} ∈ R^128    : estado oculto anterior (inicializado em zeros)
        #     h_t   ∈ R^128      : novo estado oculto
        #     W_ih ∈ R^(128×128) : pesos de entrada-oculto
        #     W_hh ∈ R^(128×128) : pesos oculto-oculto (recorrentes)
        #     b_ih, b_hh ∈ R^128 : bias
        #     tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z)) ∈ (-1, 1)
        #
        # Após processar todos os max_len passos de tempo, o estado final h_T
        # carrega uma representação resumida (encoding) de toda a sequência.
        #
        # CIÊNCIA DA COMPUTAÇÃO:
        #   RNNs processam dados sequenciais mantendo um estado oculto atualizado
        #   a cada passo, capturando dependências temporais/contextuais entre
        #   palavras. Complexidade: O(T · H^2) por sequência, onde T = max_len
        #   e H = hidden_dim. O gradiente desvanecente (vanishing gradient) pode
        #   limitar a captura de dependências de longo prazo.
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        # -----------------------------------------------------------------
        # Camada Linear (saída, fully connected)
        # -----------------------------------------------------------------
        # nn.Linear(hidden_dim, output_dim) cria pesos W_fc de shape
        # (output_dim, hidden_dim) e bias b_fc de shape (output_dim,).
        #
        # MATEMÁTICA:
        #   z = W_fc · h_T + b_fc          (logit, escalar quando output_dim=1)
        #   onde W_fc ∈ R^(1×128), b_fc ∈ R.
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # forward define a passagem para frente (forward pass) da rede.
        #
        # x: tensor de entrada de shape (batch_size, max_len) com índices de
        # palavras.

        # Passagem pelo Embedding: cada índice vira um vetor denso.
        # Resultado: shape (batch_size, max_len, embedding_dim)
        embedded = self.embedding(x)

        # Passagem pelo RNN:
        #   output: shape (batch_size, max_len, hidden_dim) — estados de todo
        #           passo de tempo.
        #   hidden: shape (num_layers, batch_size, hidden_dim) — estado final.
        output, hidden = self.rnn(embedded)

        # hidden.squeeze(0) remove a dimensão de num_layers (=1), resultando em
        # shape (batch_size, hidden_dim). Usamos apenas o último estado oculto,
        # que resume a sequência inteira.
        #
        # Passagem pela camada linear:
        #   z = W_fc · h_T + b_fc   -> shape (batch_size, 1)
        # Aplicamos sigmoid para obter probabilidade ∈ (0, 1):
        #   y_hat = sigmoid(z) = 1 / (1 + e^(-z))
        # Interpretável como P(classe=1 | entrada) = probabilidade de avaliação
        # positiva.
        return torch.sigmoid(self.fc(hidden.squeeze(0)))

# -----------------------------------------------------------------------------
# Instanciação do modelo
# -----------------------------------------------------------------------------
# embedding_dim=128: dimensão dos vetores de palavra.
# hidden_dim=128: tamanho do estado oculto da RNN.
# output_dim=1: um único neurônio de saída (classificação binária).
model = RNNModel(vocab_size=10000, embedding_dim=128, hidden_dim=128, output_dim=1)

# -----------------------------------------------------------------------------
# Função de perda e otimizador
# -----------------------------------------------------------------------------
# nn.BCELoss(): Binary Cross-Entropy Loss. Para um exemplo com rótulo y ∈ {0,1}
# e predição p = y_hat:
#   L = -[ y · log(p) + (1 - y) · log(1 - p) ]
# MATEMÁTICA: penaliza predições confiantes e erradas.
#   Quando y=1, L = -log(p); quando y=0, L = -log(1-p).
#   A derivada ∂L/∂z = p - y é simples e estável (quando combinada com sigmoid).
criterion = nn.BCELoss()

# optim.Adam: Adaptive Moment Estimation. Combina momento (SGD com momentum)
# e RMSProp, ajustando a taxa de aprendizado por parâmetro com base em médias
# móveis dos gradientes (primeiro momento m) e gradientes ao quadrado
# (segundo momento v).
# lr=0.001: taxa de aprendizado padrão.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------------------------------------------------------
# Loop de treinamento
# -----------------------------------------------------------------------------
def train_rnn(model, train_loader, criterion, optimizer, epochs=5):
    # model.train(): coloca o modelo em modo de treinamento. Ativa camadas como
    # dropout e batch normalization (não usadas aqui, mas é boa prática).
    model.train()

    # Loop sobre épocas (passagens completas pelo conjunto de treino).
    for epoch in range(epochs):
        epoch_loss = 0

        # Loop sobre mini-batches do DataLoader.
        # shuffle=True garante batches diferentes a cada época.
        for X_batch, y_batch in train_loader:
            # X_batch: shape (batch_size, max_len)
            # y_batch: shape (batch_size,) com rótulos 0/1

            # optimizer.zero_grad(): limpa os gradientes acumulados de iterações
            # anteriores. No PyTorch, os gradientes se acumulam por padrão, então
            # deve-se zerar antes de cada backward().
            optimizer.zero_grad()

            # Forward pass: calcula predições para o batch.
            # predictions: shape (batch_size, 1) antes do squeeze.
            predictions = model(X_batch).squeeze(1)

            # Calcula a perda (binary cross-entropy) entre predições e rótulos.
            # y_batch.float() converte inteiros para float (necessário para BCELoss).
            loss = criterion(predictions, y_batch.float())

            # Backward pass: backpropagation via regra da cadeia. Computa
            # ∇_θ L = ∂L/∂θ para todos os parâmetros θ do modelo.
            loss.backward()

            # Atualização dos parâmetros: Adam realiza θ ← atualização baseada
            # em m_hat e v_hat (médias móveis de gradientes e gradientes²).
            optimizer.step()

            # loss.item(): extrai o valor escalar do tensor de perda.
            epoch_loss += loss.item()

        # Média da perda sobre todos os batches da época.
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

# Executa o treinamento por 5 épocas.
train_rnn(model, train_loader, criterion, optimizer)

# -----------------------------------------------------------------------------
# Avaliação no conjunto de teste
# -----------------------------------------------------------------------------
def evaluate_rnn(model, X_test, y_test):
    # model.eval(): coloca o modelo em modo de avaliação. Desativa dropout e
    # batch normalization (se existissem), garantindo comportamento determinístico.
    model.eval()

    # torch.no_grad(): desativa o cálculo de gradientes. Economia de memória e
    # velocidade, pois não é necessário salvar o grafo computacional para backward.
    with torch.no_grad():
        # Converte dados de teste em tensores e faz a predição.
        # X_test: shape (n_test, max_len)
        predications = model(torch.tensor(X_test)).squeeze(1)

        # Calcula a perda no conjunto de teste (sem atualizar pesos).
        loss = criterion(predications, torch.tensor(y_test).float())

        # Acurácia:
        #   (predications > 0.5): aplica limiar 0.5 para decidir classe prevista
        #     (1 se probabilidade > 0.5, senão 0).
        #   == torch.tensor(y_test).float(): compara com o rótulo verdadeiro,
        #     resultando em tensor de booleanos (True/False).
        #   .float().mean(): converte booleanos para 1.0/0.0 e tira a média.
        #   .item(): extrai o escalar.
        # Acurácia = (número de acertos) / (total de amostras).
        accuracy = ((predications > 0.5) == torch.tensor(y_test).float()).float().mean().item()

    print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

# Executa a avaliação nos dados de teste (vistos nunca durante o treino).
# Fornece estimativa não tendenciosa do desempenho em dados novos (generalização).
evaluate_rnn(model, X_test, y_test)
