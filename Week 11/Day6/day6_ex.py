# =============================================================================
# IMPORTAÇÃO DE BIBLIOTECAS
# =============================================================================
import torch                                      # PyTorch: framework de deep learning (tensores, autograd, suporte a GPU)
import torch.nn as nn                             # Módulos de redes neurais (camadas, funções de perda, containers)
import torch.optim as optim                       # Otimizadores (Adam, SGD, etc.) para atualizar os pesos
from torch.utils.data import DataLoader, Dataset  # Dataset: base de dados customizada | DataLoader: carrega dados em lotes (batches)
import numpy as np                                # NumPy: arrays e operações matemáticas vetoriais eficientes

# -----------------------------------------------------------------------------
# DADOS DE EXEMPLO: frases em inglês e suas traduções em francês
# Corpus paralelo (pares fonte → alvo), muito pequeno, só para fins didáticos
# -----------------------------------------------------------------------------
english_sentences = ["hello", "how are you", "good morning", "thank you", "good night"]
french_sentences  = ["bonjour", "comment ça va", "bon matin", "merci", "bonne nuit"]

# -----------------------------------------------------------------------------
# VOCABULÁRIO E TOKENIZAÇÃO
# -----------------------------------------------------------------------------
def build_vocab(sentences):
    """
    Constrói um dicionário que mapeia cada palavra para um índice inteiro.
    Tokens especiais inclusos:
      <PAD> : preenchimento (padding) para igualar o tamanho de todas as frases
      <SOS> : início de sentença  (Start Of Sequence)
      <EOS> : fim de sentença     (End Of Sequence)
      <UNK> : palavra desconhecida (out-of-vocabulary / palavra não vista no treino)
    """
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for sentence in sentences:
        for word in sentence.split():          # separa a frase em palavras (split por whitespace)
            if word not in vocab:
                vocab[word] = len(vocab)       # atribui o próximo índice inteiro disponível
    return vocab

english_vocab = build_vocab(english_sentences)   # vocabulário do inglês  (idioma de origem / source)
french_vocab  = build_vocab(french_sentences)    # vocabulário do francês (idioma alvo / target)

# -----------------------------------------------------------------------------
# TOKENIZAÇÃO E PADDING (ENCHIMENTO)
# -----------------------------------------------------------------------------
def tokenize(sentences, vocab, max_len):
    """
    Transforma uma lista de frases em uma matriz de índices inteiros.
    Cada frase recebe:
      1. token <SOS> no início
      2. os índices das palavras
      3. token <EOS> no final
      4. tokens <PAD> até atingir max_len (padding à direita)
    Retorna um array NumPy de shape (numero_de_frases, max_len).
    """
    tokenized = []
    for sentence in sentences:
        # Converte cada palavra no seu índice; palavra desconhecida vira <UNK>
        tokens = [vocab.get(word, vocab['<UNK>']) for word in sentence.split()]
        # Adiciona marcadores de início e fim de sentença
        tokens = [vocab["<SOS>"]] + tokens + [vocab["<EOS>"]]
        # Preenche o restante com <PAD> até completar max_len
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        tokenized.append(tokens)
    return np.array(tokenized)

# Comprimento máximo = frase mais longa (em palavras) + 2 (por causa de <SOS> e <EOS>)
max_len_eng = max(len(sentence.split()) for sentence in english_sentences) + 2
max_len_fr  = max(len(sentence.split()) for sentence in french_sentences) + 2

# Converte as frases em matrizes numéricas (índices do vocabulário)
english_data = tokenize(english_sentences, english_vocab, max_len_eng)  # dados de entrada  (source)
french_data  = tokenize(french_sentences,  french_vocab,  max_len_fr)   # dados de saída alvo (target)

class TranslationDataset(Dataset):
    """
    Dataset customizada que herda de torch.utils.data.Dataset.
    Armazena os pares (frase em inglês, frase em francês) já tokenizados,
    permitindo que o DataLoader os entregue lotes a lotes.
    """
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data   # frases de entrada  (inglês / source)
        self.tgt_data = tgt_data   # frases de saída    (francês / target)

    def __len__(self):
        # Retorna o número total de exemplos disponíveis no dataset
        return len(self.src_data)

    def __getitem__(self, idx):
        # Retorna o exemplo na posição idx como dois tensores do PyTorch
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])

# Instancia o dataset com os pares inglês ↔ francês
dataset = TranslationDataset(english_data, french_data)

# DataLoader carrega os dados em lotes (batches) e embaralha a cada época
# batch_size=2  → processa 2 frases por vez (atualiza o gradiente a cada 2 exemplos)
# shuffle=True  → embaralha os exemplos a cada época (evita que o modelo aprenda a ordem dos dados)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class Encoder(nn.Module):
    """
    Codificador da arquitetura Seq2Seq.
    Recebe a sequencia de entrada (fonte) e a comprime em vetores de estado
    (hidden e cell) que servem de "memoria/contexto" para o decodificador.
    """
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        # Camada de embeddings: converte indices de palavras (inteiros) em vetores densos
        # de tamanho embed_dim. Matriz de pesos E de shape (vocabulario_fonte, embed_dim).
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # LSTM (Long Short-Term Memory) que processa a sequencia embutida.
        # batch_first=True significa que os tensores entram no formato (batch, sequencia, features).
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: tensor de indices de palavras, shape (batch, comprimento_da_frase)
        embedded = self.embedding(x)
        # embedded: shape (batch, comprimento, embed_dim)

        # A LSTM processa a sequencia inteira e retorna:
        #   outputs : saidas em cada passo de tempo (nao usadas aqui pelo encoder simples)
        #   hidden  : estado oculto final   de cada camada, shape (num_layers, batch, hidden_dim)
        #   cell    : estado de memoria final de cada camada, shape (num_layers, batch, hidden_dim)
        # Matematicamente, em cada passo t a LSTM atualiza celula e estado oculto por portoes:
        #   f_t = sigma(W_f . [h_{t-1}, x_t] + b_f)      (portao de esquecimento)
        #   i_t = sigma(W_i . [h_{t-1}, x_t] + b_i)      (portao de entrada)
        #   C~_t = tanh(W_C . [h_{t-1}, x_t] + b_C)      (candidato a nova memoria)
        #   C_t  = f_t . C_{t-1} + i_t . C~_t            (nova celula de memoria)
        #   o_t = sigma(W_o . [h_{t-1}, x_t] + b_o)      (portao de saida)
        #   h_t = o_t . tanh(C_t)                        (novo estado oculto)
        # onde "." e produto matriz-vetor, sigma e sigmoide, e . (entre vetores) e produto de Hadamard.
        outputs, (hidden, cell) = self.lstm(embedded)

        # Retorna apenas hidden e cell, que servem de contexto inicial do decodificador
        return hidden, cell

class Decoder(nn.Module):
    """
    Decodificador da arquitetura Seq2Seq.
    Gera a sequencia de saida (alvo) uma palavra de cada vez, condicionada:
      - ao estado oculto/celula finais do encoder (contexto da frase fonte)
      - a palavra gerada no passo anterior (modo autoregressivo)
    """
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        # Embedding do vocabulário alvo (frances)
        self.embbedding = nn.Embedding(output_dim, embed_dim)  # nome com erro de digitação no original: embbedding
        # LSTM que recebe o embedding da palavra anterior + estados (hidden, cell)
        # vindos do encoder ou do passo anterior do decoder
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        # Camada linear (fully connected) que mapeia o estado oculto para
        # uma distribuicao de probabilidades sobre o vocabulário alvo
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        # x: indice da palavra atual (ou <SOS> no primeiro passo), shape (batch,)
        # hidden, cell: estados vindos do encoder ou do passo anterior do decoder

        x = x.unsqueeze(1)                       # adiciona dimensao de sequencia: shape (batch, 1)
        embedded = self.embbedding(x)            # shape (batch, 1, embed_dim)

        # A LSTM processa UM passo de tempo, usando hidden e cell como estado inicial
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # outputs: shape (batch, 1, hidden_dim)

        predictions = self.fc(outputs.squeeze(1))  # remove dim do tempo -> (batch, hidden_dim) -> (batch, output_dim)
        # predictions: logits (pontuacoes nao normalizadas) para cada palavra do vocabulario alvo

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    """
    Modelo completo Encoder-Decoder sequencia-para-sequencia.
    Combina o encoder (comprime a frase fonte) com o decoder (gera a frase alvo).
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device  # "cuda" ou "cpu"

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: tensor de indices da frase fonte, shape (batch, comprimento_fonte)
        tgt: tensor de indices da frase alvo, shape (batch, comprimento_alvo)
        teacher_forcing_ratio: probabilidade de usar a palavra verdadeira (do alvo)
                               como entrada do proximo passo, em vez da palavra prevista pelo modelo.
        """
        batch_size = src.size(0)                 # numero de frases no lote
        tgt_len = tgt.size(1)                    # comprimento da frase alvo (inclui <SOS> e <EOS>)
        tgt_vocab_size = self.decoder.fc.out_features  # tamanho do vocabulário alvo

        # Tensor para guardar as previsoes (logits) do decoder em cada passo de tempo
        # shape (batch, comprimento_alvo, tamanho_vocab_alvo)
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # 1) ENCODER: processa a frase fonte e produz o estado inicial (contexto)
        hidden, cell = self.encoder(src)

        # 2) DECODER: primeiro input e o token <SOS> da frase alvo
        input = tgt[:, 0]                        # shape (batch,) — todos comecam com <SOS>

        # 3) Gera a frase alvo palavra por palavra (modo autoregressivo)
        #    Comeca no indice 1 porque o indice 0 e o <SOS> (ja usado como input inicial)
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output            # armazena os logits do passo t

            # top1: indice da palavra com maior probabilidade (predicao greedy deste passo)
            top1 = output.argmax(1)

            # TEACHER FORCING:
            #   com probabilidade teacher_forcing_ratio, o proximo input eh a palavra REAL do alvo (tgt[:, t])
            #   caso contrario, o proximo input eh a propria previsao do modelo (top1)
            # Isso ajuda o treino a convergir mais rapido, mas pode gerar "exposure bias"
            # (no teste o modelo so ve suas proprias predicoes).
            input = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs  # tensor de logits para todo o comprimento alvo

# -----------------------------------------------------------------------------
# DISPOSITIVO DE COMPUTACAO (GPU se disponivel, senao CPU)
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# HIPERPARAMETROS DO MODELO
# -----------------------------------------------------------------------------
input_dim  = len(english_vocab)   # tamanho do vocabulário fonte (ingles)
output_dim = len(french_vocab)    # tamanho do vocabulário alvo (frances)
embed_dim  = 64                   # dimensao do vetor de embedding de cada palavra
hidden_dim = 128                  # dimensao do estado oculto da LSTM
num_layers = 2                    # numero de camadas LSTM empilhadas

# Instancia encoder, decoder e o modelo Seq2Seq completo, movendo tudo para o dispositivo
encoder = Encoder(input_dim, embed_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers)
model = Seq2Seq(encoder, decoder, device).to(device)

# Otimizador Adam: atualiza os pesos do modelo usando gradientes adaptativos
# lr=0.001 e a taxa de aprendizado inicial
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Funcao de perda: CrossEntropyLoss mede a diferenca entre a distribuicao prevista
# (logits) e a palavra verdadeira. ignore_index descarta os tokens <PAD> do calculo
# de perda, para que o padding nao seja tratado como uma palavra real a ser prevista.
criterion = nn.CrossEntropyLoss(ignore_index=french_vocab["<PAD>"])

def train(model, dataloader, optimizer, criterion, device, num_epochs=20):
    """
    Loop de treinamento por epocas.
    Em cada epoca, percorre todo o dataloader, faz forward, calcula perda,
    retropropaga o gradiente (backward) e atualiza os pesos (optimizer.step).
    """
    model.train()  # coloca o modelo em modo de treinamento (ativa dropout, se houvesse)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for src, tgt in dataloader:
            # Move os lotes (src: ingles, tgt: frances) para o dispositivo (GPU/CPU)
            src, tgt = src.to(device), tgt.to(device)

            # Zera os gradientes acumulados de lotes anteriores
            optimizer.zero_grad()

            # Forward: modelo produz logits para toda a sequencia alvo
            output = model(src, tgt)

            # Descarta o primeiro passo (<SOS) tanto nas previsoes quanto no alvo,
            # pois nao prevemos o <SOS>. Achata os tensores para o formato esperado
            # pela CrossEntropyLoss: (batch * sequencia, vocab_size) e (batch * sequencia,)
            output = output[:, 1:].reshape(-1, output.shape[2])
            tgt    = tgt[:, 1:].reshape(-1)

            # Calcula a entropia cruzada entre previsoes e alvos reais
            loss = criterion(output, tgt)

            # Backpropagation through time: calcula gradientes da perda em relacao aos pesos
            loss.backward()

            # Atualiza os pesos usando o otimizador Adam
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epochs {epoch+1},/{num_epochs}, Loss:{epoch_loss / len(dataloader):.4f}")

# Executa o treinamento por 20 epocas
train(model, dataloader, optimizer, criterion, device)

def translate_sentence(model, sentence, english_vocab, french_vocab, max_len_fr, device):
    """
    Traduz uma frase fonte (ingles) para a frase alvo (frances) usando greedy decoding.
    Passos:
      1. Tokeniza a frase de entrada e adiciona <SOS> e <EOS>.
      2. Passa pelo encoder para obter o contexto (hidden, cell).
      3. Gera o alvo palavra por palavra: sempre escolhe a palavra de maior probabilidade
         (argmax) ate encontrar <EOS> ou atingir o comprimento maximo.
    """
    model.eval()  # modo avaliacao (desativa dropout/batch norm em treino, se existissem)

    # Tokeniza a frase de entrada (fonte) e adiciona marcadores de inicio e fim
    tokens = [english_vocab.get(word, english_vocab["<UNK>"]) for word in sentence.split()]
    tokens = [english_vocab["<SOS>"]] + tokens + [english_vocab["<EOS>"]]
    src = torch.tensor(tokens).unsqueeze(0).to(device)  # shape (1, comprimento)

    # Encodifica a frase fonte em contexto (hidden, cell) sem calcular gradientes
    with torch.no_grad():
        hidden, cell = model.encoder(src)

    # Inverte o dicionario frances: indice -> palavra, para ler a saida
    tgt_vocab = {v: k for k, v in french_vocab.items()}

    # Comeca a geracao com o token <SOS>
    tgt_indices = [french_vocab["<SOS>"]]

    # Decodificacao greedy (gananciosa): um passo de cada vez
    for _ in range(max_len_fr):
        # Input do passo atual e a ultima palavra gerada (autoregressivo)
        tgt_tensor = torch.tensor([tgt_indices[-1]]).to(device)

        # Forward em um unico passo do decoder
        output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)

        # Escolhe a palavra com maior score (argmax = greedy decoding)
        pred = output.argmax(1).item()
        tgt_indices.append(pred)

        # Para quando o decoder gerar <EOS>
        if pred == french_vocab["<EOS>"]:
            break

    # Remove os marcadores <SOS> (indice 0) e <EOS> (ultimo) da lista de indices
    translated_sentence = [tgt_vocab[i] for i in tgt_indices[1:-1]]
    return " ".join(translated_sentence)


# -----------------------------------------------------------------------------
# TESTE DE TRADUCAO
# -----------------------------------------------------------------------------
# Nota didatico: a frase de teste esta em frances ("bonne nuit"), embora os
# vocabularios tenham sido construidos com ingles como fonte. Isso e apenas um
# exemplo de execucao do pipeline; em um uso real a frase de entrada deveria estar
# no idioma fonte (ingles).
sentence = "bonne nuit"
translation = translate_sentence(model, sentence, english_vocab, french_vocab, max_len_fr, device)
print(f"Translated Sentence: {translation}")