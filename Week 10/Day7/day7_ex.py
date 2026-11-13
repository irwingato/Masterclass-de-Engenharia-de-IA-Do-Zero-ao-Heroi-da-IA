import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# =============================================================================
# 1. TRANSFORMAÇÕES E DATA AUGMENTATION
# =============================================================================

# transforms.Compose encadeia várias transformações em sequência.
# Cada imagem do CIFAR-10 passa por este pipeline antes de entrar na rede.

transform_train = transforms.Compose([
    # RandomHorizontalFlip: espelha a imagem horizontalmente com probabilidade 0.5.
    # Data augmentation (aumento de dados): cria variações do mesmo exemplo para
    # o modelo não decorar a posição exata dos objetos (invariança à tradução/reflexão).
    transforms.RandomHorizontalFlip(),

    # RandomCrop(32, padding=4): adiciona 4 pixels de borda (padding) e faz um corte
    # aleatório de 32x32. Força o modelo a reconhecer objetos mesmo quando não estão
    # centralizados. Reduz overfitting (memorização) e melhora generalização.
    transforms.RandomCrop(32, padding=4),

    # ToTensor: converte imagem PIL ou NumPy (valores 0-255, canal último HxWxC)
    # para torch.Tensor (valores 0-1 float, canal primeiro CxHxW).
    # Necessário porque as camadas do PyTorch esperam tensores no formato NCHW.
    transforms.ToTensor(),

    # Normalize: padroniza cada canal pelo fórmula x' = (x - mean) / std.
    # Com mean=0.5 e std=0.5 em cada canal (R, G, B), os valores saem aproximadamente
    # no intervalo [-1, 1]. Padronização ajuda o gradiente a descer mais rápido e
    # evita que um canal com valores maiores domine os outros.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Transformação para teste: sem data augmentation (queremos avaliar na imagem original,
# apenas converte para tensor e padroniza com os mesmos parâmetros do treino).
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =============================================================================
# 2. CARREGAMENTO DO DATASET CIFAR-10
# =============================================================================

# CIFAR-10: 60.000 imagens 32x32 coloridas em 10 classes (avião, automóvel, pássaro,
# gato, cervo, cão, sapo, cavalo, navio, caminhão). 50.000 treino, 10.000 teste.
# download=True baixa automaticamente se não estiver na pasta ./data.

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

print(f"Training Data Size: {len(train_dataset)}")  # 50000
print(f"Test Data Size: {len(test_dataset)}")        # 10000

# =============================================================================
# 3. DATALOADER (CARREGAMENTO EM LOTES / BATCHES)
# =============================================================================

# DataLoader encapsula o dataset e entrega dados em lotes (batches) durante o treino.
# batch_size=64: 64 imagens por iteração. Afeta estabilidade do gradiente e uso de memória.
# shuffle=True (treino): embaralha as amostras a cada época. Essencial para Descida
# Estocástica do Gradiente (SGD) não ficar presa em padrões de ordem do dataset.
# shuffle=False (teste): mantém ordem fixa para avaliação reprodutível.

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =============================================================================
# 4. ARQUITETURA DA REDE NEURAL CONVOLUCIONAL (EnhancedCNN)
# =============================================================================

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        # ---------------------------------------------------------------------
        # Camadas convolucionais
        # ---------------------------------------------------------------------

        # nn.Conv2d(c_in, c_out, kernel_size): aplica convolução discreta 2D.
        # Matematicamente: saida[i,j] = sum_{m,n} entrada[i+m, j+n] * kernel[m,n] + bias.
        # Extrai características locais (bordas, texturas, formas).
        # conv1: 3 canais de entrada (RGB) -> 6 mapas de características (filtros), kernel 5x5.
        # Parâmetros treináveis: 6 * 3 * 5 * 5 + 6 (bias) = 456.
        self.conv1 = nn.Conv2d(3, 6, 5)

        # nn.BatchNorm2d(n_features): normalização em lote por canal.
        # Para cada canal c: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta.
        # E[x], Var[x] são média e variância do batch. Reduz "covariate shift",
        # acelera convergência e permite learning rate maior.
        self.bn1 = nn.BatchNorm2d(6)

        # conv2: 6 canais de entrada -> 16 mapas de saída, kernel 5x5.
        # Parâmetros: 16 * 6 * 5 * 5 + 16 = 2416.
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.bn2 = nn.BatchNorm2d(16)

        # nn.MaxPool2d(2, 2): pooling máximo em janela 2x2 com passo 2.
        # Reduz a dimensão espacial pela metade (downsampling), mantém o valor mais
        # forte (ativações mais relevantes), traz invariância translacional local e
        # reduz custo computacional.
        self.pool = nn.MaxPool2d(2, 2)

        # nn.Dropout(p): desativa aleatoriamente p=50% dos neurônios durante o TREINO.
        # Regularização que impede co-adaptação de neurônios (overfitting), forçando
        # a rede a aprender representações mais robustas. No teste (eval) é desligado.
        self.dropout = nn.Dropout(0.5)

        # ---------------------------------------------------------------------
        # Cálculo automático do tamanho da entrada da camada totalmente conectada
        # ---------------------------------------------------------------------

        # Passa um tensor fictício (dummy input) com o mesmo formato das imagens
        # (batch=1, canais=3, altura=32, largura=32) pelas camadas convolucionais
        # para descobrir quantos neurônios saem depois do flatten. Assim não
        # precisamos calcular à mão o tamanho da entrada do fc1.
        self._calculate_conv_output()

        # ---------------------------------------------------------------------
        # Camadas totalmente conectadas (Linear / Dense)
        # ---------------------------------------------------------------------

        # nn.Linear(entrada, saída): transforma y = x @ W.T + b.
        # Cada neurônio se conecta a todos da camada anterior (camada densa).
        # fc1: achatado convolucional -> 120 neurônios.
        self.fc1 = nn.Linear(self.conv_output_size, 120)

        # fc2: 120 -> 84 neurônios.
        self.fc2 = nn.Linear(120, 84)

        # fc3: 84 -> 10 neurônios (um por classe do CIFAR-10). Saída são "logits"
        # (pontuações brutas por classe, sem ativação).
        self.fc3 = nn.Linear(84, 10)

    def _calculate_conv_output(self):
        # Dummy input com mesmo tamanho das imagens do CIFAR-10: 1 imagem, 3 canais, 32x32.
        dummy_input = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():  # não precisa calcular gradiente neste passo
            # Passa pelo mesmo caminho do forward: conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> pool.
            # Cálculo dimensional esperado:
            #   Entrada: 32x32
            #   Conv1 (k=5, s=1, p=0): 32 - 5 + 1 = 28 -> 28x28
            #   Pool (k=2, s=2): 28 / 2 = 14 -> 14x14
            #   Conv2 (k=5, s=1, p=0): 14 - 5 + 1 = 10 -> 10x10
            #   Pool (k=2, s=2): 10 / 2 = 5 -> 5x5
            #   Canais: 16
            #   Total achatado (flatten): 16 * 5 * 5 = 400
            output = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(dummy_input)))))))
        # numel() retorna o número total de elementos do tensor (400 neste caso).
        self.conv_output_size = output.numel()

    def forward(self, x):
        # =====================================================================
        # PASSO FRENTE (forward pass): propagação da entrada até a saída.
        # =====================================================================

        # Bloco 1: Convolução + BatchNorm + ReLU.
        # ReLU(x) = max(0, x): ativação não-linear que descarta valores negativos.
        # Sem não-linearidade, múltiplas camadas lineares colapsariam em uma só.
        x = F.relu(self.bn1(self.conv1(x)))

        # Bloco 2: Convolução + BatchNorm + ReLU + MaxPool (downsampling).
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Achata o tensor preservando o tamanho do batch (x.size(0)).
        # Transforma (batch, 16, 5, 5) em (batch, 400) para entrar nas camadas densas.
        x = x.view(x.size(0), -1)

        # Dropout aplica máscara aleatória apenas no modo treino.
        x = self.dropout(x)

        # Camadas densas com ReLU, exceto a última (saída bruta/logits).
        # CrossEntropyLoss já aplica log-softmax internamente, então não colocamos
        # softmax aqui (evitaria problemas numéricos e duplicação).
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x  # shape: (batch_size, 10)

# Instancia o modelo e imprime a arquitetura (resumo das camadas e parâmetros).
model = EnhancedCNN()
print(model)

# =============================================================================
# 5. FUNÇÃO DE PERDA (LOSS) E OTIMIZADOR
# =============================================================================

# nn.CrossEntropyLoss: combina LogSoftmax + Negative Log Likelihood.
# Matematicamente: L = - sum_i y_i * log(softmax(x)_i), onde y é one-hot da classe verdadeira.
# Espera logits (saída bruta da rede) e rótulos inteiros (0-9).
# Quanto maior a diferença entre a predição e o rótulo, maior a perda.
criterion = nn.CrossEntropyLoss()

# optim.Adam: otimizador adaptativo que combina momento (primeiro momento do gradiente)
# e escala adaptativa (segundo momento), semelhante a RMSprop.
# Atualização simplificada: m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*g^2;
# theta = theta - lr * m_hat / (sqrt(v_hat) + eps), com correção de viés.
# lr=0.001 é o tamanho do passo (learning rate).
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lista para armazenar a perda média de cada época (usada no gráfico).
training_loss = []

# =============================================================================
# 6. LOOP DE TREINAMENTO
# =============================================================================

def train_model(model, train_loader, criterion, optimizer, epochs=20):
    # model.train(): coloca o modelo em modo treinamento.
    # Ativa Dropout (aplica máscara aleatória) e faz BatchNorm usar estatísticas
    # do batch atual (além de acumular médias móveis para teste).
    model.train()

    for epoch in range(epochs):  # uma época = passar por todo o dataset de treino uma vez
        running_loss = 0.0

        # Itera sobre os batches do DataLoader: imagens (batch, 3, 32, 32) e labels (batch,).
        for images, labels in train_loader:
            # Limpa gradientes acumulados em iterações anteriores.
            # O PyTorch acumula gradientes por padrão (.backward() soma), então
            # devemos zerar antes de cada batch.
            optimizer.zero_grad()

            # Passo frente: gera predições (logits) para o batch.
            outputs = model(images)

            # Calcula a perda entre predições e rótulos verdadeiros.
            loss = criterion(outputs, labels)

            # Retropropagação (backpropagation): aplica a regra da cadeia do cálculo
            # para computar o gradiente da perda em relação a cada parâmetro (dL/dw).
            loss.backward()

            # Atualiza os pesos usando o otimizador Adam com os gradientes calculados.
            optimizer.step()

            # Acumula a perda do batch para calcular a média da época.
            # += adiciona o valor atual ao acumulador (bug corrigido: antes era += escrito errado).
            running_loss += loss.item()

        # Perda média da época: soma das perdas dos batches dividida pelo número de batches.
        epoch_loss = running_loss / len(train_loader)
        training_loss.append(epoch_loss)  # armazena para plotar a curva depois

        # Exibe o progresso a cada época.
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# Executa o treino por 20 épocas. Após isso, training_loss terá 20 valores.
train_model(model, train_loader, criterion, optimizer)

# =============================================================================
# 7. AVALIAÇÃO NO CONJUNTO DE TESTE
# =============================================================================

def evaluate_model(model, test_loader):
    # model.eval(): coloca o modelo em modo avaliação.
    # Desliga Dropout e faz BatchNorm usar estatísticas acumuladas (média/variância
    # do treino), garantindo predições determinísticas.
    model.eval()

    correct = 0
    total = 0

    # torch.no_grad(): desativa o grafo de computação de gradientes.
    # Economia de memória e velocidade, pois não precisamos de backprop na avaliação.
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            # torch.max(outputs, 1): retorna o índice da classe com maior logit
            # (predição da rede) ao longo da dimensão das classes (dim=1).
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)  # acumula total de amostras avaliadas

            # Compara predição com rótulo verdadeiro; soma quantas estão corretas.
            correct += (predicted == labels).sum().item()

    # Acurácia = (acertos / total) * 100. Métrica de classificação padrão.
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Executa avaliação no conjunto de teste (10.000 imagens).
evaluate_model(model, test_loader)

# =============================================================================
# 8. VISUALIZAÇÃO DA CURVA DE PERDA (LOSS CURVE)
# =============================================================================

# Plota a perda média de cada época.
# Se o modelo estiver aprendendo, a curva deve diminuir (convergência).
# Curva vazia ou plana indica que training_loss não foi preenchido (bug de
# indentação/escopo corrigido acima) ou que o modelo não está aprendendo.
plt.plot(training_loss, label="Training Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()