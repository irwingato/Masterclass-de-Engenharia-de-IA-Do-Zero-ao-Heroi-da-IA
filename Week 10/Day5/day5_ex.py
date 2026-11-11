import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =============================================================================
# 1. TRANSFORMACOES DAS IMAGENS (PRE-PROCESSAMENTO)
# =============================================================================

# transforms.Compose encadeia varias transformacoes, aplicando-as sequencialmente
# a cada imagem carregada do dataset.
transform = transforms.Compose([
    # Converte uma imagem PIL (valores inteiros 0-255, formato HxWxC) para um
    # torch.FloatTensor (valores float 0.0-1.0, formato CxHxW).
    transforms.ToTensor(),
    # Normaliza cada canal pela formula: x' = (x - media) / desvio_padrao
    # Com media=0.5 e desvio=0.5 em R, G, B:
    #   x' = (x - 0.5) / 0.5 = 2*x - 1
    # Mapeia o intervalo [0, 1] para [-1, 1], centrando os dados em zero.
    # Isso ajuda a convergencia do gradiente durante o treinamento da rede neural.
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =============================================================================
# 2. CARREGAMENTO DO DATASET CIFAR-10
# =============================================================================

# CIFAR-10 contem 60.000 imagens coloridas 32x32x3 divididas em 10 classes.
# - root="./data": pasta onde o dataset sera armazenado localmente
# - train=True / False: separa treino (50.000) e teste (10.000)
# - download=True: baixa o dataset automaticamente caso nao exista localmente
# - transform: aplica as pre-processacoes definidas acima a cada imagem
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# =============================================================================
# 3. DATALOADERS (BATCHING E EMBARALHAMENTO)
# =============================================================================

# DataLoader envolve o Dataset e entrega os dados em lotes (mini-batches).
# - batch_size=64: cada iteracao retorna 64 amostras (mini-batch gradient descent)
# - shuffle=True: embaralha os dados a cada epoca (evita viés de ordem no treino)
# - shuffle=False: mantem ordem fixa (adequado para avaliacao/teste)
# NOTA: linha original usava train_dataset no test_loader; corrigido para test_dataset.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training Data Size: {len(train_dataset)}")
print(f"Test Data Size: {len(test_dataset)}")

# =============================================================================
# 4. DEFINICAO DO MODELO CNN (REDE NEURAL CONVOLUCIONAL)
# =============================================================================

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Primeira camada convolucional: 3 canais de entrada (RGB), 6 filtros,
        # tamanho do kernel 5x5.
        # Saida espacial: 32 - 5 + 1 = 28 -> tensor (6, 28, 28)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # MaxPooling 2x2 com passo 2: reduz a metade cada dimensao espacial
        # (28, 28) -> (14, 14)
        self.pool = nn.MaxPool2d(2, 2)
        # Segunda camada convolucional: 6 entradas, 16 filtros, kernel 5x5.
        # Saida espacial: 14 - 5 + 1 = 10 -> tensor (16, 10, 10)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Apos segundo pooling: (10, 10) -> (5, 5), tensor (16, 5, 5)
        # Camadas totalmente conectadas (dense/linear):
        # fc1: 16*5*5 = 400 entradas -> 120 neuronios
        # fc2: 120 -> 84
        # fc3: 84 -> 10 (numero de classes do CIFAR-10)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Convolucao + ReLU + MaxPooling
        x = self.pool(torch.relu(self.conv1(x)))
        # Segunda convolucao + ReLU + MaxPooling
        x = self.pool(torch.relu(self.conv2(x)))
        # Achata o tensor (batch, 16, 5, 5) para (batch, 16*5*5) antes das camadas densas
        x = x.view(-1, 16 * 5 * 5)
        # Camadas densas com ativacao ReLU
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Camada de saida (logits brutos, sem softmax - o CrossEntropyLoss ja aplica)
        x = self.fc3(x)
        return x

model = CNN()
print(model)

# =============================================================================
# 5. FUNCAO DE PERDA E OTIMIZADOR
# =============================================================================

# CrossEntropyLoss combina LogSoftmax + NLLLoss. Espera logits (nao normalizados)
# como entrada e indices de classe como alvo. Matematicamente:
#   L = -log(exp(z_y) / sum(exp(z_k)))  onde z_y é o logit da classe correta
criterion = nn.CrossEntropyLoss()

# SGD (Descida do Gradiente Estocastica) com momento.
# Atualizacao: v = momento * v - lr * grad;  theta = theta + v
# O momento ajuda a acelerar direcoes consistentes e atenuar oscilacoes.
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Alternativa: Adam (taxa de aprendizado adaptativa por parametro)
# optim.Adam(model.parameters(), lr=0.01)

# =============================================================================
# 6. LOOP DE TREINAMENTO
# =============================================================================

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()  # Modo treino: habilita dropout/batchnorm em modo treinamento
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Zera os gradientes acumulados (evita acumulacao entre lotes)
            optimizer.zero_grad()

            # Forward pass: predicao do modelo
            outputs = model(images)
            # Calcula a perda entre predicoes e labels reais
            loss = criterion(outputs, labels)

            # Backward pass: computacao de gradientes via backpropagation (autograd)
            loss.backward()
            # Atualiza os pesos: theta = theta - lr * grad (ajustado pelo momento)
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

train_model(model, train_loader, criterion, optimizer)

# =============================================================================
# 7. AVALIACAO / TESTE
# =============================================================================

def evaluate_model(model, test_loader):
    model.eval()  # Modo avaliacao: desativa dropout/batchnorm
    correct = 0
    total = 0
    # desabilita calculo de gradientes para economizar memoria e acelerar
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            # Pega o indice da classe com maior logit (predicao da rede)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

evaluate_model(model, test_loader)