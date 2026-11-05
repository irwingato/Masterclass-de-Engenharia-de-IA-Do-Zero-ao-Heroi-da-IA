import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. IMPORTS E PRÉ-PROCESSAMENTO
# ==============================================================================

# torch: biblioteca principal de deep learning (tensores, autograd, otimizadores).
# torchvision.datasets: datasets prontos (MNIST, CIFAR, etc.).
# torchvision.transforms: transformações aplicadas às imagens (redimensionar, normalizar, etc.).
# torch.utils.data.DataLoader: carrega dados em batches, embaralha e paraleliza leitura.
# torch.nn: módulos de rede neural (camadas lineares, convolucionais, funções de ativação, loss).
# torch.nn.functional: versão funcional (sem estado) de ativações e losses.

# ==============================================================================
# 2. TRANSFORMAÇÕES DE IMAGEM (DATA PREPROCESSING)
# ==============================================================================

# Compose encadeia várias transformações, aplicando-as sequencialmente a cada imagem.
transform = transforms.Compose([
    # ToTensor(): converte imagem PIL (0-255, uint8, HxWxC) para tensor float (0-1, CxHxW).
    # Para MNIST (escala de cinza), passa de [0,255] para [0.0,1.0] com shape [1,28,28].
    transforms.ToTensor(),
    # Normalize(mean, std): aplica (x - mean) / std canal a canal.
    # Aqui mean=0.5, std=0.5, então x_normalizado = (x - 0.5) / 0.5 = 2x - 1.
    # Resultado: valores aproximadamente no intervalo [-1, 1], centralizados em zero.
    # Centralizar os dados ajuda a otimização (gradientes mais estáveis, convergência mais rápida).
    transforms.Normalize((0.5,), (0.5,))
])

# ==============================================================================
# 3. CARREGAMENTO DOS DATASETS (MNIST)
# ==============================================================================

# MNIST: 70.000 imagens 28x28 de dígitos manuscritos (0-9), divididos em:
#   - 60.000 para treino (train=True)
#   - 10.000 para teste (train=False)
# root='./data': pasta onde os dados são baixados/armazenados localmente.
# download=True: baixa automaticamente se não encontrar os arquivos.
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# ==============================================================================
# 4. DATALOADER (BATCHING E EMBARALHAMENTO)
# ==============================================================================

# DataLoader encapsula o dataset e entrega dados em batches.
# batch_size=32: processa 32 amostras por iteração (trade-off: memória vs. estabilidade do gradiente).
# shuffle=True (treino): embaralha a cada época para evitar viés de ordem e ajudar a generalização.
# shuffle=False seria ideal para teste, mas True aqui não prejudica a acurácia final (apenas ordem).
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

print(f"Training Data Size: {len(train_dataset)}")  # 60000
print(f"Test Data Size: {len(test_dataset)}")        # 10000

# ==============================================================================
# 5. DEFINIÇÃO DA REDE NEURAL (MLP - Multi-Layer Perceptron)
# ==============================================================================

class NeuralNetwork(nn.Module):
    # nn.Module é a classe base de todos os módulos de rede neural no PyTorch.
    # Ela gerencia parâmetros (pesos e biases), dispositivos (CPU/GPU) e modo treino/avaliação.

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Chama o construtor da classe pai (nn.Module) para registrar os submodulos e parâmetros.

        # Flatten: transforma o tensor de entrada [batch, 1, 28, 28] em [batch, 784].
        # Achata as dimensões espacial (28x28) e de canal (1) em um único vetor de 784 features.
        # Não tem parâmetros treináveis; é apenas uma operação de reorganização de memória (view/reshape).
        self.flatten = nn.Flatten()

        # Camada Linear (fully connected / densa): y = x * W^T + b
        # W tem shape [out_features, in_features], b tem shape [out_features].
        # fc1: 784 entradas -> 128 neurônios escondidos. Parâmetros: 784*128 + 128 = 100.480.
        self.fc1 = nn.Linear(28*28, 128)
        # fc2: 128 -> 64 neurônios. Parâmetros: 128*64 + 64 = 8.256.
        self.fc2 = nn.Linear(128, 64)
        # fc3: 64 -> 10 neurônios (um por classe/dígito 0-9). Parâmetros: 64*10 + 10 = 650.
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # forward define o passe frente (forward pass): como os dados fluem pela rede.
        # É chamado automaticamente quando fazemos model(entradas).

        x = self.flatten(x)                         # [batch, 1, 28, 28] -> [batch, 784]
        x = F.relu(self.fc1(x))                     # ReLU introduce não-linearidade (veja abaixo)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                             # Camada de saída: logits brutos (sem ativação)
        # CrossEntropyLoss espera logits (não probabilidades), então não aplicamos softmax aqui.
        return x

# ==============================================================================
# 6. FUNÇÕES DE ATIVAÇÃO E NÃO-LINERIDADE (BASE MATEMÁTICA)
# ==============================================================================

# ReLU(x) = max(0, x)
# Por que não-linearidade? Sem ela, composição de camadas lineares equivale a uma única
# camada linear (W2 * W1 * x), limitando o modelo a funções lineares. ReLU permite
# aproximar qualquer função contínua (Teorema de Aproximação Universal).
# Vantagem do ReLU: computação barata, não sofre saturação para x > 0 (gradiente constante 1),
# o que mitiga o problema do desaparecimento de gradientes em redes profundas.
# Desvantagem: "dying ReLU" (neurônios com x sempre < 0 ficam com gradiente 0 e nunca atualizam).

# Instancia o modelo. Os pesos são inicializados aleatoriamente (default: Kaiming Uniform para Linear).
model = NeuralNetwork()
print(model)  # Imprime a arquitetura e número de parâmetros

# ==============================================================================
# 7. FUNÇÃO DE PERDA (LOSS) E OTIMIZADOR
# ==============================================================================

# CrossEntropyLoss combina LogSoftmax + NLLLoss (Negative Log Likelihood) em uma única classe.
# Matematicamente, para uma amostra com verdadeira classe y e logits z:
#   L = -log( exp(z_y) / sum_j exp(z_j) ) = -z_y + log( sum_j exp(z_j) )
# Isso mede a distância entre a distribuição prevista (softmax dos logits) e a distribuição one-hot do alvo.
# Quanto maior a probabilidade atribuída à classe correta, menor a perda.
criterion = nn.CrossEntropyLoss()

# Adam (Adaptive Moment Estimation): otimizador de descida do gradiente estocástico com
# taxas de aprendizado adaptativas por parâmetro. Combina momento (média móvel dos gradientes)
# e RMSProp (média móvel dos quadrados dos gradientes).
# lr=0.001: tamanho do passo na atualização dos pesos. Muito grande pode divergir; muito pequeno fica lento.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==============================================================================
# 8. LOOP DE TREINAMENTO
# ==============================================================================

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    # model.train(): ativa modo treino. Habilita dropout e batch norm (se existirem).
    model.train()

    # Uma época (epoch) = uma passagem completa por todo o dataset de treino.
    for epoch in range(epochs):
        running_loss = 0.0  # Acumula a perda média da época para exibição.

        for images, labels in train_loader:
            # images: [batch, 1, 28, 28], labels: [batch] (inteiros 0-9).

            # optimizer.zero_grad(): zera os gradientes acumulados (.grad) de todos os parâmetros.
            # PyTorch acumula gradientes por padrão (.grad += novo_grad), então devemos zerar a cada batch
            # para não somar gradientes de batches diferentes.
            optimizer.zero_grad()

            # --- FORWARD PASS ---
            # Calcula saídas da rede para o batch atual.
            # Autograd registra todas as operações para calcular gradientes depois.
            outputs = model(images)                 # [batch, 10] logits
            loss = criterion(outputs, labels)       # perda escalar (média sobre o batch)

            # --- BACKWARD PASS (RETROPROPAGAÇÃO) ---
            # loss.backward(): calcula dL/dparametro para todo parâmetro com requires_grad=True,
            # usando a regra da cadeia (chain rule) do cálculo, propagando do output para as entradas.
            loss.backward()

            # --- OTIMIZAÇÃO ---
            # optimizer.step(): atualiza cada parâmetro: param = param - lr * grad (simplificado do Adam).
            optimizer.step()

            running_loss += loss.item()  # .item() extrai o valor Python do tensor (desconecta do grafo).

        # Média da perda sobre todos os batches da época.
        print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader):.4f}")

# Treina por 10 épocas (sobrescreve o default de 5).
train_model(model, train_loader, criterion, optimizer, epochs=10)

# ==============================================================================
# 9. LOOP DE AVALIAÇÃO (TESTE)
# ==============================================================================

def evaluate_model(model, test_loader):
    # model.eval(): ativa modo avaliação. Desativa dropout e fixa batch norm em estatísticas aprendidas.
    model.eval()

    correct = 0  # Contador de predições corretas
    total = 0    # Total de amostras avaliadas

    # torch.no_grad(): desativa o rastreamento de autograd.
    # Durante avaliação não precisamos de gradientes, então isso economiza memória e acelera a execução.
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)                      # [batch, 10] logits
            # torch.max(input, dim=1): retorna (valores_máximos, índices_máximos) ao longo da coluna.
            # O índice do logit máximo é a classe prevista pelo modelo.
            _, predicted = torch.max(outputs.data, 1)    # predicted: [batch] com classes 0-9

            total += labels.size(0)                      # soma o tamanho do batch
            # (predicted == labels) gera um tensor booleano; .sum() conta os True; .item() converte para int.
            correct += (predicted == labels).sum().item()

    # Acurácia = percentual de predições corretas sobre o total de amostras.
    print(f"Test Accuracy: {100 * correct/total:.2f}%")

# Avalia o modelo treinado no dataset de teste (dados não vistos durante o treino).
evaluate_model(model, test_loader)

# ==============================================================================
# 10. SALVAMENTO E CARREGAMENTO DO MODELO
# ==============================================================================

# state_dict: dicionário Python que mapeia cada camada para seus tensores de pesos e biases.
# torch.save serializa esse dicionário para arquivo (pickle). Não salva a arquitetura, apenas os parâmetros.
torch.save(model.state_dict(), 'mnist_model.pth')

# Para recarregar: instanciar a mesma arquitetura e carregar os pesos salvos.
loaded_model = NeuralNetwork()
# torch.load lê o arquivo; load_state_dict copia os pesos para as camadas correspondentes.
# Importante: a arquitetura deve ser idêntica à usada no salvamento.
loaded_model.load_state_dict(torch.load('mnist_model.pth'))

# Verifica se o modelo carregado tem o mesmo desempenho (sanity check).
evaluate_model(loaded_model, test_loader)

# ==============================================================================
# 11. (OPCIONAL) RETREINO COM NOVA TAXA DE APRENDIZAGEM
# ==============================================================================

# Comentado: exemplo de como ajustar o learning rate e continuar treinando (fine-tuning).
# Reduzir o lr nas épocas finais ajuda o modelo a convergir para um mínimo mais preciso
# (estratégia conhecida como learning rate decay / scheduling).
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# train_model(model, train_loader, criterion, optimizer)
# evaluate_model(model, test_loader)
