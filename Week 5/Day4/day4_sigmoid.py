import numpy as np
import matplotlib.pyplot as plt

# Função Sigmoide: sigma(z) = 1 / (1 + e^(-z))
# Matemática: mapeia qualquer real z em (0, 1), interpretável como probabilidade.
#   - lim z->+inf sigma = 1, lim z->-inf sigma = 0, sigma(0) = 0.5
#   - Derivada: sigma'(z) = sigma(z) * (1 - sigma(z)) -> máxima em z=0 (0.25), útil no gradiente.
#   - É a função de ligação da Regressão Logística: p(y=1|x) = sigma(w.x + b).
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gera 100 valores de z uniformemente espaçados em [-10, 10] para visualizar a curva em S.
z = np.linspace(-10, 10, 100)
# Aplica a sigmoide elemento a elemento: vetor de probabilidades entre ~0 e ~1.
sigmoid_values = sigmoid(z)

# Plota a curva: eixo x = escore linear z, eixo y = sigma(z) = P(classe 1).
plt.plot(z, sigmoid_values)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("σ(z)")
plt.grid()
plt.show()