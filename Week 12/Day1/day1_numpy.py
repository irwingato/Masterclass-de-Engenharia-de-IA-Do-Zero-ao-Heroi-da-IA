import numpy as np

# Mecanismo de Atenção (como no Transformer):
# scores = Q @ K^T  -> similaridade entre queries e keys
# attention_weights = softmax(scores) -> normaliza os scores para soma = 1
# context = attention_weights @ V -> soma ponderada das values
#
# Define query, keys and values
# Q (queries): representa o que o modelo quer saber / a consulta atual
# K (keys): candidatas que podem atender a query
# V (values): valores associados às keys, resultado final da atenção
queries = np.array([[1, 0, 1], [0, 1, 1]])
keys = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
values = np.array([[10, 0], [0, 10], [5, 5]])

# Compute attention scores 
# Matematicamente: scores[i, j] = query_i · key_j  (produto escalar / inner product)
# A entrada [i, j] indica o quanto a query i deve prestar atenção na key j
scores = np.dot(queries, keys.T)

# Apply softmax to normalize scores
# softmax(x)_i = e^(x_i - max) / Σ_j e^(x_j - max)  -> distribui em probabilidades
def softmax(x):
    # Subtrai o máximo para estabilidade numérica antes de aplicar exp
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # Soma ao longo da última dimensão (axis=-1) com keepdims para broadcast correto
    # Garante que cada linha da matriz resultante soma 1 (distribuição de probabilidade)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)

# Compute weighted sum of values
# context[i] = Σ_j attention_weights[i, j] * values[j]  (soma ponderada)
context = np.dot(attention_weights, values)

# Exibe os pesos de atenção e o contexto final
print("Attention Weights: \n", attention_weights)
print("Context Vector:\n", context)