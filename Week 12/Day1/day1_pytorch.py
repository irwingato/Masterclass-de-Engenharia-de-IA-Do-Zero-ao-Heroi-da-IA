import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt 

# Mecanismo de Atenção (como no Transformer):
# scores = Q @ K^T  -> similaridade entre queries e keys
# attention_weights = softmax(scores) -> normaliza os scores para soma = 1
# context = attention_weights @ V -> soma ponderada das values
#
# Define query, keys and values
# Q (queries): representa o que o modelo quer saber / a consulta atual
# K (keys): candidatas que podem atender a query
# V (values): valores associados às keys, resultado final da atenção
queries = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
keys = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
values = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])

# Compute attention scores
# scores = Q @ K^T  (matmul = produto matricial / inner products)
# Cada elemento scores[i, j] = query_i · key_j
scores = torch.matmul(queries, keys.T)

# Apply softmax to normaliza scores
# F.softmax aplica a função softmax sobre a última dimensão (dim=-1)
# softmax converte os scores em pesos que somem 1 (probabilidades)
attention_weights = F.softmax(scores, dim=-1)

# Compute weighted sum of values
# context = attention_weights @ values  (soma ponderada das values)
context = torch.matmul(attention_weights, values)

print("Attention Weights: \n", attention_weights)
print("Context Vector:\n", context)

# Visualize attention weights
# plt.matshow exibe a matriz de pesos de atenção como heatmap
plt.matshow(attention_weights)
plt.colorbar()
plt.title("Attention Weights")
plt.show()