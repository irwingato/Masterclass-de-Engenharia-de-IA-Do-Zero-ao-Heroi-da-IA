"""
day2_ex1.py - Forward pass do BERT pré-treinado.

Explicações matemáticas:
- A tokenização mapeia o texto bruto em uma sequência de token ids (busca no vocabulário).
- Os embeddings de token são vetores obtidos da matriz de embeddings.
- As codificações posicionais são adicionadas aos embeddings de token para preservar a ordem.
- A auto-atenção computa a matriz de atenção A = softmax(QK^T / sqrt(d_k)) V.
- O `last_hidden_state` tem shape (batch_size, sequence_length, hidden_dim).

Notas de ciência da computação:
- O forward pass é executado sem o `torch.no_grad()`.
"""

from transformers import BertTokenizer, BertModel

# Carregar o tokenizador e o modelo BERT pré-treinado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenizar um exemplo de entrada
text = "Transformers are powerful models for NLP tasks"
inputs = tokenizer(text, return_tensors='pt')

# Passar a entrada pelo modelo
outputs = model(**inputs)
print("Hidden States Shape:", outputs.last_hidden_state.shape)
