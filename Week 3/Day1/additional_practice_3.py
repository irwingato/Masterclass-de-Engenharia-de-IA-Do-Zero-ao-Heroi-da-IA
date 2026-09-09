import numpy as np

# =============================================================================
# MATRIZ DIAGONAL EM BLOCO (BLOCK DIAGONAL MATRIX)
# =============================================================================
#
# MATEMÁTICA:
# Uma matriz diagonal em bloco é uma matriz onde os blocos estão posicionados
# na diagonal principal, e tudo fora desses blocos é ZERO.
#
# Exemplo com 3 blocos A (2x2), B (2x3), C (1x1):
#
#     | a11 a12 | 0 0 0 | 0 |
#     | a21 a22 | 0 0 0 | 0 |
#     |---------+-------+---|
#     | 0   0   | b1 b1 b1| 0 |
#     | 0   0   | b2 b2 b2| 0 |
#     |---------+-------+---|
#     | 0   0   | 0  0  0 | c |
#
# Essa estrutura é útil em:
# - Sistemas de equações desacoplados
# - Processamento de sinais
# - Machine Learning (blocos independentes)
# =============================================================================

def criar_diagonal_bloco(lista_de_blocos):
    """
    Cria uma matriz diagonal em bloco a partir de uma lista de matrizes.
    
    Parâmetros:
        lista_de_blocos: Lista de arrays NumPy representando os blocos diagonais
    
    Retorna:
        Uma única matriz maior com os blocos posicionados na diagonal
    """
    
    # 1. Calcula as dimensões totais da matriz final
    # Somamos todas as linhas e todas as colunas dos blocos individuais
    # para saber o tamanho da matriz resultante
    linhas_totais = sum([b.shape[0] for b in lista_de_blocos])
    colunas_totais = sum([b.shape[1] for b in lista_de_blocos])

    # 2. Cria uma matriz de zeros com as dimensões calculadas
    # np.zeros cria uma matriz preenchida com zeros
    # Esses zeros serão substituídos pelos valores dos blocos na diagonal
    matriz_final = np.zeros((linhas_totais, colunas_totais))
    
    # 3. Posiciona cada bloco na diagonal correspondente
    # Usamos slicing de array para colocar cada bloco na posição correta
    # linha_atual e coluna_atual rastreiam onde colocar o próximo bloco
    linha_atual, coluna_atual = 0, 0
    for bloco in lista_de_blocos:
        # r = número de linhas do bloco, c = número de colunas do bloco
        r, c = bloco.shape
        # Slicing: matriz_final[linha_inicio:linha_fim, col_inicio:col_fim]
        # Copia os valores do bloco para essa posição na matriz final
        matriz_final[linha_atual:linha_atual+r, coluna_atual:coluna_atual+c] = bloco
        # Avança a posição para o próximo bloco (deslocamento)
        linha_atual += r
        coluna_atual += c
    
    return matriz_final

# --- TESTANDO A FUNÇÃO ---

# Criando blocos de tamanhos e formatos diferentes
# Cada bloco pode ter dimensões diferentes, a função se adapta
A = np.array([[1, 2], 
              [3, 4]])  # Bloco 2x2

B = np.array([[5, 6, 7], 
              [8, 9, 1]])  # Bloco 2x3

C = np.array([[9]])  # Bloco 1x1

# Gerando a matriz em bloco
# A matriz resultante terá dimensão (2+2+1) x (2+3+1) = 5x6
resultado = criar_diagonal_bloco([A, B, C])

print("Matriz diagonal em Bloco resultante:\n")
print(resultado)
print("\nDimensão:", resultado.shape)