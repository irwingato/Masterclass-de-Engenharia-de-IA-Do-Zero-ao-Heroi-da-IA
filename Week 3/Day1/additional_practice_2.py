import numpy as np

# =============================================================================
# PROPRIEDADES DA MULTIPLICAÇÃO MATRICIAL
# =============================================================================
# IMPORTANTE: A multiplicação de matrizes é DIFERENTE da multiplicação
# de números comuns. Uma das principais diferenças é que ela NÃO é comutativa.
#
# Operador @: Multiplicação matricial em Python/NumPy
# Equivalente a: np.dot(A, B) ou A.dot(B)
# =============================================================================

# Criando matrizes 2x2 para o teste
A = np.array([[2, 3], [1, 4]])

B = np.array([[5, 6], [7, 8]])

C = np.array([[9, 10], [11, 12]])

# Matriz Identidade (Elemento neutro)
# np.eye(2) cria uma matriz 2x2 identidade: [[1,0],[0,1]]
# Propriedade: A @ I = I @ A = A (qualquer matriz multiplicada pela identidade
# permanece igual, assim como multiplicar por 1 em números)
I = np.eye(2)

# --- 1. NÃO COMUTATIVA ---
# MATEMÁTICA: A × B ≠ B × A (em geral)
# Isso é diferente da multiplicação de números onde 2×3 = 3×2
# Na multiplicação matricial, a ORDEM importa!
print("--- 1. NÃO COMUTATIVA (A @ B != B @ A) ---")
AB = A @ B  # Multiplica A por B
BA = B @ A  # Multiplica B por A (ordem trocada)
print("A @ B: \n", AB)
print("B @ A: \n", BA)
print("São iguais: ", np.array_equal(AB, BA))

# --- 2. ASSOCIATIVA ---
# MATEMÁTICA: (A × B) × C = A × (B × C)
# Quando multiplicamos 3 matrizes, a ordem das operações (parênteses) não importa
# O resultado final será o mesmo
print("\n--- 2. ASSOCIATIVA [ (A @ B) @ C == A @ (B @ C) ] ---")
lado_esquerdo_assoc = (A @ B) @ C  # Primeiro A×B, depois resultado×C
lado_direito_assoc = A @ (B @ C)   # Primeiro B×C, depois A×resultado
# np.allclose é usado para evitar problemas com arredondamento de ponto flutuante
# Compara se dois arrays são "quase iguais" (com tolerância de 1e-8)
print("Válida?", np.allclose(lado_esquerdo_assoc, lado_direito_assoc))

# --- 3. DISTRIBUTIVA ---
# MATEMÁTICA: A × (B + C) = (A × B) + (A × C)
# Podemos distribuir a multiplicação sobre a soma, assim como com números
print("\n--- 3. DISTRIBUTIVA [ A @ (B + C) == (A @ B) + (A @ C) ] ---")
lado_esquerdo_dist = A @ (B + C)      # A vezes (soma de B e C)
lado_direito_dist = (A @ B) + (A @ C)  # Soma de (A×B) com (A×C)
print("Válida?", np.allclose(lado_esquerdo_dist, lado_direito_dist))

# --- 4. ELEMENTO NEUTRO ---
# MATEMÁTICA: A × I = I × A = A
# A matriz identidade é o "elemento neutro" da multiplicação matricial
# Assim como multiplicar qualquer número por 1 não o altera,
# multiplicar qualquer matriz pela identidade não a altera
print("\n--- 4. ELEMENTO NEUTRO (A @ I == A == I @ A) ---")
AI = A @ I  # A multiplicada pela identidade
IA = I @ A  # Identidade multiplicada por A
print("A @ I igual a A?", np.allclose(AI, A))
print("I @ A igual a A?", np.allclose(IA, A))