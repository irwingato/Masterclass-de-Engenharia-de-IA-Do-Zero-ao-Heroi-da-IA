import numpy as np

# 1. Geramos uma matriz 4x4 aleatória com valores inteiros de 1 a 9
# convertida para float para permitir cálculos com autovalores complexos
np.random.seed(42) # Semente para reprodutibilidade
A = np.random.randint(1, 10, size=(4, 4)).astype(float)

# 2. Calculamos os autovalores (eigenvalues) da matriz A
# A propriedade fundamental é: det(A - λI) = 0
# Onde λ é o autovalor e I é a matriz identidade
# Se o determinante de (A - λI) for 0, então λ é de fato um autovalor de A
eigenvalues, _ = np.linalg.eig(A)

print("--- Verificação da Propriedade det(A - λI) = 0 ---")

# 3. Criamos a matriz identidade I do mesmo tamanho (4x4)
I = np.eye(4)

# 4. Vamos testar a propriedade para CADA um dos autovalores encontrados
# Para cada autovalor λ, calculamos det(A - λI) e verificamos se é ~0
for i, lam in enumerate(eigenvalues):
    # Calcula a matriz (A - lambda * I)
    # Se λ é um autovalor, esta matriz deve ser singular (det = 0)
    matriz_subtraida = A - lam * I
    
    # Calcula o determinante dessa nova matriz
    # O resultado deve ser muito próximo de 0 (ou 0 em precisão exata)
    determinante_resíduo = np.linalg.det(matriz_subtraida)

    print(f"\nAutovalor λ{i+1}: {lam}")
    print(f"-> det(A - λ{i+1}*I) = {determinante_resíduo}")