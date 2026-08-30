# Módulo 'sys' fornece acesso a variáveis e funções específicas do interpretador Python
import sys

# sys.argv: lista de argumentos passados na linha de comando ao executar o script
# (sys.argv[0] é o próprio nome do script; demais posições são os argumentos fornecidos)
print(sys.argv)

# sys.version: string com a versão do interpretador Python em execução
print(sys.version)

# ============ MÓDULO 'os' (interações com o sistema operacional) ============
# import os

# print(os.getcwd())          # Retorna o diretório de trabalho atual (caminho onde o script roda)
# os.mkdir("test_dir")        # Cria um novo diretório chamado "test_dir"
# os.remove("file.txt")       # Exclui um arquivo chamado "file.txt"

# ============ 'reduce' - FUNCIONAIS (functools) ============
# from functools import reduce

# numbers = [1, 2, 3, 4]
# product = reduce(lambda x,y: x * y, numbers)  # Acumula os valores aplicando a função de forma repetida:
#                                               # (((1*2)*3)*4) = 24
# print(product)

# ============ 'filter' - FILTRA ELEMENTOS DE UMA LISTA ============
# evenList = filter(lambda x: x % 2 == 0, numbers)  # Retorna apenas os números pares (x % 2 == 0)
# # print(list(evenList))  # 'filter' retorna um iterador; converter para list() para visualizar

# ============ 'map' - APLICA UMA FUNÇÃO A CADA ELEMENTO ============
# squares = map(lambda x: x**2, numbers)  # Eleva cada número ao quadrado
# # print(list(squares))

# ============ LIST COMPREHENSION (compreensão de lista) ============
# Sintaxe: [expression for item in iterable if condition]
# É uma forma concisa e idiomática de gerar listas em Python

# # Cria uma lista de quadrados de 0 a 9
# squares = [x**2 for x in range(10)]
# # print(squares)

# # Filtra números ímpares de 0 a 99 (x % 2 != 0)
# evens = [x for x in range(100) if x % 2 != 0]
# # print(evens)

# ============ 'lambda' - FUNÇÕES ANÔNIMAS ============
# Sintaxe: lambda arguments: expression
# Função de uma linha sem nome, útil para operações simples

# add = lambda x, y: x + y  # Equivale a: def add(x, y): return x + y
# # print(add(3,5))