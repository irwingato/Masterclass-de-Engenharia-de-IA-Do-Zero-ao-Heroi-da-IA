# ============================================================
# CONJUNTOS - SET
# ============================================================

# Cria um conjunto com os números 1, 2 e 3.
# Conjuntos não possuem elementos repetidos e não possuem
# uma ordem fixa de exibição.
set1 = {1, 2, 3}

# Cria outro conjunto com os números 3, 4 e 5.
set2 = {3, 4, 5}


# Subtração de conjuntos:
# Mostra os elementos que existem em set1, mas não existem em set2.
#
# Resultado:
# {1, 2}
print(set1 - set2)


# União de conjuntos:
# Junta os elementos de set1 e set2.
# O número 3 aparece apenas uma vez, pois conjuntos não aceitam
# elementos duplicados.
#
# Resultado possível:
# {1, 2, 3, 4, 5}
print(set1 | set2)


# Interseção de conjuntos:
# Mostra os elementos que existem simultaneamente nos dois conjuntos.
#
# Resultado:
# {3}
print(set1 & set2)


# ============================================================
# DICIONÁRIOS - DICT
# ============================================================

# Cria um dicionário chamado student.
#
# Um dicionário armazena dados no formato:
# chave: valor
#
# Neste caso:
# "name"  é a chave e "Alice" é o valor.
student = {
    "name": "Alice",
    "age": 25,
    "grade": "A"
}


# Percorre todas as chaves e valores do dicionário.
#
# O método .items() retorna pares no formato:
# chave, valor
#
# Em cada repetição:
# - key recebe o nome da chave;
# - value recebe o valor correspondente.
for key, value in student.items():
    print(key, value)


# Adiciona uma nova chave chamada "subject".
# Como essa chave ainda não existia, ela será adicionada ao dicionário.
student["subject"] = "Math"


# Atualiza o valor da chave "age".
# O valor 25 será substituído por 32.
student["age"] = 32


# Exibe o dicionário atualizado.
#
# Resultado:
# {
#     "name": "Alice",
#     "age": 32,
#     "grade": "A",
#     "subject": "Math"
# }
print(student)


# Remove a chave "grade" e o valor associado a ela.
del student["grade"]


# Exibe o dicionário depois da remoção de "grade".
#
# Resultado:
# {
#     "name": "Alice",
#     "age": 32,
#     "subject": "Math"
# }
print(student)


# Remove a chave "subject" e o valor "Math".
#
# O método .pop() remove um item do dicionário usando sua chave.
student.pop("subject")


# Exibe o dicionário depois da remoção de "subject".
#
# Resultado:
# {
#     "name": "Alice",
#     "age": 32
# }
print(student)


# ============================================================
# TUPLAS - TUPLE
# ============================================================

# Cria uma tupla com três elementos.
#
# Tuplas são coleções ordenadas e imutáveis.
# Isso significa que, depois de criada, uma tupla não pode
# ter seus elementos alterados, adicionados ou removidos.
colors = ("red", "green", "blue")


# Cria uma tupla com apenas um elemento.
#
# A vírgula é obrigatória para que o Python reconheça isso
# como uma tupla.
#
# Sem a vírgula, ("glass") seria apenas uma string entre parênteses.
single_item = ("glass",)


# Acessa o primeiro elemento da tupla.
# Os índices começam em 0:
# 0 -> "red"
print(colors[0])


# Acessa o último elemento da tupla.
# O índice -1 representa o último elemento.
print(colors[-1])


# ============================================================
# LISTAS - LIST
# ============================================================

# Cria uma lista de números inteiros.
#
# Listas são ordenadas, podem ser alteradas e aceitam elementos
# repetidos.
numbers = [1, 2, 3, 4, 5]


# Cria uma lista de frutas.
fruits = ["apple", "banana", "cherry"]


# Cria uma lista com valores de tipos diferentes:
# - número inteiro;
# - texto;
# - valor booleano.
mixed = [1, "apple", True]


# Acessa o terceiro elemento da lista numbers.
#
# Como a contagem começa em zero:
# índice 0 -> 1
# índice 1 -> 2
# índice 2 -> 3
#
# Resultado:
# 3
print(numbers[2])


# Acessa o último elemento da lista fruits.
#
# O índice -1 representa o último item.
#
# Resultado:
# "cherry"
print(fruits[-1])


# Acessa o segundo elemento da lista mixed.
#
# Índice 1 corresponde ao valor "apple".
print(mixed[1])


# Adiciona "orange" ao final da lista fruits.
#
# Depois dessa operação:
# ["apple", "banana", "cherry", "orange"]
fruits.append("orange")


# Insere "grape" na posição de índice 1.
#
# O elemento que estava nessa posição e os seguintes
# são deslocados para a direita.
#
# Antes:
# ["apple", "banana", "cherry", "orange"]
#
# Depois:
# ["apple", "grape", "banana", "cherry", "orange"]
fruits.insert(1, "grape")


# Exibe a lista depois das operações append() e insert().
print(fruits)


# Cria uma nova lista contendo os elementos dos índices 2 até 3.
#
# Em um fatiamento, o índice final não é incluído.
#
# Lista atual:
# índice 0 -> "apple"
# índice 1 -> "grape"
# índice 2 -> "banana"
# índice 3 -> "cherry"
# índice 4 -> "orange"
#
# fruits[2:4] pega os índices 2 e 3:
# ["banana", "cherry"]
slicedFruits = fruits[2:4]


# Exibe a lista fatiada.
print(slicedFruits)


# Remove o primeiro elemento encontrado com o valor "banana".
#
# Depois da remoção:
# ["apple", "grape", "cherry", "orange"]
fruits.remove("banana")


# Exibe a lista depois da remoção de "banana".
print(fruits)


# Remove o elemento localizado no índice 0.
#
# Nesse momento, o índice 0 contém "apple".
#
# Depois da remoção:
# ["grape", "cherry", "orange"]
del fruits[0]


# Exibe a lista depois da remoção do primeiro elemento.
print(fruits)


# Remove e retorna o último elemento da lista.
#
# Nesse momento, o último elemento é "orange".
#
# Depois da operação:
# ["grape", "cherry"]
fruits.pop()


# Exibe a lista final.
print(fruits)