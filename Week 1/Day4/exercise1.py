# Cria um dicionário chamado "person" com informações sobre uma pessoa:
# nome, idade e nota.
person = {"name": "Alice", "age": 25, "grade": "A"}

# Exibe o conteúdo inicial do dicionário.
print(person)

# Adiciona um novo par de chave e valor ao dicionário.
# A chave é "address" e o valor é o endereço da pessoa.
person["address"] = "123 Main St"

# Atualiza o valor associado à chave "age".
# A idade passa de 25 para 32 anos.
person["age"] = 32

# Verifica se a chave "grade" existe no dicionário.
if "grade" in person:
    # Remove a chave "grade" e o seu respectivo valor.
    del person["grade"]

# Exibe o conteúdo final do dicionário após todas as alterações.
print(person)