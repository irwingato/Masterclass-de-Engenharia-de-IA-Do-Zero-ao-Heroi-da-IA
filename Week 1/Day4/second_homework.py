# Dicionário para armazenar os alunos e suas notas
student_grades = {}

# Solicita a quantidade de alunos
while True:
    try:
        quantidade_alunos = int(
            input("Quantos alunos deseja cadastrar? ")
        )

        if quantidade_alunos <= 0:
            print("Digite uma quantidade maior que zero.")
        else:
            break

    except ValueError:
        print("Digite um número inteiro válido.")


# Solicita a quantidade de notas por aluno
while True:
    try:
        quantidade_notas = int(
            input("Quantas notas cada aluno terá? ")
        )

        if quantidade_notas <= 0:
            print("Digite uma quantidade maior que zero.")
        else:
            break

    except ValueError:
        print("Digite um número inteiro válido.")


# Cadastro dos alunos
for numero_aluno in range(quantidade_alunos):
    print(f"\n--- Cadastro do {numero_aluno + 1}º aluno ---")

    # Solicita um nome válido
    while True:
        nome = input("Digite o nome do aluno: ").strip()

        if not nome:
            print("O nome não pode ficar vazio.")
        elif nome in student_grades:
            print("Esse aluno já foi cadastrado.")
        else:
            break

    # Lista para armazenar as notas do aluno
    notas = []

    # Solicita uma nota por vez
    for numero_nota in range(quantidade_notas):
        while True:
            try:
                nota = float(
                    input(
                        f"Digite a {numero_nota + 1}ª nota de {nome}: "
                    ).replace(",", ".")
                )

                if nota < 0 or nota > 100:
                    print("A nota deve estar entre 0 e 100.")
                else:
                    notas.append(nota)
                    break

            except ValueError:
                print("Digite uma nota válida.")

    # Armazena o aluno e suas notas no dicionário
    student_grades[nome] = notas


# Exibe o resultado final
print("\n--- RESULTADO FINAL ---")

for nome, notas in student_grades.items():
    media = sum(notas) / len(notas)

    if media >= 70:
        situacao = "Aprovado"
    elif media >= 50:
        situacao = "Recuperação"
    else:
        situacao = "Reprovado"

    print(f"\nAluno: {nome}")
    print(f"Notas: {notas}")
    print(f"Média: {media:.2f}")
    print(f"Situação: {situacao}")