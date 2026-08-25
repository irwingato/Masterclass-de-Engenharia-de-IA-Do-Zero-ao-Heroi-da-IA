def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b!= 0:
        return a / b
    else:
        return "Division by zero is not allowed"

while True:
    print("\nMenu:")
    print("1. Addition")
    print("2. Subtractration")
    print("3. Multiplication")
    print("4. Division")
    print("5. Exit")

    choice = input("Enter your choice (1/2/3/4/5): ")

    if choice == "5":
        break

    num1 = float(input("Enter the first number: "))
    num2 = float(input("Enter the second number: "))

    if choice == "1":
        print("Result:", add(num1, num2))
    elif choice == "2":
        print("Result:", subtract(num1, num2))
    elif choice == "3":
        print("Result:", multiply(num1, num2))
    elif choice == "4":
        print("Result:", divide(num1, num2))
    else:
        print("Invalid choice")
        
# Versão alternativa com maior qualidade, robustez, legibilidade
# import operator

# OPERATIONS = {
#     "1": ("Addition", operator.add),
#     "2": ("Subtraction", operator.sub),
#     "3": ("Multiplication", operator.mul),
#     "4": ("Division", operator.truediv),
# }

# def get_number(prompt):
#     while True:
#         try:
#             return float(input(prompt))
#         except ValueError:
#             print("Por favor, digite um número válido.")

# def main():
#     while True:
#         print("\nMenu:")
#         for key, (name, _) in OPERATIONS.items():
#             print(f"{key}. {name}")
#         print("5. Exit")

#         choice = input("Enter your choice (1/2/3/4/5): ")
#         if choice == "5":
#             break

#         if choice not in OPERATIONS:
#             print("Invalid choice")
#             continue

#         num1 = get_number("Enter the first number: ")
#         num2 = get_number("Enter the second number: ")

#         # Tratamento especial para divisão por zero
#         if choice == "4" and num2 == 0:
#             print("Result: Division by zero is not allowed")
#             continue

#         func = OPERATIONS[choice][1]
#         print("Result:", func(num1, num2))

# if __name__ == "__main__":
#     main()