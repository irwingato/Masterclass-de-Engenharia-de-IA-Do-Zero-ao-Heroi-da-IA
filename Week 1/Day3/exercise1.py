def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def print_factorial(n):
    result = factorial(n)
    print(f"The factorial of {n} is {result}")

# Um por entrada fixa
print_factorial(6)

# E outro pede entrada do usuário
n = int(input("Enter a number: "))
print_factorial(n)