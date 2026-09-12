import sympy as sp

# ============================================================================
# Exemplo: Integrais Impróprias com SymPy
# ----------------------------------------------------------------------------
# Este script demonstra integrais da exponencial f(x) = e^(-x), muito
# usadas em probabilidade (distribuição exponencial) e aprendizado de máquina.
# ============================================================================

# Define a function
x = sp.Symbol('x')
f = sp.exp(-x)             # f(x) = e^(-x)

# Compute indefinite integral
#   ∫ e^(-x) dx = -e^(-x) + C
# Pois d/dx [-e^(-x)] = -(-e^(-x)) = e^(-x).
indefinite_integral = sp.integrate(f, x)
print("Indefinite Integral:", indefinite_integral)   # Saída: -exp(-x)

# Compute definite integral
#   ∫₀^∞ e^(-x) dx : integral IMPRÓPRIA (limite superior = infinito).
#   Avaliação: [ -e^(-x) ]₀^∞ = lim(t→∞) -e^(-t) - (--e^0) = 0 + 1 = 1.
#   Este resultado é a base da distribuição exponencial (área total = 1).
definite_integral = sp.integrate(f, (x, 0, sp.oo))
print("Definite Integral:", definite_integral)      # Saída: 1