import sympy as sp

# ============================================================================
# Exemplo: Integrais com SymPy (Cálculo Simbólico)
# ----------------------------------------------------------------------------
# O SymPy resolve integrais de forma ANALÍTICA (simbólica),
# ou seja, retorna a função antiderivada (fórmula) em vez de aproximar
# numericamente como faria um método de Simpson/Riemann.
# ============================================================================

x = sp.Symbol('x')          # Declara 'x' como símbolo matemático
f = x**2                    # Define a função f(x) = x²

# INTEGRAL DEFINIDA: área sob a curva de f(x) entre x=0 e x=2
#   ∫₀² x² dx = [x³/3]₀² = (2³/3) - (0³/3) = 8/3
definite_integral = sp.integrate(f, (x, 0, 2))

# INTEGRAL INDEFINIDA (primitiva): F(x) tal que F'(x) = f(x)
#   ∫ x² dx = x³/3 + C  (o SymPy omite a constante C)
indefinite_integral = sp.integrate(f, x)

print("Definite Integral:", definite_integral)    # Saída esperada: 8/3
print("Indefinite Integral:", indefinite_integral) # Saída esperada: x**3/3