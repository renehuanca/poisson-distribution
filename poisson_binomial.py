
import numpy as np
import pandas as pd
from scipy.stats import binom, poisson

# Parámetros
n = 1000       # número de ensayos
p = 0.001      # probabilidad de éxito (defectuoso)
lmbda = n * p  # parámetro de Poisson

# Rango de valores de x
x_values = np.arange(0, 20)

# Distribuciones
binomial_probs = binom.pmf(x_values, n, p)
poisson_probs = poisson.pmf(x_values, lmbda)

# Tabla comparativa
df = pd.DataFrame({
    'x': x_values,
    'Binomial_P(X=x)': binomial_probs,
    'Poisson_P(X=x)': poisson_probs
})

# Cálculos solicitados
p_x0_binom = binom.pmf(0, n, p)
p_x0_poiss = poisson.pmf(0, lmbda)

p_xge3_binom = 1 - binom.cdf(2, n, p)
p_xge3_poiss = 1 - poisson.cdf(2, lmbda)

p_x4_binom = binom.pmf(4, n, p)
p_x4_poiss = poisson.pmf(4, lmbda)

# Mostrar resultados
print("Tabla comparativa Binomial vs Poisson (x de 0 a 10):")
print(df.to_string(index=False, float_format="{:,.8f}".format))


print("\nProbabilidad de que ningún motor sea defectuoso (X = 0):")
print("Binomial:", round(p_x0_binom, 6))
print("Poisson:", round(p_x0_poiss, 6))

print("\nProbabilidad de que haya 3 o más motores defectuosos (X ≥ 3):")
print("Binomial:", round(p_xge3_binom, 6))
print("Poisson:", round(p_xge3_poiss, 6))

print("\nProbabilidad de que haya exactamente 4 motores defectuosos (X = 4):")
print("Binomial:", round(p_x4_binom, 6))
print("Poisson:", round(p_x4_poiss, 6))