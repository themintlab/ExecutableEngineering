# --- CELL 7 ---
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre

# Data points
x = [0, .5, 2]
y = [1, 3, 2]

# Calculate the Lagrange basis polynomials
n = len(x)
P = []
for i in range(n):
  numerator = 1
  denominator = 1
  for j in range(n):
    if i != j:
      numerator = np.polymul(numerator, np.poly1d([1, -x[j]]))
      denominator = denominator * (x[i] - x[j])
  P.append(np.poly1d(np.polydiv(numerator, denominator)[0]))


# Plot the Lagrange basis polynomials
x_plot = np.linspace(-1, 3, 100)

for i in range(n):
    y_plot = P[i](x_plot)
    plt.plot(x_plot, y_plot, label=f'P_{i+1}(x)')

plt.scatter(x, [1] * len(x), color='black')
plt.scatter(x, [0] * len(x), color='red')
plt.scatter(x, [0] * len(x), color='red')

plt.xlabel('x')
plt.ylabel('P_i(x)')
plt.title('Lagrange Basis Polynomials')
plt.legend()
plt.grid(True)
plt.show()


# --- CELL 9 ---
import matplotlib.pyplot as plt
import numpy as np
# Construct the Lagrange polynomial
L = np.poly1d(0)
for i in range(n):
  L = L + y[i] * P[i]

# Plot the Lagrange polynomial
y_plot = L(x_plot)
plt.plot(x_plot, y_plot, label='L(x)')
plt.scatter(x, y, color='red', label='Data points')
plt.xlabel('x')
plt.ylabel('L(x)')
plt.title('Lagrange Polynomial Interpolation')
plt.legend()
plt.grid(True)
plt.show()


# --- CELL 12 ---
import sympy as sp

x = [0, 1, 2, 3, 4]
y = [1, 3, 2, 5, 7]

n = len(x)
x_sym = sp.Symbol('x')

L = 0
for i in range(n):
    term = y[i]
    for j in range(n):
        if i != j:
            term *= (x_sym - x[j]) / (x[i] - x[j])
    L += term


print(L)

print('which is an ugly way of writing out:')
print(L.simplify())


# --- CELL 28 ---
import matplotlib.pyplot as plt
import numpy as np

# Define the function
def f(x):
  return np.exp(-(x/2)**2)

# Create x values for plotting
x_toy = np.linspace(-6, 6, 100)
y_toy = f(x_toy)

# Sample 11 random points in the range -5 to 6
x_d = np.arange(-5, 6, 1)
y_d = f(x_d)

# Interpolate using numpy Legendre
from numpy.polynomial import legendre
coefficients = legendre.legfit(x_d, y_d, len(x_d) - 1)
legendre_polynomial = legendre.Legendre(coefficients)

# Create x values for plotting the interpolated polynomial
x_interp = np.linspace(-5.5, 5.5, 200)
y_interp = legendre_polynomial(x_interp)

# Plot the original curve, sampled points, and interpolated polynomial
plt.plot(x_toy, y_toy, label='exp(-(x/2)^2)')
plt.scatter(x_d, y_d, color='red', label='Sampled points')
plt.plot(x_interp, y_interp, label='Legendre Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Function, Sampled Points, and Legendre Interpolation')
plt.grid(True)
plt.show()

