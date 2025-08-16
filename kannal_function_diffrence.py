import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the polynomial function
def polynomial(x, coeffs):
    return x + coeffs[0] * x**3 + coeffs[1] * x**5 + coeffs[2] * x**7 + coeffs[3] * x**9

# Define coefficients for both functions
coeffs1 = [0.1, -0.05, 0.02, -0.01]  # Example coefficients for f1(x)
coeffs2 = [0.33310441, 0.13579791, 0.04324891, 0.04075595]  # Example coefficients for f2(x)

# Define the interval
x_min, x_max = 0, .8  # Example interval
x_vals = np.linspace(x_min, x_max, 1000)

# Compute function values
f1_vals = polynomial(x_vals, coeffs1)
f2_vals = polynomial(x_vals, coeffs2)
diff_vals = np.abs(f1_vals - f2_vals)

# Compute numerical differences
max_diff = np.max(diff_vals)
integrated_diff, _ = quad(lambda x: np.abs(polynomial(x, coeffs1) - polynomial(x, coeffs2)), x_min, x_max)
mse = np.mean(diff_vals ** 2)

# Plot both functions
plt.figure(figsize=(10, 5))
plt.plot(x_vals, f1_vals, label="f1(x)", linewidth=2)
plt.plot(x_vals, f2_vals, label="f2(x)", linewidth=2, linestyle="dashed")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Comparison of Two Polynomial Functions")
plt.legend()
plt.grid()
plt.show()

# Plot the difference
plt.figure(figsize=(10, 5))
plt.plot(x_vals, diff_vals, label="|f1(x) - f2(x)|", color="red")
plt.xlabel("x")
plt.ylabel("Absolute Difference")
plt.title("Absolute Difference Between Functions")
plt.legend()
plt.grid()
plt.show()

# Print numerical results
print(f"Maximum absolute difference: {max_diff:.5f}")
print(f"Integrated absolute difference: {integrated_diff:.5f}")
print(f"Mean squared error (MSE): {mse:.5f}")
