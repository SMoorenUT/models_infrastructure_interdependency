import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Original expensive function
def expensive_function(x):
    return np.sin(3 * x) + x**2

# Training data (assume these are the expensive evaluations)
X_train = np.atleast_2d(np.linspace(-2, 2, 10)).T
y_train = expensive_function(X_train).ravel()

# Define a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))  # Constant * RBF kernel
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit to training data
gp.fit(X_train, y_train)

# Predict using surrogate model
X_test = np.atleast_2d(np.linspace(-2.5, 2.5, 500)).T
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(X_test, expensive_function(X_test), 'r--', label="True function")
plt.plot(X_test, y_pred, 'b-', label="Surrogate model (GPR)")
plt.fill_between(X_test.ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma, alpha=0.2, label="95% CI")
plt.scatter(X_train, y_train, c='black', label="Training points")
plt.title("Surrogate Modeling with Gaussian Process Regression")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
