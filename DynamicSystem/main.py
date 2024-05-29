from calc.jax_calc import *
import jax
from methods.numarical_methods import *
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Define the system matrix and initial conditions
A = jnp.array([[0, 1], [-1, 0]], dtype=jnp.float64)
dt = 0.1
ini = jnp.array([1, 0], dtype=jnp.float64)
steps = 63

# Forward, Backward, and Central setup
Uf = forward_setup(A, dt)
Ub = backward_setup(A, dt)
Uc = central_setup(A, dt)

# Compute the state propagation for forward, backward, and central methods
U1 = fast_calc(steps, method="F", include_init=True)(Uf, ini)
U2 = fast_calc(steps, method="B", include_init=True)(Ub, ini)

# For the central method, we need to compute the initial steps differently
u_0 = ini
u_1 = fast_calc(1, method="F", include_init=False)(Uf, ini)
ini_c = jnp.vstack([u_0, u_1])
U3 = fast_calc(steps, method="C", include_init=True)(Uc, ini_c)

# Extract the coordinates
u1, u2 = U1.T
a1, a2 = U2.T
c1, c2 = U3.T

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(u1, u2, "-ro", label='Forward Method')
plt.plot(a1, a2, "-ko", label='Backward Method')
plt.plot(c1, c2, "-bo", label='Central Method')
plt.grid(True)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('State Propagation Using Different Methods')
plt.legend()
plt.show()
