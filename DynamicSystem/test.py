import jax.numpy as jnp
from jax import lax, config, jit, vmap
from calc.jax_calc import *
import matplotlib.pyplot as plt


def forward_setup(system_matrix, dt):
    U_matrix = jnp.eye(system_matrix.shape[0]) + dt * system_matrix
    return U_matrix


A = jnp.array([[0, 1], [-1, 0]])


def central_setup(system_matrix, dt):
    U_matrix = 2 * dt * system_matrix
    I = jnp.identity(system_matrix.shape[0])
    return U_matrix, I


u0 = jnp.array([1, 0])
dt = 0.01
UC = central_setup(A, dt)
UF = forward_setup(A, dt)
u1 = UF @ u0
n_steps = 630
U = jnp.zeros(shape=(n_steps, 2))

U = U.at[0].set(u0)
U = U.at[1].set(u1)

for i in range(1, n_steps):
    U = U.at[i + 1].set(UC[0] @ U[i] + UC[1] @ U[i - 1])

u1, u2 = U.T

plt.plot(u1, u2, "-ro")
plt.grid(True)
plt.ylim(-2, 2)
plt.xlim(-2, 2)
plt.show()
