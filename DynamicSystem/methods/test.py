import jax.numpy as jnp
import matplotlib.pyplot as plt

u0 = jnp.array([1, 0])
dt = 0.1
A = jnp.array([[1, dt], [-dt, 1]])

I = jnp.identity(2)

U = jnp.zeros(shape=(63, 2))

U = U.at[0].set(u0)

for i in range(1, 63):
    U = U.at[i].set(A @ U[i - 1])

u1, u2 = U.T

plt.plot(u1, u2)
plt.show()
