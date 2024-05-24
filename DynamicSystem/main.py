import jax as jnp
from calc.jax_calc import *
from methods.numarical_methods import *
import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

import jax.lax as lax


A = jnp.array([[-0.1, 1], [-1, -0.2]], dtype=jnp.float64)
dt = 0.1
ini = jnp.array([1, 0], dtype=jnp.float64)
Uf = forward_setup(A, dt)
t = jnp.linspace(0, 2 * jnp.pi, 63, dtype=jnp.float64)
U1 = fast_calc(64)(Uf, ini)
u1, u2 = U1.T

Ub = backward_setup(A, dt)
U2 = fast_calc(64)(Ub, ini)
a1, a2 = U2.T


plt.figure()
plt.plot(u1, u2, "-ro")
plt.plot(a1, a2, "-ko")
plt.grid(True)
plt.show()
print("this is new line ")
