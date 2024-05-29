from calc.jax_calc import *
import jax
from methods.numarical_methods import *
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

import jax.lax as lax

A = jnp.array([[0, 1], [-1, 0]])
dt = 0.1
ini = jnp.array([1, 0], dtype=jnp.float64)
Uf = forward_setup(A, dt)
Ub = backward_setup(A, dt)
steps = 63
U1 = fast_calc(steps, method="F", include_init=True)(Uf, ini)
u1, u2 = U1.T
U2 = fast_calc(steps, method="B", include_init=True)(Ub, ini)
a1, a2 = U2.T

Uc = central_setup(A, dt)
u_0 = ini
u_1 = fast_calc(1, method="F", include_init=False)(Uf, ini)
ini_c = jnp.stack([u_0, *u_1])
U3 = fast_calc(steps, method="C", include_init=True)(Uc, ini_c)
c1, c2 = U3.T

plt.figure()
plt.plot(u1, u2, "-ro")
plt.plot(a1, a2, "-ko")
plt.plot(c1, c2, "-bo")
plt.grid(True)
plt.show()

"""
this is new comment to test 
vim git ;)

"""
