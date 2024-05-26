from calc.jax_calc import *
from methods.numarical_methods import *
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

import jax.lax as lax

A = jnp.array([[0, 1], [-1, 0]])
dt = 0.1
ini = jnp.array([1, 0], dtype=jnp.float64)
Uf = forward_setup(A, dt)
U1 = fast_calc(62, include_init=True)(Uf, ini)
u1, u2 = U1.T

Ub = central_setup(A, dt)
U2 = fast_calc(63, include_init=True)(Ub, ini)
a1, a2 = U2.T


plt.figure()
plt.plot(u1, u2, "-ro")
plt.plot(a1, a2, "-ko")
plt.show()
"""
this is new comment to test 
vim git ;)

"""
