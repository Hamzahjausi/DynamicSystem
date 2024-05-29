import jax
import jax.numpy as jnp
from jax import lax, jit, config
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)


def fast_calc(n, method: str, include_init=False):
    def rollout_fn_f_b(U_matrix, Un):
        def scan_body(Un, _):
            Un_1 = U_matrix @ Un
            return Un_1, Un_1

        _, history = lax.scan(jit(scan_body, backend="cpu"), Un, None, length=n)
        if include_init:
            return jnp.concatenate([jnp.expand_dims(Un, axis=0), history], axis=0)
        return history

    def rollout_fn_c(UC, UN):
        def scan_body(INI, _):
            step = jnp.einsum("ij,j->i", UC[1], INI[0]) + jnp.einsum(
                "ij,j->i", UC[0], INI[1]
            )
            INI = jnp.roll(INI, shift=-1, axis=0).at[-1].set(step)
            return INI, step

        _, history = lax.scan(scan_body, UN, None, length=n)
        if include_init:
            return jnp.concatenate([jnp.expand_dims(UN[0], axis=0), history], axis=0)
        return history

    method_map = {
        "F": rollout_fn_f_b,
        "B": rollout_fn_f_b,
        "C": rollout_fn_c,
    }

    rollout_fn = method_map.get(method.upper())
    if not rollout_fn:
        raise ValueError(f"Unknown method '{method}'")

    return rollout_fn


def backward_setup(system_matrix, dt):
    I = jnp.eye(system_matrix.shape[0])
    U_matrix = I - dt * system_matrix
    return U_matrix


def forward_setup(system_matrix, dt):
    U_matrix = jnp.eye(system_matrix.shape[0]) + dt * system_matrix
    return U_matrix


def central_setup(system_matrix, dt):
    U_matrix = 2 * dt * system_matrix
    I_matrix = jnp.eye(system_matrix.shape[0])
    return U_matrix, I_matrix


A = jnp.array([[0, 1], [-1, -0.3]], dtype=jnp.float64)
dt = 0.01
ini = jnp.array([1, 0], dtype=jnp.float64)
n = int(2 * jnp.pi / dt)
UC = central_setup(A, dt)
UF = forward_setup(A, dt)
ini_next_step = fast_calc(n=1, method="F", include_init=False)(UF, ini)

INI = jnp.stack([ini, *ini_next_step])
calc_fn = fast_calc(n, method="C", include_init=True)
U = calc_fn(UC, INI).T
plt.plot(*U, "-ro")
plt.show()
