from jax import lax, jit, vmap
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


def fast_calc(n, include_init=False):

    def rollout_fn(U_matrix, Un):
        def scan_body(Un, _):
            Un_1 = jnp.einsum("ij , j ", U_matrix, Un)
            return Un_1, Un_1

        _, history = lax.scan(jit(scan_body, backend="cpu"), Un, None, length=n)
        if include_init:
            return jnp.concatenate([jnp.expand_dims(Un, axis=0), history], axis=0)
        return history

    return rollout_fn


def fast_analytical_solver(X, Lambda, c):
    vectorize = lambda func: vmap(jit(func))

    def process(t):

        def calc(t):
            Lambda_t_exp = jnp.diag(jnp.exp(Lambda * t))
            return jnp.einsum("ij,jh,h", X, Lambda_exp, c)

        return vectorize(calc)(t)

    return process
