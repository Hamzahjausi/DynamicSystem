from jax import lax, jit, vmap
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


def fast_calc(n: int, method: str, include_init):

    def rollout_fn_FB(U_matrix, init_value):

        def scan_body(init_value, _):
            next_step = jnp.einsum("ij ,  j ", U_matrix, init_value)
            return next_step, next_step

        _, resaut = lax.scan(jit(scan_body, backend="cpu"), init_value, None, length=n)
        return resaut

    def rollout_fn_C(U_matrix, init_value):

        def scan_body(init_value, _):
            next_step = jnp.einsum(
                "ij ,j ->i", U_matrix[1], init_value[0]
            ) + jnp.einsum("ij,j ->i", U_matrix[0], init_value[1])
            init_value = jnp.roll(init_value, shift=-1, axis=0).at[-1].set(next_step)
            return init_value, next_step

        _, resaut = lax.scan(
            jit(scan_body, backend="cpu"), init_value, None, length=n + 1
        )
        return resaut

    method_map = {"F": rollout_fn_FB, "B": rollout_fn_FB, "C": rollout_fn_C}
    rollout_fn = method_map.get(method.upper())
    return rollout_fn
