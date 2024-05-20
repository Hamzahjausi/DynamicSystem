from jax import lax ,jit
import jax.numpy as jnp


def fast_calc (n,
               include_init = False 
               ):

    def rollout_fn (U_matrix , Un ):
        def scan_body (Un ,_) :
            Un_1 = U_matrix@ Un
            return Un_1 , Un_1
        _ , history = lax.scan (jit(scan_body , backend="cpu") ,Un , None ,
                                 length= n)
        if include_init :
            return jnp.concatenate ([
                jnp.expand_dims(Un , axis =0),history] ,
                axis = 0 
            )
        return history
    
    return rollout_fn

