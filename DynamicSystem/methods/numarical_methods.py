import jax.numpy as jnp
from jax import config
config.update ("jax_enable_x64", True)


def forward_setup(system_matrix, dt):
    U_matrix = jnp.eye(system_matrix.shape[0]) +\
    dt * system_matrix
    return  U_matrix

def backward_setup(system_matrix, dt):
    I = jnp.eye(system_matrix.shape[0])
    U_matrix = jnp.linalg.inv(I - dt *
                               system_matrix)
    return  U_matrix

def analytical_setup (A,
                       ini) : 
    Lambda , X = jnp.linalg.eig (A)
    c= jnp.linalg.inv (X)@ini
    return X,Lambda,c 
    


