import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

def forward_setup(system_matrix, dt):
    """
    Sets up the forward Euler method.
    
    The forward Euler method approximates the next state as:
    x(t + dt) ≈ (I + dt * A) * x(t)
    
    Args:
    system_matrix (array): The system matrix A.
    dt (float): The time step.
    
    Returns:
    array: The transformation matrix for the forward Euler method.
    """
    # Identity matrix of the same dimension as system_matrix
    I = jnp.eye(system_matrix.shape[0])
    
    # Transformation matrix for forward Euler method
    U_matrix = I + dt * system_matrix
    
    return U_matrix

def backward_setup(system_matrix, dt):
    """
    Sets up the backward Euler method.
    
    The backward Euler method approximates the next state as:
    x(t + dt) ≈ (I - dt * A)^(-1) * x(t)
    
    Args:
    system_matrix (array): The system matrix A.
    dt (float): The time step.
    
    Returns:
    array: The transformation matrix for the backward Euler method.
    """
    # Identity matrix of the same dimension as system_matrix
    I = jnp.eye(system_matrix.shape[0])
    
    # Transformation matrix for backward Euler method
    U_matrix = jnp.linalg.inv(I - dt * system_matrix)
    
    return U_matrix

def analytical_setup(A, ini):
    """
    Sets up the analytical solution for the system.
    
    The analytical solution involves diagonalizing the system matrix A.
    A = X * Λ * X^(-1)
    The solution is given by:
    x(t) = X * exp(Λ * t) * c
    where c = X^(-1) * x(0)
    
    Args:
    A (array): The system matrix A.
    ini (array): The initial state vector x(0).
    
    Returns:
    tuple: (X, Λ, c) where
           X (array) - matrix of eigenvectors of A,
           Λ (array) - diagonal matrix of eigenvalues of A,
           c (array) - initial state in the eigenvector basis.
    """
    # Eigen decomposition of the system matrix A
    Lambda, X = jnp.linalg.eig(A)
    
    # Initial state in the eigenvector basis
    c = jnp.linalg.inv(X) @ ini
    
    return X, Lambda, c

def central_setup(system_matrix, dt):
    """
    Sets up the central difference method.
    
    The central difference method uses the approximation:
    x(t + dt) ≈ x(t - dt) + 2 * dt * A * x(t)
    
    Args:
    system_matrix (array): The system matrix A.
    dt (float): The time step.
    
    Returns:
    tuple: (U_matrix, I) where
           U_matrix (array) - 2 * dt * A,
           I (array) - identity matrix of the same dimension as system_matrix.
    """
    # Transformation matrix for central difference method
    U_matrix = 2 * dt * system_matrix
    
    # Identity matrix of the same dimension as system_matrix
    I = jnp.eye(system_matrix.shape[0])
    
    return U_matrix, I
