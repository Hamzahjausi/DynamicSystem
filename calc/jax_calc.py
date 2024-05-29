from jax import lax, jit, vmap
import jax.numpy as jnp
import jax

# Enable 64-bit precision in JAX for better numerical accuracy
jax.config.update("jax_enable_x64", True)

# Function to perform fast calculations based on the chosen method
def fast_calc(n: int, method: str, include_init):
    """
    Calculates the steps for the given number of iterations 'n' using the specified method.
    
    Args:
    n (int): Number of iterations.
    method (str): Method to be used ('F', 'B', or 'C').
    include_init (bool): Whether to include the initial value in the result.
    
    Returns:
    Function that performs the specified method.
    """

    # Function to roll out the calculations for methods 'F' and 'B'
    def rollout_fn_FB(U_matrix, init_value):
        """
        Performs calculations for methods 'F' and 'B'.
        
        Args:
        U_matrix (array): Transformation matrix.
        init_value (array): Initial value.
        
        Returns:
        array: Resulting values after 'n' steps.
        """

        def scan_body(init_value, _):
            # Calculate the next step using the transformation matrix
            next_step = jnp.einsum("ij ,  j ", U_matrix, init_value)
            return next_step, next_step

        # Use lax.scan to iteratively apply the transformation
        _, resaut = lax.scan(jit(scan_body, backend="cpu"), init_value, None, length=n)
        
        # Include the initial value in the result if specified
        if include_init:
            return jnp.concatenate([jnp.expand_dims(init_value, axis=0), resaut], axis=0)
        return resaut

    # Function to roll out the calculations for method 'C'
    def rollout_fn_C(U_matrix, init_value):
        """
        Performs calculations for method 'C'.
        
        Args:
        U_matrix (array): Transformation matrices.
        init_value (array): Initial values.
        
        Returns:
        array: Resulting values after 'n' steps.
        """

        def scan_body(init_value, _):
            # Calculate the next step using two transformation matrices
            next_step = jnp.einsum("ij ,j ->i", U_matrix[1], init_value[0]) + jnp.einsum("ij,j ->i", U_matrix[0], init_value[1])
            
            # Roll the initial values to prepare for the next iteration
            init_value = jnp.roll(init_value, shift=-1, axis=0).at[-1].set(next_step)
            return init_value, next_step

        # Use lax.scan to iteratively apply the transformation
        _, resaut = lax.scan(jit(scan_body, backend="cpu"), init_value, None, length=n)
        
        # Include the initial value in the result if specified
        if include_init:
            return jnp.concatenate([jnp.expand_dims(init_value[0], axis=0), resaut], axis=0)
        return resaut

    # Map the method string to the corresponding function
    method_map = {"F": rollout_fn_FB, "B": rollout_fn_FB, "C": rollout_fn_C}
    rollout_fn = method_map.get(method.upper())
    return rollout_fn

# Function to perform an analytical solution
def fast_analytical_solver(X, Lambda, c):
    """
    Solves the system analytically using matrix exponentiation.
    
    Args:
    X (array): Matrix for transformation.
    Lambda (array): Diagonal matrix of eigenvalues.
    c (array): Initial condition vector.
    
    Returns:
    Function that calculates the analytical solution for given times.
    """
    
    # Vectorize the function for efficiency
    vectorize = lambda func: vmap(func)
    
    def process(t):
        """
        Processes the time steps.
        
        Args:
        t (array): Array of time steps.
        
        Returns:
        array: Analytical solution at each time step.
        """
        
        def calc(t):
            """
            Calculates the analytical solution at a given time 't'.
            
            Args:
            t (float): Time step.
            
            Returns:
            array: Analytical solution at time 't'.
            """
            Lambda_t_exp = jnp.diag(jnp.exp(Lambda * t))
            return X @ Lambda_t_exp @ c
        
        return vectorize(calc)(t)
    
    return process
