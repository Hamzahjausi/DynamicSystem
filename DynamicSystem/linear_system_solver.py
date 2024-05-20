import jax.numpy as jnp
import jax.random as random 
from jax import config, lax ,vmap , jit
config.update ("jax_enable_x64", True)
import jax
from functools import partial

class DynamicSystem:

    def __init__(self, system_matrix, initial_state):
        self.system_matrix = system_matrix.astype(jnp.float64)
        self.initial_state = initial_state.astype(jnp.float64)
        self._solve_for_initial_conditions()

    def _solve_for_initial_conditions(self):
        self.Lambda, self.X = jnp.linalg.eig(self.system_matrix)
        self.X_inv = jnp.linalg.inv(self.X)
        self.c = self.X_inv @ self.initial_state
    @partial (jit , static_argnums = 0)
    def _state_at_time(self, t):
        exp_diag_Lambda = jnp.diag(jnp.exp(self.Lambda * t))
        state_vector = self.X @ (exp_diag_Lambda @ self.c)  
        return state_vector
    @partial (jit , static_argnums = 0)
    def states_over_time(self, time_values):
        return vmap(self._state_at_time, in_axes=0, out_axes=1)(time_values)
    





    def forward_setup (self, dt,t_max):
        self.n_steps = int(t_max / dt) + 1
        self.U_matrix= jnp.eye (self.system_matrix.shape[0]) + dt * self.system_matrix

    @partial(jit, static_argnums=0)
    def _forwar_calc (self , Un , _):
        Un_1 = self.U_matrix@ Un
        return Un_1,Un_1
    
    @partial(jit, static_argnums=0)
    def forward_calc (self):
        un = self.initial_state 
        steps = jnp.arange (self.n_steps) 
        _,result= lax.scan (self._forwar_calc , un , steps )
        return result
    



















class Model:
    def __init__(self, generate, *args, **kwargs):
        if generate:
            self.initialize(*args, **kwargs)

    def initialize(self, shape, order, key, noise=None, theta=None):
        self.theta = theta if theta is not None else random.uniform(key=key, shape=(order,))
        self.x = jnp.linspace(*shape)
        X = jnp.column_stack([self.x**i for i in range(order)])
        self.y = X @ self.theta + (noise if noise is not None else 0)
        self.shape = shape

    def system(self, order, N):
        x = jnp.linspace(-1, 1, N)
        return x, jnp.column_stack([x**i for i in range(order)])

    def fit(self, order):
        x, X = self.system(order, N=self.shape[-1])
        U, S, VT = jnp.linalg.svd(X, full_matrices=False)
        X_plus = VT.T @ jnp.diag(1 / S) @ U.T
        self.theta_hat = X_plus @ self.y
        x, X = self.system(order, N=200)
        return x, X @ self.theta_hat
    def V (self , i):
        return self.fit (i)
    

