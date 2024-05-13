from linear_system_solver import DynamicSystem , Model
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.random as random 

system_matrix = jnp.array([[0, 1], [-1.4, -0.3]])
initial_state = jnp.array([1, 0])
system = DynamicSystem(system_matrix, initial_state)

# Compute the states over time
time_values = jnp.linspace(0, 10, 200)
states = system.states_over_time(time_values)
u1, u2 = states[:, 0], states[:, 1]


K = random.PRNGKey(0)
noise = random.normal(key=K, shape=(20,))
theta=jnp.array([0.5,-2,-3,4,6])

data = Model(True, shape=(-1, 1, 20), order=len (theta), key=random.PRNGKey(0), noise=noise, theta = theta)
print (data.fit(order=4))