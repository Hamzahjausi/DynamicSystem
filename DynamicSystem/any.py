import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import jax.numpy as jnp
from jax import lax, jit, config

config.update("jax_enable_x64", True)

# إعداد Dash
app = dash.Dash(__name__)

# تخطيط واجهة المستخدم
app.layout = html.Div([
    dcc.Graph(id='live-update-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # تحديث كل ثانية
        n_intervals=0
    )
])

def forward_setup(system_matrix, dt, t_max):
    n_steps = int(t_max / dt) + 1
    U_matrix = jnp.eye(system_matrix.shape[0]) + dt * system_matrix
    return n_steps, U_matrix

def calc_forward(Un, U_matrix):
    Un_1 = U_matrix @ Un
    return Un_1, Un_1

def fast_calc(U_matrix, Un, n_steps):
    func = jit(lambda Un, _: calc_forward(Un, U_matrix))
    steps = jnp.arange(n_steps)
    _, result = lax.scan(func, Un, steps)
    return result

# إعداد البيانات
system_matrix = jnp.array([
    [0, 1],
    [-1, -0.1]
], dtype=jnp.float64)

initial_state = jnp.array([1, 1], dtype=jnp.float64)
dt = 0.01
t_max = 2 * jnp.pi
n_steps, U_matrix = forward_setup(system_matrix, dt, t_max)

# دالة لتحديث الرسم البياني بشكل حي
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    solution_forward = fast_calc(U_matrix, initial_state, n_steps)
    
    heatmap = go.Heatmap(
        z=solution_forward.T,
        colorscale='RdBu',
        zmin=-2,
        zmax=2
    )

    layout = go.Layout(
        title='Live Update of State Trajectory (u0 vs. u1)',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Space')
    )

    return {'data': [heatmap], 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
