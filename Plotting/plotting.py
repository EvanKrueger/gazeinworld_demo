from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import plotly.graph_objs as go


data = go.Data([
    go.Mesh3d(
        x=[0, 1, 2, 0],
        y=[0, 0, 1, 2],
        z=[0, 2, 0, 1],
        colorbar=go.ColorBar(
            title='z'
        ),
        colorscale=[['0', 'rgb(255, 0, 0)'], ['0.5', 'rgb(0, 255, 0)'], [
            '1', 'rgb(0, 0, 255)']],
        intensity=[0, 0.33, 0.66, 1],
        i=[0, 0, 0, 1],
        j=[1, 2, 3, 2],
        k=[2, 3, 1, 3],
        name='y',
        showscale=True
    )
])


layout = dict(
    autosize=True,
    scene=dict(
        xaxis=dict(
            autorange=False,
            range=[-20, 20]
        ),
        yaxis=dict(
            autorange=False,
            range=[-20, 20]
        ),
        zaxis=dict(
            autorange=False,
            range=[-20, 20]
        )
    )
)
fig = go.Figure(data=data, layout=layout)
plot(fig, filename='3d-mesh-tetrahedron-python.html')
