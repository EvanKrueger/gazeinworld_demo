from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import numpy as np

class Tetrahedron:
    def __init__(self):
        pass

#     go.Mesh3d(
#         x=[0, 1, 2, 0],
#         y=[0, 0, 1, 2],
#         z=[0, 2, 0, 1],
#         colorbar=go.ColorBar(title='z'),
#         colorscale=[
#             ['0', 'rgb(255, 0, 0)'],
#             ['0.5', 'rgb(0, 255, 0)'],
#             ['1', 'rgb(0, 0, 255)']],
#         intensity=[0, 0.33, 0.66, 1],
#         i=[0, 0, 0, 1],
#         j=[1, 2, 3, 2],
#         k=[2, 3, 1, 3],
#         name='y',
#         showscale=True)])

def show(scene, data, notebook=False):  #, *pargs):
    # TODO: Turn off legend
    """Create plot based on the World scene dimensions, including all shapes passed in."""

    layout = {
        "autosize" : True,
        "scene" : scene
    }

    fig = go.Figure(data=data, layout=layout)

    import os
    filename = os.getcwd() + os.sep + "figures" + os.sep + "giwPlot.html"
    if notebook:
        iplot(fig, filename=filename)
    else:
        plot(fig, filename=filename)

