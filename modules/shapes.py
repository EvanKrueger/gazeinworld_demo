from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import plotly.graph_objs as go
import numpy as np

class Sphere:
    def __init__(self):
        #just a sphere
        theta = np.linspace(0,2*np.pi,100)
        phi = np.linspace(0,np.pi,100)
        self.x = np.outer(np.cos(theta),np.sin(phi))
        self.y = np.outer(np.sin(theta),np.sin(phi))
        self.z = np.outer(np.ones(100),np.cos(phi))  # note this is 2d now
        self.view = go.Surface(x=self.x, y=self.y, z=self.z)

    def transform(self, refframe):
        mat = np.matrix(np.dstack((self.x.ravel(), self.y.ravel(), self.z.ravel(), np.ones(len(self.z.ravel())))))
        mat = refframe.transform * mat.T
        mat = mat.T
        self.x = mat[:, 0].reshape(100, 100)
        self.y = mat[:, 1].reshape(100, 100)
        self.z = mat[:, 2].reshape(100, 100)
        self.view = go.Surface(x=self.x, y=self.z, z=self.y)# flip y, z for plotly

    def scale(self, scaleVec):
        scalarX, scalarY, scalarZ = scaleVec
        self.x *= scalarX
        self.y *= scalarY
        self.z *= scalarZ
        self.view = go.Surface(x=self.x, y=self.y, z=self.z)


class Vector:
    def __init__(self, origin, point):
        self.origin = origin
        self.point = point
        # flip y, z for plotly
        self.view = go.Scatter3d(
            x=[origin[0], point[0]],
            # y=[origin[1], point[1]],
            y=[origin[2], point[2]],
            # z=[origin[2], point[2]],
            z=[origin[1], point[1]],
            marker=dict(size=4, colorscale='Viridis'),
            line=dict(color='#1f77b4', width=1))

