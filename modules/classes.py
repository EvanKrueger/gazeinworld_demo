"""
Container classes for use in demo
"""
from __future__ import division, print_function
import numpy as np

# TODO: make sure data is matrix as expected - not Pandas' stupid matrix
# of lists
# TODO: Explore using osg model file in plotly
# TODO: Look into getting verticies out of ogjb model file:
# https://plot.ly/python/3d-mesh/#mesh-tetrahedron
# https://plot.ly/python/surface-triangulation/


class World:
    """
    Class for gobal world attributes.
    Initializes the current time and addressable duration of time.
    """
    def __init__(self, data):
        self.data = data  # keep data in world for now
        self.scene = {}
        self.now = 0  # still in frames
        self.history = len(data)

    def setTime(self, time):
        """
        Updates the current time
        """
        self.now = time

    def createScene(self, dims):
        """
        Create plotting environment for plotly.
        dims : [x, y, z]
        """
        self.scene["xaxis"] = {"autorange" : False, "range" : [-dims[0], dims[0]]}
        self.scene["yaxis"] = {"autorange" : False, "range" : [-dims[1], dims[1]]}
        self.scene["zaxis"] = {"autorange" : False, "range" : [-dims[2], dims[2]]}


class Transformable:
    """
    Encapsulates object-to-world transforms
    """

    def __init__(self, data, key):
        # reshape Pandas series of lists into 4x4xn numpy array
        transformData = np.concatenate([np.array(row) for row in data])
        dims = (int(len(transformData) / 16), 4, 4)
        self.transforms = self.transforms.reshape(dims)
        self.transform = np.eye(4)

    def setTransform(self, index=0):
        """
        Sets the current object-to-world transform.
        """
        self.transform = np.asmatrix(self.transforms[0, index]).T  # column notation

    def getTransform(self):
        """
        Returns object-to-world 4x4 transform.
        """
        return self.transform

    def getInverseTransform(self):
        """
        Returns world-to-object transform.
        """
        return np.linalg.inv(self.transform)

    def getposition(self):
        """
        Return node position as 3 element matrix.
        Position is derived from 4x4 transform - assumes column dominant representation
        """
        return self.transform[:3, -1].T


    def swapAxes(self):
        """
        Swaps z and y axes for plotting in Plotly from Vizard
        """
        pass


class Node(Transformable):
    """
    Base Class for objets in GazeInWord Demo.
    """

    def __init__(self, data, key):
        Transformable.__init__(self, data, key)
        self.setTransform()



class Head(Node):
    """
    Class encapsulating all head and eye information.
    """

    class Eye(Node):
        """
        Class encapsulating eye-tracking information.
        Might be best to think of the eye as a vector, and not an independent node.
        """

        def __init__(self, data, key, gazepoints):
            Node.__init__(self, data, key)
            """
            Here key == "leftEyeInHead_XYZ", "rightEyeInHead_XYZ", or "cycEyeInHead_XYZ"
            """
            self.gazepoints = gazepoints

        def gaze(self, index, node=None):
            gazepoint = self.gazepoints[index]

            if node:
                gazepoint.append(1)
                gazepoint = node.transform * np.asmatrix(gazepoint).T
                gazepoint = gazepoint.T[0,:3].tolist()[0]

            return gazepoint




        def gazeinhead(self, time):
            """
            Returns gaze point in head coordinates
            """
            gazeinhead = self.gazepoints[time]
            return gazeinhead

        def gazeinworld(self, refframe, time):
            """
            Returns gaze point in world frame
            """
            gazeinhead = np.hstack((np.matrix(self.gazepoints[time]), np.matrix([1])))

            # eyeTransformInv = self.getInverseTransform()
            refTransform = refframe.getTransform()

            # gazeinworld = eyeTransformInv * refTransform * gazeinhead.T
            gazeinworld = refTransform * gazeinhead.T
            print(gazeinhead, gazeinworld.T)

            gazeinworld = gazeinworld.T
            gazeinworld = gazeinworld[0, :3].tolist()

            return gazeinworld[0]


    def __init__(self, data, key="viewMat_4x4", IOD=0):
        Node.__init__(self, data, key)
        self.setTransform()

        self.IOD = IOD  # static attribute (for now)

        # create cyc eye
        self.eyeC = Head.Eye(data, "cycMat_4x4", data["cycEyeInHead_XYZ"])

        self.eyeL = Head.Eye(data, "leftEyeMat_4x4", data["leftEyeInHead_XYZ"])

        self.eyeR = Head.Eye(data, "rightEyeMat_4x4", data["rightEyeInHead_XYZ"])
