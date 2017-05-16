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
        self.matricies = np.asmatrix(data[key].values)
        self.transform = np.eye(4)

    def setTransform(self, index=0):
        """
        Sets the current object-to-world transform.
        """
        self.transform = np.asmatrix(self.matricies[0, index]).reshape(4, 4)

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


class Node(Transformable):
    """
    Base Class for objets in GazeInWord Demo.
    """

    def __init__(self, data, key):
        Transformable.__init__(self, data, key)


# TODO: Get gaze point from 4x4
# - negates getting xyz from file
# - negates all the setting of xyz to mat
class Head(Node):
    """
    Class encapsulating all head and eye information.
    """

    class Eye(Node):
        """
        Class encapsulating eye-tracking information.
        Might be best to think of the eye as a vector, and not an independent node.

        rightEyeBasePoint_XYZ, leftEyeBasePoint_XYZ
        """
        # TODO: Get gaze point from 4x4 - is this possible??
        # - negates getting xyz from file
        # - negates all the setting of xyz to mat

        def __init__(self, data, key, basepoints, gazepoints):
            Node.__init__(self, data, key)
            """
            Here key == "leftEyeInHead_XYZ", "rightEyeInHead_XYZ", or "cycEyeInHead_XYZ"
            """
            self.basepoints = basepoints
            self.gazepoints = gazepoints

        def gazeinhead(self, time):
            """
            Returns gaze point in head coordinates
            """
            gazeinhead = self.gazepoints[time]
            return self.basepoints[time], gazeinhead

        def gazeinworld(self, refframe, time):
            """
            Returns gaze point in world frame
            """
            gazeinhead = np.hstack(
                (np.matrix(self.gazepoints[time]), np.matrix([1])))
            gazeinworld = gazeinhead * refframe.transform
            gazeinworld = gazeinworld[0, :3].tolist()

            return self.basepoints[time], gazeinworld[0]


    def __init__(self, data, key="viewMat_4x4", IOD=0):
        Node.__init__(self, data, key)
        self.setTransform()

        self.IOD = IOD  # static attribute (for now)

        # create cyc eye
        self.eyeC = Head.Eye(data, "cycMat_4x4", data["cycEyeBasePoint_XYZ"], data["cycGazeNodeInWorld_XYZ"])
        self.eyeL = Head.Eye(data, "leftEyeMat_4x4", data["leftEyeBasePoint_XYZ"], data["leftEyeBasePoint_XYZ"])
        self.eyeR = Head.Eye(data, "rightEyeMat_4x4", data["rightEyeBasePoint_XYZ"], data["rightEyeBasePoint_XYZ"])