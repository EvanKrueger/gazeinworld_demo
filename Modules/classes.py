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
        self.time = 0

    def settime(self, frame):
        """
        Time is specified by a frame number.
        """
        self.time = frame


class Head(Node):
    """
    Class encapsulating all head information
    """

    def __init__(self, data, key):
        Node.__init__(self, data, key)
        self.setTransform()


class Eye(Head):
    """
    Class encapsulating eye-tracking information.
    """
    # TODO: Get gaze point from 4x4
    # - negates getting xyz from file
    # - negates all the setting of xyz to mat

    def __init__(self, data, key, parent, key2):
        """
        Here key == "leftEyeInHead_XYZ", "rightEyeInHead_XYZ", or "cycEyeInHead_XYZ"
        """
        Head.__init__(self, data, key)
        self.head = parent
        self.gazepoints = data[key2].values

    def gazeinhead(self):
        """
        Returns gaze point in head coordinates
        """
        gazeinhead = np.matrix(self.gazepoints[self.time])
        return gazeinhead

    def gazeinworld(self):
        """
        Returns gaze point in world frame
        """
        gazeinhead = np.hstack(
            (np.matrix(self.gazepoints[self.time]), np.matrix([1])))
        gazeinworld = gazeinhead * self.head.transform

        return gazeinworld[0, :3]


# class Geometry(Node):
#     """
#     Base class for shapes in GazeInWorldDemo
#     """
#     pass


# class PrimitiveGeometry(Geometry):

#     """
#     """
#     pass


if __name__ == "__main__":
    pass
