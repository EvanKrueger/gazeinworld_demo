"""
Module containing functions useful for processing eye-tracking data
"""
import numpy as np
import pandas

def convert4x4(series):
    """
    Apply function to convert 16 element list to 4x4 ndarray.
    """
    arr = np.asarray(series).reshape(4, 4)
    return arr.T


def angularDistance(vec1, vec2):
    """
    Returns the angle between two vectors in degrees

    note:
        requires that ||vec|| > 0
        cannot divide my zero
    """
    # make vectors unit length
    vecNorm1 = vec1 / np.linalg.norm(vec1)
    vecNorm2 = vec2 / np.linalg.norm(vec2)

    # calculate angle between vectors in radians
    prod = np.inner(vecNorm1, vecNorm2)
    angle = np.arccos(prod)

    return np.degrees(angle)


def sphericalCoordinates(vector3D):
    """
    Converts positional vector into spherical coordinates.
    """
    x, y, z = vector3D

    # TODO: check these correspond to Vizard coordinate system
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(y / radius)
    rho = np.arctan2(z, x)

    # TODO: check these are in correct order
    theta = np.degrees(theta)
    rho = np.degrees(rho)
    coords = np.array([radius, theta, rho])

    return coords


def angularError(gazePoint3D, objectPos3D):
    """
    Calculates the angular error between a gaze vector and a known object position.
    """
    # TODO: transform object position in world frame to a head-centered frame (or) transform gaze-in-head into world coordinates before proceeding.

    # convert the two vectors to spherical coordinates
    gazePoint = sphericalCoordinates(gazePoint3D)
    objectPos = sphericalCoordinates(objectPos3D)

    # subtract elementwise
    error = gazePoint - objectPos

    # return last two elements, as difference of radius is irrelevant
    return error[1:]



if __name__ == "__main__":

    vec1 = np.array([10, 0, 0])
    vec2 = np.array([1, 1, 1])

    print("Inner Product:\t", angularDistance(vec1, vec2))
    print("Axis Angle:\t", np.cross(vec1, vec2))
    print("Angular Error:\t", angularError(vec1, vec2))
