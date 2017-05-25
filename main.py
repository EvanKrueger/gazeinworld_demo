"""
This module is a stand-in for the ipython notebook that will be used in the demonstration.
"""
from __future__ import division, print_function
from modules.performparser import dictFileToDataFrame
from modules.plotting import show
from modules.shapes import Sphere, Vector
from modules.scene import World, TimeMgr
from modules.classes import *

# define nan var for "NaN"-encoded-floats in dataframe
nan = float("NaN")
# read dict file, save as pandas DF
datafile = dictFileToDataFrame("data/2017-4-14-18-36/exp_data-2017-4-14-18-36.dict")

# instantiate world
world = World(datafile)
world.createScene([25, 25, 25])


if __name__ == "__main__":
    ball = Node(datafile, "ballMat_4x4")
    ballshape = Sphere()
    ballshape.scale([0.5, 0.5, 0.5])
    ballshape.transform(ball)

    head = Head(datafile, "viewMat_4x4")
    headshape = Sphere()
    headshape.transform(head)

    # L, R eye spheres
    lefteye = Sphere()
    lefteye.transform(head.eyeL)
    # lefteye.scale([0.1, 0.1, 0.1])
    righteye = Sphere()
    righteye.transform(head.eyeR)
    # righteye.scale([0.1, 0.1, 0.1])

    # gaze in head vector
    point1 = head.eyeL.gaze(0, head)
    origin = [np.average(lefteye.x), np.average(lefteye.y), np.average(lefteye.z)]
    gazeIH = Vector(origin, point1)
    # gaze in world vector
    point2 = head.eyeL.gaze(0)
    origin = [np.average(lefteye.x), np.average(lefteye.y), np.average(lefteye.z)]
    gazeIW = Vector(origin, point2)

    # display
    show(world.scene, [ballshape.view, lefteye.view, gazeIW.view])

    # print(head.eyeR.transform == head.eyeL.transform)
    # print(datafile.columns)