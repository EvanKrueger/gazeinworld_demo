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
world.createScene([7, 7, 7])


if __name__ == "__main__":
    head = Head(datafile, "viewMat_4x4")

    # head shape
    shape = Sphere()
    shape.transform(head)

    # gaze in head vector
    origin, point = head.eyeC.gazeinhead(0)
    origin = [np.average(shape.x), np.average(shape.y), np.average(shape.z)]
    gazeIH = Vector(origin, point)
    # gaze in head vector
    origin, point = head.eyeC.gazeinworld(head, 0)
    origin = [np.average(shape.x), np.average(shape.y), np.average(shape.z)]
    gazeIW = Vector(origin, point)


    # display
    show(world.scene, [shape.view, gazeIW.view, gazeIH.view])

    # for col in datafile.columns:
    #     print(col)
