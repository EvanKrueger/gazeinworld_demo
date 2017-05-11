"""
This module is a stand-in for the ipython notebook that will be used in the demonstration.
"""
from __future__ import division, print_function
from Modules.PerformParser import dictFileToDataFrame
from Modules.timeManager import TimeMgr
from Modules.classes import *

# define nan var for "NaN"-encoded-floats in dataframe
nan = float("NaN")
# read dict file, save as pandas DF
datafile = dictFileToDataFrame(
    "Data/2017-4-14-18-36/exp_data-2017-4-14-18-36.dict")

# create time manager instance for controlling global time
time = TimeMgr(datafile)
