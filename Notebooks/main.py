from __future__ import division
import os
cwd = os.getcwd()
print(cwd)
import sys
sys.path.append(cwd)
sys.path.append(cwd + "/Modules/")
# sys.path.append(cwd + "/../")

import pandas as pd

import numpy as np
from scipy import signal as sig

import performFun as pF
import catchE1Funs as expFun
import recalibration as rc

from plottingFuns import *
from analysisParameters import loadParameters

bkP.output_notebook()

fileTime = '2017-4-14-18-36'

analysisParameters = loadParameters(fileTime)
startFresh = True
loadProcessed = False

##############################################################################
##############################################################################
# loadSessionDict
# We can just tell members of the audience that any way they can get the
# data into a dataframe is OK
sessionDict = pF.loadSessionDict(
    analysisParameters, startFresh=startFresh, loadProcessed=loadProcessed)
