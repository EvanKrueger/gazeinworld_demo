"""
Module created for singluar purpose: converting existing dictionary-literal files to pickled
objects. This should be depreciated whenever I can replace the current method of writing data.
"""
from __future__ import print_function
from __future__ import division

# make Modules dir accessible
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "/Modules/")

import pandas as pd


def dictFileToDataFrame(filename):
    '''a simple reader for the Perform dict-log-dump'''
    leek = []
    # remap nan's (which are np.nans which are different from NaNa)
    nan = float('NaN')  # NOQA

    with open(filename) as f:
        for line in f:
            leek.append(eval(line))

    df = pd.DataFrame(leek)
    # we'll want to do some fun stuff with this
    return df


# # convert
# import PerformParser
# PerformParser.readPerformDictFlat(
#     "/home/evan/Dev/Lab/gazeinworld_demo/data/2017-4-14-18-36/exp_data-2017-4-14-18-36.dict")

data = dictFileToDataFrame(
    "/home/evan/Dev/Lab/gazeinworld_demo/data/2017-4-14-18-36/exp_data-2017-4-14-18-36.dict")
print(data)
