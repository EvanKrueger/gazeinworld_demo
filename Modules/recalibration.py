from __future__ import division
import PerformParser as pp
import pandas as pd
import numpy as np
import bokeh.plotting as bkP
import bokeh.models as bkM
from scipy import signal as sig
import catchE1Funs as expFun
import cv2
import Quaternion as qu
import matplotlib.pyplot as plt

from configobj import ConfigObj
from configobj import flatten_errors
from validate import Validator

import performFun as pF

def calcFeaturesOnScreen(sessionDict):
    '''
    Calculate 
    '''

    ##########################################################
    ##########################################################
    ## cycMetric eye on Screen for calibration
    
    cycMetricEyeOnScreen = pF.eyeOnScreenToMetricEyeOnScreen(sessionDict,sessionDict['calibration']['cycEyeOnScreen'],
                            'cycMetricEyeOnScreen')

    sessionDict['calibration'][('cycMetricEyeOnScreen','X')] = cycMetricEyeOnScreen[('cycMetricEyeOnScreen','X')] 
    sessionDict['calibration'][('cycMetricEyeOnScreen','Y')] = cycMetricEyeOnScreen[('cycMetricEyeOnScreen','Y')] 
    sessionDict['calibration'][('cycMetricEyeOnScreen','Z')] = cycMetricEyeOnScreen[('cycMetricEyeOnScreen','Z')] 
    
    ##########################################################
    ##########################################################
    ## cycMetric eye on Screen for processed
    
    cycMetricEyeOnScreen = pF.eyeOnScreenToMetricEyeOnScreen(sessionDict,sessionDict['processed']['cycFiltEyeOnScreen'],
                            'cycFiltMetricEyeOnScreen')

    sessionDict['processed'][('cycFiltMetricEyeOnScreen','X')] = cycMetricEyeOnScreen[('cycFiltMetricEyeOnScreen','X')] 
    sessionDict['processed'][('cycFiltMetricEyeOnScreen','Y')] = cycMetricEyeOnScreen[('cycFiltMetricEyeOnScreen','Y')] 
    sessionDict['processed'][('cycFiltMetricEyeOnScreen','Z')] = cycMetricEyeOnScreen[('cycFiltMetricEyeOnScreen','Z')] 
    
    ##########################################################
    ##########################################################
    ## Convert cycMetricEyeOnScreen to pixel values
    
    cycEyeOnScreenDf = pF.metricEyeOnScreenToPixels(sessionDict,
                                                    sessionDict['calibration']['cycMetricEyeOnScreen'],
                                                    'cycEyeOnScreen')
    
    sessionDict['calibration'][('cycEyeOnScreen','X')] = cycEyeOnScreenDf[('cycEyeOnScreen','X')] 
    sessionDict['calibration'][('cycEyeOnScreen','Y')] = cycEyeOnScreenDf[('cycEyeOnScreen','Y')] 
    
    ##########################################################
    ##########################################################
    ## calibration point metric eye on Screen
    
    sessionDict = pF.calcCalibPointMetricEyeOnScreen(sessionDict)

    ##########################################################
    ##########################################################
    ## Convert calibPointMetricEyeOnScreen to pixel values

    calibPointEyeOnScreenDf = pF.metricEyeOnScreenToPixels(sessionDict,
                                                        sessionDict['calibration']['calibPointMetricEyeOnScreen'],
                                                        'calibPointEyeOnScreen')
    
    sessionDict['calibration'][('calibPointEyeOnScreen','X')] = calibPointEyeOnScreenDf[('calibPointEyeOnScreen','X')] 
    sessionDict['calibration'][('calibPointEyeOnScreen','Y')] = calibPointEyeOnScreenDf[('calibPointEyeOnScreen','Y')] 
    
    ##########################################################
    ##########################################################
    ## Eye to calibraiton point
    
    eyeToCalibrationPointDirDf = pF.metricEyeOnScreenToEyeInHead(sessionDict,
                                                                 sessionDict['calibration']['calibPointMetricEyeOnScreen'],
                                                                 'eyeToCalibrationPoint')

        
    sessionDict['calibration'][('eyeToCalibrationPoint','X')] = eyeToCalibrationPointDirDf[('eyeToCalibrationPoint','X')]
    sessionDict['calibration'][('eyeToCalibrationPoint','Y')] = eyeToCalibrationPointDirDf[('eyeToCalibrationPoint','Y')]
    sessionDict['calibration'][('eyeToCalibrationPoint','Z')] = eyeToCalibrationPointDirDf[('eyeToCalibrationPoint','Z')] 
    
    
    ##########################################################
    ##########################################################
    ## Eye to gaze point (really, this is just cyc EIH)

    cycEIHDirDf = pF.metricEyeOnScreenToEyeInHead(sessionDict,sessionDict['calibration']['cycMetricEyeOnScreen'],'cycEIH')
    
    sessionDict['calibration'][('cycEIH','X')] = cycEIHDirDf[('cycEIH','X')]
    sessionDict['calibration'][('cycEIH','Y')] = cycEIHDirDf[('cycEIH','Y')]
    sessionDict['calibration'][('cycEIH','Z')] = cycEIHDirDf[('cycEIH','Z')] 
            

    ##########################################################
    ##########################################################
    ## cycAngle_elAz
    
    cycEIHDir_fr_xyz = sessionDict['calibration']['cycEIH'].values
    cycVertAngle_fr = np.rad2deg( np.arctan2( cycEIHDir_fr_xyz[:,1], cycEIHDir_fr_xyz[:,2]))
    cycHorizAngle_fr = np.rad2deg( np.arctan2( cycEIHDir_fr_xyz[:,0], cycEIHDir_fr_xyz[:,2]))
    #cycAngle_elAz = np.array([cycHorizAngle_fr,cycVertAngle_fr],dtype=np.float)
    sessionDict['calibration'][('cycAzElInHead','az')] = cycHorizAngle_fr
    sessionDict['calibration'][('cycAzElInHead','el')] = cycVertAngle_fr

    eyeToCalibDir_fr_xyz = sessionDict['calibration']['eyeToCalibrationPoint'].values
    calibVertAngle_fr = np.rad2deg( np.arctan2( eyeToCalibDir_fr_xyz[:,1], eyeToCalibDir_fr_xyz[:,2]))
    calibHorizAngle_fr = np.rad2deg( np.arctan2( eyeToCalibDir_fr_xyz[:,0], eyeToCalibDir_fr_xyz[:,2]))
    sessionDict['calibration'][('calibAzElInHead','az')] = calibHorizAngle_fr
    sessionDict['calibration'][('calibAzElInHead','el')] = calibVertAngle_fr
    
    return sessionDict


def calcLinearHomography(sessionDict,method = cv2.LMEDS, threshold = 10):

    print 'calculateLinearHomography():  Currently only reprojects  cyc gaze data'

    #sessionDict['calibration']['cycEyeOnScreen']
    calibDataFrame = sessionDict['calibration']
    gbCalibSession = calibDataFrame.groupby(['trialNumber'])

    numberOfCalibrationSession =  (max(calibDataFrame.trialNumber.values)%1000)//100

    trialOffset = 1000;
    frameOffset = 0

    numberOfCalibrationPoints = 27
    numberOfCalibrationSession = 0
    
    # Calculate a homography for each session
    for i in range(numberOfCalibrationSession + 1):

        startTrialNumber = trialOffset + 100*i 
        endTrialNumber = trialOffset + 100*i + numberOfCalibrationPoints - 1

        firstCalibrationSession = gbCalibSession.get_group(startTrialNumber)
        lastCalibrationSession = gbCalibSession.get_group(endTrialNumber)

        numberOfCalibrationFrames = max(lastCalibrationSession.index) - min(firstCalibrationSession.index) + 1
        dataRange = range(frameOffset, frameOffset + numberOfCalibrationFrames)

        ### Get  cycMetricEyeOnScreen data for calib session. Drop the Z values.
        cycMetricEyeOnScreen = sessionDict['calibration']['cycMetricEyeOnScreen'].values[dataRange,:2]
        cycMetricEyeOnScreen = np.array(cycMetricEyeOnScreen,dtype=np.float)

        ### Get cycMetricEyeOnScreen data for calib session. Drop the Z values.
        calibPointMetricEyeOnScreen = sessionDict['calibration']['calibPointMetricEyeOnScreen'].values[dataRange,:2]
        calibPointMetricEyeOnScreen = np.array(calibPointMetricEyeOnScreen,dtype=np.float)
        
        #homography = cv2.findHomography(cycMetricEyeOnScreen, calibPointMetricEyeOnScreen)#, method , ransacReprojThreshold = threshold)
        #homogrophy = result[0]

        ### Get  cycMetricEyeOnScreen data for calib session. Drop the Z values.
        cycMetricEyeOnScreen = sessionDict['calibration']['cycMetricEyeOnScreen'].values[dataRange,:2]
        cycMetricEyeOnScreen = np.array(cycMetricEyeOnScreen,dtype=np.float)

        ### Get cycMetricEyeOnScreen data for calib session. Drop the Z values.
        calibPointMetricEyeOnScreen = sessionDict['calibration']['calibPointMetricEyeOnScreen'].values[dataRange,:2]
        calibPointMetricEyeOnScreen = np.array(calibPointMetricEyeOnScreen,dtype=np.float)

        ##########################################################
        ## Calculate homography
        
        result = cv2.findHomography(cycMetricEyeOnScreen, calibPointMetricEyeOnScreen) # Defaulting to ransac
        #, method , ransacReprojThreshold = threshold)
        homography = result[0]
        
        ##########################################################
        ##########################################################
        ## Save homography in sessionDict

        #print 'homogrophy = ', homogrophy
        tempMatrix = np.zeros((len(sessionDict['processed']),3,3))
        tempMatrix[:,0:3,0:3] = homography
        tempMatrix = tempMatrix.reshape((len(sessionDict['processed']),9))
        
        for i in range(9):
            sessionDict['processed'][('linearHomography',str(i))] = tempMatrix[:,i]
        
        tempMatrix = np.zeros((len(sessionDict['calibration']),3,3))
        tempMatrix[:,0:3,0:3] = homography
        tempMatrix = tempMatrix.reshape((len(sessionDict['calibration']),9))
        
        for i in range(9):
            sessionDict['calibration'][('linearHomography',str(i))] = tempMatrix[:,i]
            
    return sessionDict

def applyHomographyToCyc(sessionDict):

    ###############################################################
    ###############################################################
    ### FIrst to processed
    
    totalFrameNumber = len(sessionDict['processed'])
    
    def applyHomographyToCyc(row):
       
        #cycMetric_XYZ = np.array(np.hstack((row['cycMetricEyeOnScreen'],1)),dtype=np.float)
        
        XYZ = np.array(row['cycFiltMetricEyeOnScreen'].values,dtype=np.float)
        XYZ[2] = 1
        
        homography = np.array(row['linearHomography'].values,dtype=np.float)
        homography = homography.reshape(3,3)
        reproj_XYZ = np.dot(homography, XYZ)
        
        return pd.Series({'X': reproj_XYZ[0], 'Y':reproj_XYZ[1],'Z':reproj_XYZ[0]})

    reprojDf = sessionDict['processed'].apply(applyHomographyToCyc,axis=1)

    print '*****applyHomographyToCyc():  Filtered data now transformed by homogrphy*****'

    sessionDict['processed'][('cycFiltMetricEyeOnScreen','X')] = reprojDf['X']
    sessionDict['processed'][('cycFiltMetricEyeOnScreen','Y')] = reprojDf['Y']
    sessionDict['processed'][('cycFiltMetricEyeOnScreen','Z')] = reprojDf['Z']

    ###############################################################
    ###############################################################
    ### Now to calibrationg
    def applyHomographyToCyc(row):
       
        #cycMetric_XYZ = np.array(np.hstack((row['cycMetricEyeOnScreen'],1)),dtype=np.float)
        
        XYZ = np.array(row['cycMetricEyeOnScreen'].values,dtype=np.float)
        XYZ[2] = 1
        
        homography = np.array(row['linearHomography'].values,dtype=np.float)
        homography = homography.reshape(3,3)
        reproj_XYZ = np.dot(homography, XYZ)
        
        return pd.Series({'X': reproj_XYZ[0], 'Y':reproj_XYZ[1],'Z':reproj_XYZ[0]})
    
    reprojDf = sessionDict['calibration'].apply(applyHomographyToCyc,axis=1)

    sessionDict['calibration'][('reprojectedCycMetricEyeOnScreen','X')] = reprojDf['X']
    sessionDict['calibration'][('reprojectedCycMetricEyeOnScreen','Y')] = reprojDf['Y']
    sessionDict['calibration'][('reprojectedCycMetricEyeOnScreen','Z')] = reprojDf['Z']
    
    
#     # Normalize
#     reprojDf = reprojDf.apply(lambda XYZ: np.divide(XYZ,np.linalg.norm(XYZ)),axis=1)
#     reprojCycEIHDir_fr_XYZ = reprojDf.values

#     reprojCycEIHVertAngle_fr = np.rad2deg( np.arctan2( reprojCycEIHDir_fr_XYZ[:,1], reprojCycEIHDir_fr_XYZ[:,2]))
#     reprojCycEIHHorizAngle_fr = np.rad2deg( np.arctan2( reprojCycEIHDir_fr_XYZ[:,0], reprojCycEIHDir_fr_XYZ[:,2]))
#     sessionDict['processed'][('reprojCycAzElInHead','az')] = reprojCycEIHHorizAngle_fr
#     sessionDict['processed'][('reprojCycAzElInHead','el')] = reprojCycEIHVertAngle_fr
    
    return sessionDict


