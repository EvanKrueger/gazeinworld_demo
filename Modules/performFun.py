from __future__ import division
import PerformParser as pp
import pandas as pd
import numpy as np
#import bokeh.plotting as bkP
#import bokeh.models as bkM
from scipy import signal as sig
import catchE1Funs as expFun
import cv2
import Quaternion as qu
import matplotlib.pyplot as plt

from configobj import ConfigObj
from configobj import flatten_errors
from validate import Validator

from classy import *

def calcSavitskyGolayVelocity(sessionDict, deriv = 1,polyorder = 2):

    sessionDict['processed'][('cycGIWAngles','azimuth')]   = sessionDict['processed'][('cycGIWDir','X')].apply(lambda x: np.rad2deg(np.arccos(x)))
    sessionDict['processed'][('cycGIWAngles','elevation')] = sessionDict['processed'][('cycGIWDir','Y')].apply(lambda y: np.rad2deg(np.arccos(y)))

    from scipy.signal import savgol_filter

    x = sessionDict['processed'][('cycGIWAngles')].values
    fixDurFrames =  int(np.floor(0.1 / sessionDict['analysisParameters']['fps']))
    delta = sessionDict['analysisParameters']['fps']

    #print 'Filter width: %u frames' % fixDurFrames

    vel = savgol_filter(x, fixDurFrames, polyorder, deriv=1, delta=delta, axis=0, mode='interp', cval=0.0)

    sessionDict['processed'][('cycSGVel','azimuth')] = vel[:,0]
    sessionDict['processed'][('cycSGVel','elevation')] = vel[:,1]

    vel = np.sqrt(np.sum(np.power(vel,2),1))
    sessionDict['processed'][('cycSGVel','2D')] = vel

    print('Added sessionDict[\'processed\'][\'cycSGVel\']')
    return sessionDict

def calcPursuitGainDuringBlank(sessionDict):

    ################################################
    ################################################
    # Apply classifier  
    import pickle
    modelLoc = open(sessionDict['analysisParameters']['eventClassifierLoc'], 'rb')
    model = pickle.load(modelLoc)
    (gazeEventsDf,gazeEvent_fr) = applyClassifier(sessionDict,model,[8])

    sessionDict['gazeEvents'] = gazeEventsDf
    sessionDict['processed']['gazeEvent'] = gazeEvent_fr

    ################################################
    ################################################
    ## Calculate pursuit params
    
    gbProc = sessionDict['processed'].groupby(['trialNumber'])

    avgPursuitGainDuringBlank = []

    for tr in range(len(sessionDict['trialInfo'])):

        startFr = sessionDict['trialInfo'].loc[tr]['ballOffFr']
        endFr = sessionDict['trialInfo'].loc[tr]['ballOnFr']
        trFr = range(startFr,endFr)

        pursuitFr = trFr[0] + np.where(sessionDict['processed']['gazeEvent'][trFr].values==2)[0]

        numPursuitFr = len(pursuitFr)
        propPursuit = numPursuitFr / len(trFr)

        giwVel_blankFr = sessionDict['processed'][('cycSGVel','2D')].values[pursuitFr]
        ballVel_blankFr = sessionDict['processed']['cycToBallVelocity']['2D'].values[pursuitFr]

        pursuitVel_blankFr = giwVel_blankFr / ballVel_blankFr

        avgPursuitGainDuringBlank.append( np.mean(pursuitVel_blankFr) )

    sessionDict['trialInfo'][('avgPursuitGainDuringBlank','')] = avgPursuitGainDuringBlank

    return sessionDict

def calcAngularVel(sessionDict):
    '''
    Calculates angular velocity for cycGIW and the cyc-to-ball vectors, and the ratio of these velocities. 
    '''

    ##############################################
    ## Cyc GIW velocity 
    
    # If needed, get eye tracker time stamp
    if( columnExists( sessionDict['processed'], 'smiDeltaT') is False):
        sessionDict = calcSMIDeltaT(sessionDict)    
        
    #  If needed, calculate cyc gaze angle and velocity
    if( columnExists(sessionDict['processed'],'cycGIWDir') is False):
        sessionDict = calcGIWDirection(sessionDict)
        
    # Calc cyc angular velocity
    if( columnExists(sessionDict['processed'],'cycGIWVelocity') is False):
        sessionDict = calcAngularVelocityComponents(sessionDict, 'cycGIWDir', 'smiDeltaT', 'cycGIWVelocity')

    ##############################################
    ## Cyc-to-ball velocity 
    
    # If needed, get vizard time stamp
    if( columnExists( sessionDict['processed'], 'vizardDeltaT') is False):
        sessionDict = calcVizardDeltaT(sessionDict)
        
    # Calc cyc to ball direction
    if( columnExists(sessionDict['processed'],'cycToBallDir') is False):
        sessionDict = calcCycToBallVector(sessionDict)
        
    # Calc cyc to ball angular velocity
    if( columnExists(sessionDict['processed'],'cycToBallVelocity') is False):
        sessionDict = calcAngularVelocityComponents(sessionDict, 'cycToBallDir', 'vizardDeltaT', 'cycToBallVelocity')

    # ##############################################
    # ## Pursuit gain CycGIWVel / Cyc-to-ball vel

    # sessionDict['processed'][('pursuitGain','2D')] = sessionDict['processed']['cycGIWVelocity']['2D'] / sessionDict['processed']['cycToBallVelocity']['2D']
    # sessionDict['processed'][('pursuitGain','X')] = sessionDict['processed']['cycGIWVelocity']['X'] / sessionDict['processed']['cycToBallVelocity']['X']
    # sessionDict['processed'][('pursuitGain','Y')] = sessionDict['processed']['cycGIWVelocity']['Y'] / sessionDict['processed']['cycToBallVelocity']['Y']

    # print '*** calcAngVelocityAndPursuitGain(): Added sessionDict[\'processed\'][pursuitGain] ***'
    
    ##############################################
    ## Take measurements of gain and velocity at specific events
    
    if( columnExists(sessionDict['trialInfo'],'ballOnFr') is False):
        sessionDict = calcBallOffOnFr(sessionDict)

    ballOffFr = sessionDict['trialInfo']['ballOffFr']
    sessionDict['trialInfo'][('cycToBallVelocityAtBallOff','2D')] = sessionDict['processed'][('cycToBallVelocity','2D')][ballOffFr].values
    sessionDict['trialInfo'][('cycToBallVelocityAtBallOff','X')]  = sessionDict['processed'][('cycToBallVelocity','X')][ballOffFr].values
    sessionDict['trialInfo'][('cycToBallVelocityAtBallOff','Y')]  = sessionDict['processed'][('cycToBallVelocity','Y')][ballOffFr].values
    
    ballOnFr = sessionDict['trialInfo']['ballOnFr']
    sessionDict['trialInfo'][('cycToBallVelocityAtBallOn','2D')] = sessionDict['processed'][('cycToBallVelocity','2D')][ballOnFr].values
    sessionDict['trialInfo'][('cycToBallVelocityAtBallOn','X')]  = sessionDict['processed'][('cycToBallVelocity','X')][ballOnFr].values
    sessionDict['trialInfo'][('cycToBallVelocityAtBallOn','Y')]  = sessionDict['processed'][('cycToBallVelocity','Y')][ballOnFr].values

    print('*** calcAngVelocity(): Added sessionDict[\'processed\'][cycToBallVelocityAtBallOn] ***')
    print('*** calcAngVelocity(): Added sessionDict[\'processed\'][cycToBallVelocityAtBallOff] ***')
    
    return sessionDict

def calcAngularVelocityComponents(sessionDict, dataColumnName, timeColumnName, outColumnName):

    #####################################################################################
    ######################################################################################################
    ### Get unsigned angular displacement and velocity

    vecDir_fr_XYZ = sessionDict['processed'][dataColumnName].values
    
    angularDisp_fr_XYZ = [ vectorAngle(vecDir_fr_XYZ[fr-1,:],vecDir_fr_XYZ[fr,:]) for fr in range(1,len(vecDir_fr_XYZ))]
    angularDisp_fr_XYZ.append(0)
    angularVel_fr_XYZ = angularDisp_fr_XYZ / sessionDict['processed'][timeColumnName]

    sessionDict['processed'][(outColumnName,'2D')] = angularVel_fr_XYZ
    print('*** calcBallAngularVelocity(): Added sessionDict[\'processed\'][(\'ballAngularVelocity\',\'2D\')] ***')

    ######################################################################################################
    ######################################################################################################
    ### Get  angular displacement and velocity along world X/Y
    
    if(columnExists(sessionDict['processed'],'worldUpDir') is False):
        sessionDict = calcWorldUpDir(sessionDict)
        
    yDir_fr_xyz = sessionDict['processed']['worldUpDir'].values
    
    azimuthulVec = np.cross(vecDir_fr_XYZ,yDir_fr_xyz)
    azimuthulDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in azimuthulVec],dtype=np.float)
    
    # orthogonal to gaze vector and azimuthulDir
    elevationVec = np.cross(azimuthulDir,vecDir_fr_XYZ);
    elevationDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in elevationVec],dtype=np.float)
    
    # Velocity
    vecDirOffset_fr_XYZ = np.roll(vecDir_fr_XYZ,1,axis=0)
    
    def vecDot(a,b):
            res1 = np.einsum('ij, ij->i', a, b)
            res2 = np.sum(a*b, axis=1)
            return res2
    
    # Get vector lengths when projected onto the new basis
    xDist_fr_xyz = vecDot(vecDirOffset_fr_XYZ,azimuthulDir)
    yDist_fr_xyz = vecDot(vecDirOffset_fr_XYZ,elevationDir)
    zDist_fr_xyz = vecDot(vecDirOffset_fr_XYZ,vecDir_fr_XYZ)
    
    horzError_fr = np.rad2deg( np.arctan2(xDist_fr_xyz,zDist_fr_xyz))
    vertError_fr = np.rad2deg( np.arctan2(yDist_fr_xyz,zDist_fr_xyz))

    velX_fr = horzError_fr / sessionDict['processed'][timeColumnName]
    velY_fr = vertError_fr / sessionDict['processed'][timeColumnName]
    
    sessionDict['processed'][(outColumnName,'X')] = velX_fr
    sessionDict['processed'][(outColumnName,'Y')] = velY_fr
    
    print('*** calcAngularVelocity(): Added sessionDict[\'processed\'][(\'' + outColumnName + '] ***')

    return sessionDict


def calcVizardDeltaT(sessionDict):

    #sessionDict['processed']['frameTime'] = pd.to_datetime(sessionDict['raw']['frameTime'],unit='s')
    #sessionDict['processed']['frameTime'] = sessionDict['raw']['frameTime']
    deltaTime = pd.to_datetime(sessionDict['raw']['frameTime'],unit='s').diff()
    deltaTime.loc[deltaTime.dt.microseconds==0] = pd.NaT
    deltaTime = deltaTime.fillna(method='bfill', limit=1)
    sessionDict['processed']['vizardDeltaT'] = deltaTime.dt.microseconds / 1000000

    print('*** calcVizardDeltaT(): Added sessionDict[\'processed\'][\'frameTime\'] ***')

    return sessionDict

def calcAngularCalibrationError(sessionDict):

    eyeToCalibrationPointDirDf = metricEyeOnScreenToEyeInHead(sessionDict,sessionDict['calibration']['calibPointMetricEyeOnScreen'],'eyeToCalibrationPoint')
    cycEIHDirDf = metricEyeOnScreenToEyeInHead(sessionDict,sessionDict['calibration']['cycMetricEyeOnScreen'],'cycEIH')

    zVec_fr_xyz = from1x3_to_1x4([0,0,1],eyeOffsetX=0.0,numReps = len(sessionDict['calibration']))
    zVec_fr_xyz = zVec_fr_xyz[:,0:3]

    yVec_fr_xyz = from1x3_to_1x4([0,1,0],eyeOffsetX=0.0,numReps = len(sessionDict['calibration']))
    yVec_fr_xyz = zVec_fr_xyz[:,0:3]

    cycEIHDir_fr_xyz = cycEIHDirDf.values
    eyeToCalibDir_fr_xyz = eyeToCalibrationPointDirDf.values


    def vecDot(a,b):
            res1 = np.einsum('ij, ij->i', a, b)
            res2 = np.sum(a*b, axis=1)
            return res2

    cycVertAngle_fr = np.rad2deg( np.arctan2( cycEIHDir_fr_xyz[:,1], cycEIHDir_fr_xyz[:,2]))
    cycHorizAngle_fr = np.rad2deg( np.arctan2( cycEIHDir_fr_xyz[:,0], cycEIHDir_fr_xyz[:,2]))

    calibVertAngle_fr = np.rad2deg( np.arctan2( eyeToCalibDir_fr_xyz[:,1], eyeToCalibDir_fr_xyz[:,2]))
    calibHorizAngle_fr = np.rad2deg( np.arctan2( eyeToCalibDir_fr_xyz[:,0], eyeToCalibDir_fr_xyz[:,2]))

    cycAngle_elAz = np.array([cycHorizAngle_fr,cycVertAngle_fr],dtype=np.float)
    calibAngle_elAz = np.array([calibHorizAngle_fr,calibVertAngle_fr],dtype=np.float)

    angularError = findResidualError(cycAngle_elAz,calibAngle_elAz) / sessionDict['analysisParameters']['numberOfCalibrationPoints']
    print('*****Residual angular error: *****' + str(angularError) + '*****')

    sessionDict['analysisParameters']['angularCalibrationError'] = angularError
    return sessionDict

def calcAngularDisplacementsDuringBlank(sessionDict):
    '''
    Calculates

    Appends:
    sessionDict['trialInfo']['gazeBallDisplacementRatio']
    sessionDict['trialInfo']['gazeDisplacementDuringBlank']
    sessionDict['trialInfo']['ballDisplacementDuringBlank']
    
    '''
    def displacementDuringBlank(sessionDict,columnName):
    
        displacement_tr = []

        for trNum in range(len(sessionDict['trialInfo'])):

            ballOffFr = sessionDict['trialInfo']['ballOffFr'].values[trNum]
            ballOnFr = sessionDict['trialInfo']['ballOnFr'].values[trNum]

            vector_XYZ_fr = sessionDict['processed'][columnName]

            vOff = vector_XYZ_fr.loc[ballOffFr].values
            vOn = vector_XYZ_fr.loc[ballOnFr].values
            vDot = np.vdot(vOff,vOn)
            displacement_tr.append(np.rad2deg(np.arccos(vDot)))

        return displacement_tr

    sessionDict['trialInfo']['gazeDisplacementDuringBlank']  = calcDisplacementDuringBlank(sessionDict,'cycGIWDir')
    sessionDict['trialInfo']['ballDisplacementDuringBlank']  = calcDisplacementDuringBlank(sessionDict,'cycToBallDir')
    
    def calcGazeBallDisplacementRatio(sessionDict):
    
        def calcRatioForARow(row):
            return row['gazeDisplacementDuringBlank'] / row['ballDisplacementDuringBlank']
    
        sessionDict['trialInfo']['gazeBallDisplacementRatio'] = sessionDict['trialInfo'].apply(calcRatioForARow, axis=1)
    
        return sessionDict

    # Remove outliers?
    if( sessionDict['analysisParameters']['outlierThresholdSD'] ):
        print('calcAngularDisplacementsDuringBlank(): Removing outlier from gazeDisplacementDuringBlank')
        sessionDict = removeOutliers(sessionDict, 'gazeDisplacementDuringBlank')
    
    # Calculate gazeBallDisplacementRatio
    sessionDict = calcGazeBallDisplacementRatio(sessionDict)
    
    return sessionDict

def removeOutliers(sessionDict, columnName):
    '''
    Compatible with multiindex dataframes. 
    columnName can refer to the top level of a multiindex). eg columnName='CatchError'
    or a multiindex eg columnName=[('CatchError','X')]
    
    returns 
    '''
    # Add it to a list of variables that have been pruned of outliers
    if( 'removedOutliersFrom' in sessionDict['analysisParameters'].keys() ):
        
        # Prevent double-pruning
        if( columnName in sessionDict['analysisParameters']['removedOutliersFrom']):
            raise AssertionError('Column ' +  ' in trialInfo has already been pruned of outliers')
        else:
            sessionDict['analysisParameters']['removedOutliersFrom'].append(columnName)
            
    else:
        sessionDict['analysisParameters']['removedOutliersFrom'] = [columnName]
        
    outlierThresholdSD = sessionDict['analysisParameters']['outlierThresholdSD']
    
    def pruneOutliersFromSeries(series,outlierThresholdSD):
        
        data_tr = series.values
        mean = series.mean()
        std = series.std()

        outlierIdx = []

        [outlierIdx.append(i) for i,v in enumerate(data_tr) if abs(v - mean) > outlierThresholdSD * std]

        data_tr[outlierIdx] = np.NaN

        return data_tr
    
        
    if(  type(sessionDict['trialInfo'][columnName]) == pd.DataFrame ):
        
        sessionDict['trialInfo'][columnName] = sessionDict['trialInfo'][columnName].apply(
            pruneOutliersFromSeries,outlierThresholdSD=outlierThresholdSD,axis=1)

    elif( type(sessionDict['trialInfo'][columnName]) == pd.Series ):
        
        sessionDict['trialInfo'][(columnName,'')] = pruneOutliersFromSeries(sessionDict['trialInfo'][columnName],outlierThresholdSD=outlierThresholdSD)
    
    return sessionDict

def calcDisplacementDuringBlank(sessionDict,columnName):

    displacement_tr = []

    for trNum in range(len(sessionDict['trialInfo'])):

        ballOffFr = sessionDict['trialInfo']['ballOffFr'].values[trNum]
        ballOnFr = sessionDict['trialInfo']['ballOnFr'].values[trNum]

        vector_XYZ_fr = sessionDict['processed'][columnName]

        vOff = vector_XYZ_fr.loc[ballOffFr].values
        vOn = vector_XYZ_fr.loc[ballOnFr].values
        vDot = np.vdot(vOff,vOn)
        displacement_tr.append(np.rad2deg(np.arccos(vDot)))

    return displacement_tr

def calcGazeBallDisplacementRatio(sessionDict):
    '''
    Calculates angular displacement of the ball during the blank.
    Samples cyc eye-to-ball vector at ballOff and ballOn and calculates the angular difference
    Perhaps it is more accurate to estimate the measure form a fixed viewpoint?

    output: appends ['gazeBallDisplacementRatio'] to sessionDict['trialInfo']

    '''    
    def calcRatioForARow(row):
        return row['gazeDisplacementDuringBlank'] / row['ballDisplacementDuringBlank']
    
    sessionDict['trialInfo']['gazeBallDisplacementRatio'] = sessionDict['trialInfo'].apply(calcRatioForARow, axis=1)
    
    return sessionDict


def calcGazeErrorRelToBallOn(sessionDict,timeOffset,eyeString = 'cyc'):
    '''
    Calculates angular distance between cyc GIW and cyc-to-ball vector at a time relative to ball On.
    appends 'cycGIWtoBallAngle_ballOn_%1.2f' % timeOffset to sessionDict['trialInfo']
    returens sessionDict
    '''
    blankForNFrames = np.round( timeOffset / sessionDict['analysisParameters']['fps'])

    plotFr_tr = np.array(np.add(blankForNFrames,sessionDict['trialInfo']['ballOnFr']),dtype=np.int)

    # Unsigned
    for subLevelNameStr in list(sessionDict['processed']['cycGIWtoBallAngle'].columns):

        cycToBallAng_fr = sessionDict['processed']['cycGIWtoBallAngle'][subLevelNameStr].values[plotFr_tr]
        columnName = 'cycGIWtoBallAngle_ballOn_%1.2f' % timeOffset
        sessionDict['trialInfo'][(columnName,subLevelNameStr)] = cycToBallAng_fr

    return sessionDict



def calcUnsignedGazeToBallError(sessionDict,eyeString,upVecString):
    '''
    Calculates angular distance between cyc GIW and cyc-to-ball vector for all frames
    Appends [('cycGIWtoBallAngle','2D')] to sessionDict['processed]
    Returns sessionDict
    '''
    zDir_fr_xyz = []
    
    if( eyeString == 'cyc'):    

        if(columnExists(sessionDict['processed'],'cycToBallDir') is False):    
            sessionDict = calcCycToBallVector(sessionDict)
        
        zDir_fr_xyz = sessionDict['processed']['cycToBallDir'].values
        
    else:
        raise AssertionError('gazeToBallErrorInHead: Currently only works with cyc eye')

    def angDist(row):

        vdot = np.vdot(row['cycToBallDir'], row['cycGIWDir'])
        return np.rad2deg(np.arccos(vdot))

    gazeError_fr = sessionDict['processed'].apply(angDist,axis=1)

    sessionDict['processed'][('cycGIWtoBallAngle','2D')] = gazeError_fr

    print('calcUnsignedGazeToBallError(): Created sessionDict[\'processed\'](\'cycGIWtoBallAngle\',\'2D\')')

    return sessionDict


def calcGazeToBallError(sessionDict,eyeString,upVecString):
    '''
    Calculate gaze error along the vertical axis defined by gravity
    ...and the horizontal axis that is orthogonal to the gaze vector but parallel to the ground plane    
    
    '''
    
    if( eyeString is not 'cyc'):
        raise AssertionError('gazeToBallErrorInHead: Currently only works with cyc eye')
    
    ######################################################################################################
    ######################################################################################################
    ### Unsigned error
    
    sessionDict = calcUnsignedGazeToBallError(sessionDict,eyeString,upVecString)
    
    ######################################################################################################
    ######################################################################################################
    ### Calculate some vectors necessary for calculating gaze error
    
    
    if(columnExists(sessionDict['processed'],'headUpInWorldDir') is False):
        sessionDict = calcHeadUpDir(sessionDict)
    
    if(columnExists(sessionDict['processed'],'worldUpDir') is False):
        sessionDict = calcWorldUpDir(sessionDict)

    if(columnExists(sessionDict['processed'],'orthToBallAndHeadUpDir') is False):
        sessionDict = calcOrthToBallAndHeadUpDir(sessionDict)        

    if(columnExists(sessionDict['processed'],'orthToBallAndWorldUpDir') is False):
        sessionDict = calcOrthToBallAndWorldUpDir(sessionDict)        
    
    ######################################################################################################
    ######################################################################################################
    ### Signed error
    
    # xDir, yDir, and Zdir form a new basis
    # zDir extends in depth, and will form the center of the scatterplot
    # Ball direction will be scattered around it.

    # Setup a new basis.
    zDir_fr_xyz = [] # Depth along the eye to ball vector
    yDir_fr_xyz = [] # Up with gravity
    xDir_fr_xyz = [] # Orthogonal to gravity and eye to ball vector    
    

    ####################################
    ### Get zDir_fr_xyz
    if( eyeString == 'cyc'):

        if(columnExists(sessionDict['processed'],'cycToBallDir') is False):
            sessionDict = calcCycToBallVector(sessionDict)
        
        zDir_fr_xyz = sessionDict['processed']['cycToBallDir'].values

    ####################################
    ### Get yDir_fr_xyz
    
    if(upVecString == 'worldUp'):

        yDir_fr_xyz = sessionDict['processed']['worldUpDir'].values

    elif(upVecString == 'headUp'):
        
        raise AssertionError('gazeToBallErrorInHead: headUp may work, but Gabe hasn not yet reasoned through the geometry yet.  So, for now, Im throwing an error.')
        
        yDir_fr_xyz = sessionDict['processed']['headUpInWorldDir'].values
        
    else:
        raise AssertionError('gazeToBallErrorInHead: Expected upVecString of worldUp or headUp')

    ####################################
    ### Get yDir_fr_xyz
                             
    gazeDir_fr_xyz = sessionDict['processed']['cycGIWDir'].values

    # axis is orthogonal to gaze dir and world up
    # parallel to ground plane and at eye height
    azimuthulVec = np.cross(gazeDir_fr_xyz,yDir_fr_xyz)
    azimuthulDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in azimuthulVec],dtype=np.float)
    
    # orthogonal to gaze vector and azimuthulDir
    elevationVec = np.cross(azimuthulDir,gazeDir_fr_xyz);
    elevationDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in elevationVec],dtype=np.float)

    def vecDot(a,b):
        res1 = np.einsum('ij, ij->i', a, b)
        res2 = np.sum(a*b, axis=1)
        return res2

    vertError_fr = np.rad2deg( np.arctan2( vecDot(zDir_fr_xyz,elevationDir), vecDot(zDir_fr_xyz,gazeDir_fr_xyz)))
    horzError_fr = np.rad2deg( np.arctan2( vecDot(zDir_fr_xyz,azimuthulDir), vecDot(zDir_fr_xyz,gazeDir_fr_xyz)))
    
    outVarColName = 'cycGIWtoBallAngle'
    
    sessionDict['processed'][(outVarColName,'X_' + upVecString)] = horzError_fr
    sessionDict['processed'][(outVarColName,'Y_' + upVecString)] = vertError_fr

    print('calcGazeToBallError(): Created sessionDict[\'processed\']' + '[\'' + outVarColName + '\']')

    return sessionDict

def calcCycToBallVector(sessionDict):

    cycToBallVec = np.array(sessionDict['raw']['ballPos'] - sessionDict['raw']['viewPos'],dtype=np.float )
    cycToBallDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in cycToBallVec],dtype=np.float)

    sessionDict['processed'][('cycToBallDir','X')] = cycToBallDir[:,0]
    sessionDict['processed'][('cycToBallDir','Y')] = cycToBallDir[:,1]
    sessionDict['processed'][('cycToBallDir','Z')] = cycToBallDir[:,2]

    return sessionDict


def calcBallOffOnFr(sessionDict):

    gbTrials = sessionDict['processed'].groupby('trialNumber')

    xErr_tr = []
    yErr_tr = []


    ballOffIdx_tr = []
    ballOnIdx_tr = []

    for trNum,tr in gbTrials:

        ballOffIdx_tr.append(tr.index[0] + findFirst(tr['eventFlag'],'ballRenderOff'))
        ballOnIdx_tr.append(tr.index[0] + findFirst(tr['eventFlag'],'ballRenderOn'))

    sessionDict['trialInfo'][('ballOnFr','')] = ballOnIdx_tr
    sessionDict['trialInfo'][('ballOffFr','')] = ballOffIdx_tr

    return sessionDict

def calcOrthToBallAndHeadUpDir(sessionDict):
    # Lateral vector:  orthToBallAndHeadUpDir = cross( headUpDir, cycToBall)
    
    headUpLatVec = np.cross(sessionDict['processed']['cycToBallDir'],sessionDict['processed']['headUpInWorldDir'])
    headUpLatDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in headUpLatVec],dtype=np.float)

    orthToBallAndHeadUpDirDf = pd.DataFrame({('orthToBallAndHeadUpDir','X'):headUpLatDir[:,0],
                                  ('orthToBallAndHeadUpDir','Y'):headUpLatDir[:,1],
                                  ('orthToBallAndHeadUpDir','Z'):headUpLatDir[:,2]})

    sessionDict['processed'] = pd.concat([sessionDict['processed'],orthToBallAndHeadUpDirDf],axis=1)
    return sessionDict

def calcOrthToBallAndWorldUpDir(sessionDict):
    # Lateral vector:  orthToBallAndWorldUpDir = cross( worldUpDir,cycToBall)
    
    #worldUp_fr = np.array([[0,1,0]] * len(sessionDict['raw']))

    worldUpLatVec = np.cross(sessionDict['processed']['cycToBallDir'],sessionDict['processed']['worldUpDir'])
    worldUpLatDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in worldUpLatVec],dtype=np.float)

    orthToBallAndWorldUpDirDf = pd.DataFrame({('orthToBallAndWorldUpDir','X'):worldUpLatDir[:,0],
                                  ('orthToBallAndWorldUpDir','Y'):worldUpLatDir[:,1],
                                  ('orthToBallAndWorldUpDir','Z'):worldUpLatDir[:,2]})

    sessionDict['processed'] = pd.concat([sessionDict['processed'],orthToBallAndWorldUpDirDf],axis=1)

    return sessionDict



def calcHeadUpDir(sessionDict):
    import Quaternion as qu

    ###################################################
    ###################################################
    ###  Calculate head up
    
    # Get rotation matrix for the head
    viewRotMat_fr_mRow_mCol = [qu.Quat(np.array(q,dtype=np.float))._quat2transform() for q in sessionDict['raw'].viewQuat.values]

    headUpInHeadVec_fr_XYZ = np.array([[0,1,0]] * len(viewRotMat_fr_mRow_mCol))

    headUpInWorldDir_fr_XYZ = np.array([ np.dot(viewRotMat_fr_mRow_mCol[fr],headUpInHeadVec_fr_XYZ[fr]) 
             for fr in range(len(headUpInHeadVec_fr_XYZ))])

    headUpInWorldDirDf = pd.DataFrame({('headUpInWorldDir','X'):headUpInWorldDir_fr_XYZ[:,0],
                                 ('headUpInWorldDir','Y'):headUpInWorldDir_fr_XYZ[:,1],
                                 ('headUpInWorldDir','Z'):headUpInWorldDir_fr_XYZ[:,2]})
    # Concat
    sessionDict['processed'] = pd.concat([sessionDict['processed'],headUpInWorldDirDf],axis=1)

    return sessionDict

def calcWorldUpDir(sessionDict):
    
    worldUp_fr_xyz = np.array(np.array([[0,1,0]] * len(sessionDict['raw'])),dtype=np.float)
    
    worldUpDf = pd.DataFrame({('worldUpDir','X'):worldUp_fr_xyz[:,0],
                                 ('worldUpDir','Y'):worldUp_fr_xyz[:,1],
                                 ('worldUpDir','Z'):worldUp_fr_xyz[:,2]})

    sessionDict['processed'] = pd.concat([sessionDict['processed'],worldUpDf],axis=1)

    return sessionDict


def calcGIWDirection(sessionDict):

	print('calcGIWDirectionAndVelocity(): Currently only calculates GIW angle/vel for cyc eye data')

	# cycFiltEyeOnScreen -> filtCycMetricEyeOnScreen
	tempDF = eyeOnScreenToMetricEyeOnScreen(sessionDict,sessionDict['processed']['cycFiltEyeOnScreen'],'filtCycMetricEyeOnScreen')
	sessionDict['processed'] = pd.concat([sessionDict['processed'],tempDF],axis=1)

	# filtCycMetricEyeOnScreen -> filtCycEyeInHeadDirDf
	filtCycEyeInHeadDirDf = metricEyeOnScreenToEyeInHead(sessionDict,sessionDict['processed']['filtCycMetricEyeOnScreen'],'filtCycEyeInHeadDir')
	sessionDict['processed'] = pd.concat([sessionDict['processed'],filtCycEyeInHeadDirDf],axis=1)

	# filtCycEyeInHeadDir -> filtCycGazeNodeInWorld
	# filtCycGazeNodeInWorld is a point 1 meter from the head along the gaze direction 
	# This is not yet the GIW angle
	filtCycGazeNodeInWorldDF  = headToWorld(sessionDict,sessionDict['processed']['filtCycEyeInHeadDir'],'filtCycGazeNodeInWorldDF')
	sessionDict['processed'] = pd.concat([sessionDict['processed'],filtCycGazeNodeInWorldDF],axis=1)

	# filtCycGazeNodeInWorld -> filtCycGIWDir
	filtCycGIWDirDF = calcDirFromDF(sessionDict['raw']['viewPos'],sessionDict['processed']['filtCycGazeNodeInWorldDF'],'cycGIWDir')
	sessionDict['processed'] = pd.concat([sessionDict['processed'],filtCycGIWDirDF],axis=1)

	#cycGazeVelDf = GIWtoGazeVelocity(sessionDict,sessionDict['processed']['cycGIWDir'],'cycGazeVelocity')
	#sessionDict['processed'] = pd.concat([sessionDict['processed'],cycGazeVelDf],axis=1)

	return sessionDict

def calcUnsignedAngularVelocity(vector_fr,deltaTime_fr):
	'''
	Moving window takes cosine of the dot product for adjacent values.
	Appends a 0 onto the end of the vector.
	'''
	    
	angularDistance_fr = np.array(  [ np.rad2deg(np.arccos( np.vdot( vector_fr[fr,:],vector_fr[fr-1,:])))
	     for fr in range(1,len(vector_fr))]) # if range starts at 0, fr-1 wil be -1.  Keep range from 1:len(vector)

	angularDistance_fr = np.append(0, angularDistance_fr)
	angularVelocity_fr = np.divide(angularDistance_fr,deltaTime_fr)

	return angularVelocity_fr

# def GIWtoGazeVelocity(sessionDict,dfIn,columnLabelOut):
    
# 	rawDF = sessionDict['raw']
# 	procDF = sessionDict['processed']

# 	if( columnExists(procDF, 'smiDeltaT') is False):
# 	    sessionDict = calcSMIDeltaT(sessionDict)
# 	    procDF = sessionDict['processed']

# 	angularVelocity_fr = calcUnsignedAngularVelocity(dfIn.values, procDF['smiDeltaT'].values) 

# 	return pd.DataFrame({(columnLabelOut,''):angularVelocity_fr})


def calcSMIDeltaT(sessionDict):

    sessionDict['processed']['smiDateTime'] = pd.to_datetime(sessionDict['raw'].eyeTimeStamp,unit='ns')
    deltaTime = sessionDict['processed']['smiDateTime'].diff()
    deltaTime.loc[deltaTime.dt.microseconds==0] = pd.NaT
    deltaTime = deltaTime.fillna(method='bfill', limit=1)
    sessionDict['processed']['smiDeltaT'] = deltaTime.dt.microseconds / 1000000
    return sessionDict


def GIWtoGazeVelocity(sessionDict,dfIn,columnLabelOut):
    
    rawDF = sessionDict['raw']
    procDF = sessionDict['processed']

    if( columnExists(procDF, 'smiDeltaT') is False):
        sessionDict = calcSMIDeltaT(sessionDict)
        procDF = sessionDict['processed']

    angularVelocity_fr = calcUnsignedAngularVelocity(dfIn.values, procDF['smiDeltaT'].values) 
    
    return pd.DataFrame({(columnLabelOut,''):angularVelocity_fr})


# def calculateLinearHomography(sessionDict, plottingFlag = False):

# 	print 'calculateLinearHomography():  Currently only reprojects  cyc gaze data'

# 	#sessionDict['calibration']['cycEyeOnScreen']
# 	calibDataFrame = sessionDict['calibration']
# 	gbCalibSession = calibDataFrame.groupby(['trialNumber'])

# 	numberOfCalibrationSession =  (max(calibDataFrame.trialNumber.values)%1000)//100

# 	trialOffset = 1000;
# 	frameOffset = 0

# 	numberOfCalibrationPoints = 27
# 	numberOfCalibrationSession = 0

# 	# Calculate a homography for each session
# 	for i in range(numberOfCalibrationSession + 1):
	    
#                 startTrialNumber = trialOffset + 100*i 
#                 endTrialNumber = trialOffset + 100*i + numberOfCalibrationPoints - 1

#                 firstCalibrationSession = gbCalibSession.get_group(startTrialNumber)
#                 lastCalibrationSession = gbCalibSession.get_group(endTrialNumber)

#                 numberOfCalibrationFrames = max(lastCalibrationSession.index) - min(firstCalibrationSession.index) + 1
#                 dataRange = range(frameOffset, frameOffset + numberOfCalibrationFrames)

#                 #print 'Frame Range =[', min(dataRange),' ', max(dataRange), ']'  

#                 cycMetricEyeOnScreen = eyeOnScreenToMetricEyeOnScreen(sessionDict,sessionDict['calibration']['cycEyeOnScreen'],
#                                                 'cycMetricEyeOnScreen')

#                 sessionDict['calibration'][('cycMetricEyeOnScreen','X')] = cycMetricEyeOnScreen[('cycMetricEyeOnScreen','X')] 
#                 sessionDict['calibration'][('cycMetricEyeOnScreen','Y')] = cycMetricEyeOnScreen[('cycMetricEyeOnScreen','Y')] 
#                 sessionDict['calibration'][('cycMetricEyeOnScreen','Z')] = cycMetricEyeOnScreen[('cycMetricEyeOnScreen','Z')] 

#                 cycMetricEyeOnScreen.drop(('cycMetricEyeOnScreen','Z'),axis=1,inplace=True)

#                 cycMetricEyeOnScreen = cycMetricEyeOnScreen.T[dataRange]

#                 calibPointMetricEyeOnScreen_fr_XYZ = np.array([calibDataFrame['calibPointMetricEyeOnScreen']['X'][dataRange].values, 
#                                calibDataFrame['calibPointMetricEyeOnScreen']['Y'][dataRange].values], dtype = float)

#                 cycMetricEyeOnScreen = np.array(cycMetricEyeOnScreen,dtype=np.float).T
#                 calibPointMetricEyeOnScreen_fr_XYZ = np.array(calibPointMetricEyeOnScreen_fr_XYZ,dtype=np.float).T

#                 #print cycMetricEyeOnScreen
#                 sessionDict = calibrateData(sessionDict,cycMetricEyeOnScreen, calibPointMetricEyeOnScreen_fr_XYZ, cv2.RANSAC, 10, plottingFlag = False)

#                 #frameOffset = numberOfCalibrationFrames
#                 #return cycMetricEyeOnScreen, calibPointMetricEyeOnScreen_fr_XYZ,dataRange
	    
# 	#print startTrialNumber, endTrialNumber
# 	#len(calibDataFrame)
# 	return sessionDict


def findResidualError(projectedPoints, referencePoints):

    e2 = np.zeros((projectedPoints.shape[0],2))
    
    for i in range(projectedPoints.shape[0]):
        temp = np.subtract(projectedPoints[i], referencePoints[i])
        e2[i,:] = np.power(temp[0:2], 2)
    
    return [np.sqrt(sum(sum(e2[:])))]

# def calibrateData(sessionDict,cycMetricEyeOnScreen, calibPointMetricEyeOnScreen_fr_XYZ, 
#                   method = cv2.LMEDS, threshold = 10, plottingFlag = False):

#     result = cv2.findHomography(cycMetricEyeOnScreen, calibPointMetricEyeOnScreen_fr_XYZ)#, method , ransacReprojThreshold = threshold)
#     totalFrameNumber = calibPointMetricEyeOnScreen_fr_XYZ.shape[0]
    
#     arrayOfOnes = np.ones((totalFrameNumber,1), dtype = float)
#     homogrophy = result[0]
    
#     cycMetricEyeOnScreen = np.hstack((cycMetricEyeOnScreen, arrayOfOnes))
#     calibPointMetricEyeOnScreen_fr_XYZ = np.hstack((calibPointMetricEyeOnScreen_fr_XYZ, arrayOfOnes))
    
#     projCycMetricEyeOnScreen = np.zeros((totalFrameNumber,3))
    
#     for i in range(totalFrameNumber):
#         projCycMetricEyeOnScreen[i,:] = np.dot(homogrophy, cycMetricEyeOnScreen[i,:])
        
#     hmd = sessionDict['analysisParameters']['hmd']
    
#     # Convert cycMetricEyeOnScreen to pixel values
#     cycMetricEyeOnScreenDf = pd.DataFrame({('cycMetricEyeOnScreen' ,'X'):cycMetricEyeOnScreen[:,0],('cycMetricEyeOnScreen' ,'Y'):cycMetricEyeOnScreen[:,1]})
#     cycEyeOnScreenDf = metricEyeOnScreenToPixels(sessionDict,cycMetricEyeOnScreenDf['cycMetricEyeOnScreen'],'cycEyeOnScreen')
#     cycEyeOnScreen_fr_XYZ = cycEyeOnScreenDf['cycEyeOnScreen'].values
    
#     # Convert projCycMetricEyeOnScreen to pixel values
#     projCycMetricEyeOnScreenDf = pd.DataFrame({('projCycMetricEyeOnScreen' ,'X'):projCycMetricEyeOnScreen[:,0],('projCycMetricEyeOnScreen' ,'Y'):projCycMetricEyeOnScreen[:,1]})
#     projCycEyeOnScreenDf = metricEyeOnScreenToPixels(sessionDict,projCycMetricEyeOnScreenDf['projCycMetricEyeOnScreen'],'projCycEyeOnScreen')
#     projCycEyeOnScreen_fr_XYZ = projCycEyeOnScreenDf['projCycEyeOnScreen'].values
    
#     # Convert calibPointMetricEyeOnScreen to pixel values
#     calibPointMetricEyeOnScreenDf = pd.DataFrame({('calibPointMetricEyeOnScreen' ,'X'):calibPointMetricEyeOnScreen_fr_XYZ[:,0],
#                                                   ('calibPointMetricEyeOnScreen' ,'Y'):calibPointMetricEyeOnScreen_fr_XYZ[:,1]})
#     calibPointEyeOnScreenDf = metricEyeOnScreenToPixels(sessionDict,calibPointMetricEyeOnScreenDf['calibPointMetricEyeOnScreen'],
#                                                         'calibPointEyeOnScreen')
#     calibPointEyeOnScreen_fr_XYZ = calibPointEyeOnScreenDf.values
    
#     data = projCycEyeOnScreen_fr_XYZ
#     frameCount = range(len(cycMetricEyeOnScreen))
    
#     if( plottingFlag == True ):
#         xmin = 550#min(cycMetricEyeOnScreen[frameCount,0])
#         xmax = 1350#max(cycMetricEyeOnScreen[frameCount,0])
#         ymin = 250#min(cycMetricEyeOnScreen[frameCount,1])
#         ymax = 800#max(cycMetricEyeOnScreen[frameCount,1])
#         fig1 = plt.figure()
#         plt.plot(data[frameCount,0], data[frameCount,1], 'bx', label='Calibrated POR')
#         plt.plot(cycMetricEyeOnScreen[frameCount,0], cycMetricEyeOnScreen[frameCount,1], 'g2', label='Uncalibrated POR')
#         plt.plot(calibPointMetricEyeOnScreen_fr_XYZ[frameCount,0], calibPointMetricEyeOnScreen_fr_XYZ[frameCount,1], 'r8', label='Ground Truth POR')
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         if ( method == cv2.RANSAC):
#             methodTitle = ' RANSAC '
#         elif( method == cv2.LMEDS ):
#             methodTitle = ' Least Median '
#         elif( method == 0 ):
#             methodTitle = ' Homography '
#         plt.title('Calibration Result using'+ methodTitle+'\nWith System Calibration ')
#         plt.grid(True)
#         legend = plt.legend(loc=[0.85,0.6], shadow=True, fontsize='small')# 'upper center'
#         plt.show()
        
#     print 'MSE_after = ', findResidualError(projCycMetricEyeOnScreen, calibPointMetricEyeOnScreen_fr_XYZ)
#     print 'MSE_before = ', findResidualError(cycMetricEyeOnScreen, calibPointMetricEyeOnScreen_fr_XYZ)
    
#     # Leaving cycEyeOnScreenDf, because it is redundant with the raw data
#     #sessionDict['processed'] = pd.concat([sessionDict['processed'],cycEyeOnScreenDf],axis=1,verify_integrity=True)
    
#     # Add variables to sessionDict
#     sessionDict['processed'] = pd.concat([sessionDict['processed'],projCycEyeOnScreenDf],axis=1,verify_integrity=True)
#     sessionDict['processed'] = pd.concat([sessionDict['processed'],calibPointEyeOnScreenDf],axis=1,verify_integrity=True)
    
    
#     #  Store the homography for each frame in processed data
    
#     #print 'homogrophy = ', homogrophy
#     tempMatrix = np.zeros((len(sessionDict['processed']),3,3))
#     tempMatrix[:,0:3,0:3] = homogrophy
#     tempMatrix = tempMatrix.reshape((len(sessionDict['processed']),9))
#     tempDf = []
#     tempDf = pd.DataFrame({
#                             ('linearHomography','0'):tempMatrix[:,0],
#                             ('linearHomography','1'):tempMatrix[:,1],
#                             ('linearHomography','2'):tempMatrix[:,2],
#                             ('linearHomography','3'):tempMatrix[:,3],
#                             ('linearHomography','4'):tempMatrix[:,4],
#                             ('linearHomography','5'):tempMatrix[:,5],
#                             ('linearHomography','6'):tempMatrix[:,6],
#                             ('linearHomography','7'):tempMatrix[:,7],
#                             ('linearHomography','8'):tempMatrix[:,8]
#                             })


#     sessionDict['processed']  = pd.concat([sessionDict['processed'],tempDf],axis=1,verify_integrity=True)
    
#     return sessionDict


def calcCalibPointMetricEyeOnScreen(sessionDict):

    calibDataFrame = sessionDict['calibration']
    
    framesPerPoint = range(100)
    startFrame = 0
    endFrame = len(calibDataFrame)
    frameIndexRange = range(startFrame, endFrame)

    cameraCenterPosition = np.array([0.0,0.0,0.0])
    planeNormal = np.array([0.0,0.0,1.0])
    eyetoScreenDistance = sessionDict['analysisParameters']['averageEyetoScreenDistance'] 
    screenCenterPosition = np.array([0.0,0.0,eyetoScreenDistance])

    calibPointMetricLocOnScreen_XYZ = np.empty([1, 3], dtype = float)
    for i in range(endFrame):
        
        lineNormal = calibDataFrame['calibrationPos'][['X','Y','Z']][i:i+1].values
        
        # TODO: I kinda cheated here by {line[0]}
        tempPos = findLinePlaneIntersection( cameraCenterPosition, lineNormal[0], 
                                               planeNormal, screenCenterPosition ) 
            
        calibPointMetricLocOnScreen_XYZ = np.vstack((calibPointMetricLocOnScreen_XYZ, tempPos))

    # TODO: I hate creating an empty variable and deleting it later on there should be a better way
    calibPointMetricLocOnScreen_XYZ = np.delete(calibPointMetricLocOnScreen_XYZ, 0, 0)
    print('Size of TruePOR array:', calibPointMetricLocOnScreen_XYZ.shape)

    # Attaching the calculated Values to the CalibDataFrame

    sessionDict['calibration'][('calibPointMetricEyeOnScreen','X')]  =  calibPointMetricLocOnScreen_XYZ[:,0]
    sessionDict['calibration'][('calibPointMetricEyeOnScreen','Y')]  =  calibPointMetricLocOnScreen_XYZ[:,1]
    sessionDict['calibration'][('calibPointMetricEyeOnScreen','Z')]  =  calibPointMetricLocOnScreen_XYZ[:,2]

    return sessionDict


def calcDirFromDF(dF1,dF2,labelOut):
    vecDF = dF2-dF1
    dirDF = vecDF.apply(lambda x: np.divide(x,np.linalg.norm(x)),axis=1)
    mIndex = pd.MultiIndex.from_tuples([(labelOut,'X'),(labelOut,'Y'),(labelOut,'Z')])
    dirDF.columns = mIndex
    return dirDF

def calcEyePositions(sessionDict):

    rawDF = sessionDict['raw']
    procDF = sessionDict['processed']
    
    ## Right and left eye positions
    #(sessionDict,dataIn,eyeString,dataOutLabel):
    #rightEyePosDf = eyeToWorld(rawDF,[0,0,0],'right','rightEyePos') 
    
    rightEyePosDf = eyeToWorld(sessionDict,[0,0,0],'right','rightEyeInWorld') 
    procDF = pd.concat([procDF,rightEyePosDf],axis=1,verify_integrity=True)

    leftEyePosDf = eyeToWorld(sessionDict,[0,0,0],'left','leftEyeInWorld') 
    procDF = pd.concat([procDF,leftEyePosDf],axis=1,verify_integrity=True)
    
    sessionDict['processed'] = procDF
    
    return sessionDict


def from1x3_to_1x4(dataIn_n_xyz, eyeOffsetX=0, numReps = 1):
    
    '''
    Converts dataIn_n_xyz into an Nx4 array, with eyeOffsetX added to the [0] column.  
    DataIn may be either a 3 element list (XYZ values) or an N x XYZ array, where N >1 (and equal to the number of rows of the original raw dataframe)

    Output is an nx4 array in which IOD has been added to the [0] column
    '''

    # If needed, tile dataIn_fr_xyz to match length of dataIn_fr_xyzw
    if( numReps == 0 ):
        
        raise NameError('numReps must be >0.')
        
    elif(numReps == 1):
        
        dataIn_fr_xyzw = np.tile([eyeOffsetX, 0, 0, 1.0],[len(dataIn_n_xyz),1])
        dataIn_fr_xyzw[:,:3] = dataIn_fr_xyzw[:,:3] + dataIn_n_xyz
        
    else:
        
        dataIn_fr_xyzw = np.tile([eyeOffsetX, 0, 0, 1.0],[numReps,1])
        dataIn_fr_xyzw[:,:3] = dataIn_fr_xyzw[:,:3] + np.tile(dataIn_n_xyz,[numReps,1])

    return dataIn_fr_xyzw

def eyeToHead(sessionDict,dataIn,eyeString,dataOutLabel):
    
    '''
    This function takes XYZ data in eye centered coordinates (XYZ) and transforms it into head centered coordinates.
    
    - rawDF must be the raw dataframe containing the transform matrix for the mainview
    
    - dataIn may be either:
        - a dataframe of XYZ data
        - a 3 element list of XYZ data

    - eyeString is a string indicating which FOR contains dataIn, and may be of the values ['cyc','right','left'] 
    
    Returns:  vec_fr_XYZW, where W is 1.0 (homogenous coordinates)
    
    '''
    rawDF = sessionDict['raw']
        
    ### Which eye?
    if( eyeString in ['cyc','right','left'] is False):
        raise AssertionError('Second argin (eyeType) must be cyc, right, or left')
    
    eyeOffsetX = []
    
    # Set IOD.  
    # To convert dataIn into head centered coordinates, this offset must be added to dataIn. 
    
    if(eyeString == 'cyc'):
        eyeOffsetX = 0.0
        
    elif(eyeString== 'right'):
        eyeOffsetX = rawDF['IOD'].mean()/2.0/1000.0
        
    elif(eyeString== 'left'):
        eyeOffsetX = -rawDF['IOD'].mean()/2.0/1000.0
        
    ######################################################################
    ######################################################################
    # Prepare the data.  
    # After the next block of code, the data should be
    # an np.array of size [nFrames, XYZW]
    # where W is always 1
    
    vec_fr_XYZW = []
    
    if( type(dataIn) == pd.DataFrame ):
        # dataIn is a dataframe
        vec_fr_XYZW = from1x3_to_1x4(dataIn.values,eyeOffsetX,numReps = 1)
        
    elif( len(np.shape(dataIn)) == 1 and len(dataIn) == 3):
        # dataIn is a 3 element list
        vec_fr_XYZW = from1x3_to_1x4(dataIn,eyeOffsetX, numReps = len(rawDF.viewMat))

    else:
        raise AssertionError('Third argin (dataIn) must be either an Nx3 dataframe or np.array, where N = num rows in rawDF (the frist argument passed into this function)') 
    
        
    #return vec_fr_XYZW
    
    ### dataOutLabel
    dataOutDf = pd.DataFrame(vec_fr_XYZW)
    mIndex = pd.MultiIndex.from_tuples([(dataOutLabel,'X'),(dataOutLabel,'Y'),(dataOutLabel,'Z'),(dataOutLabel,'W')])
    dataOutDf.columns = mIndex
    
    return dataOutDf 
    
def worldToScreen(sessionDict,xyzPosDf,columnNameOut):
    '''
    Convert from world coordinates to head centered coordinates
    '''

    rawDf = sessionDict['raw']

    # Convert viewmat data into 4x4 transformation matrix
    #viewMat_fr_4x4 = [np.reshape(mat,[4,4]).T for mat in rawDf.viewMat.values]

    # Calculate the inverse transformation matrix for the head 
    #inverseViewMat_fr_4x4 = [np.linalg.inv(np.array(mat,dtype=np.float)) for mat in viewMat_fr_4x4]

    inverseViewMat_fr_4x4 = [np.reshape(mat,[4,4]).T for mat in rawDf.cycInverseMat.values]

    # Turn xyzPosDf position into a 1x4 array for multiplication with the inverse view matrix
    pos_fr_xyzw = np.tile([0, 0, 0, 1.0],[len(xyzPosDf),1])
    pos_fr_xyzw[:,:3] = pos_fr_xyzw[:,:3] + xyzPosDf.values

    # Calculate the cyc-to-ball vector within the head centered FOR.  
    # Take the dot product of ballPos_fr_xyzw and inverseViewMat_fr_4x4
    posInHead_fr_xyzw = np.array([np.dot(inverseViewMat_fr_4x4[fr],pos_fr_xyzw[fr])
                              for fr in range(len(pos_fr_xyzw))])

    # Convert to dataframe. Discard 4th column.
    posInHead_fr_xyz = pd.DataFrame(posInHead_fr_xyzw[:,0:3])

    # Screen distance
    screenDist = sessionDict['analysisParameters']['averageEyetoScreenDistance']

    # Normalize to screen distance
    posInHeadDf = posInHead_fr_xyz.apply(lambda x: np.multiply(x,screenDist/x[2]),axis=1)

    sessionDict['processed'][(columnNameOut,'X')] = posInHeadDf[0]
    sessionDict['processed'][(columnNameOut,'Y')] = posInHeadDf[1]
    sessionDict['processed'][(columnNameOut,'Z')] = posInHeadDf[2]

    return sessionDict

def eyeToWorld(sessionDict,dataIn,eyeString,dataOutLabel):
    
    '''
    This function takes XYZ data in eye centered coordinates (XYZ) and transforms it into world centered coordinates.
    
    - rawDF must be the raw dataframe containing the transform matrix for the mainview
    
    - dataIn may be:
        - a dataframe of XYZ data
        - a 3 element list of XYZ data

    - eyeString is a string indicating which FOR contains dataIn, and may be of the values ['cyc','right','left'] 
    
    Returns:  A multiindexed dataframe with {(label,X),(label,Y),and (label,Z)}
    
    '''
    rawDF = sessionDict['raw'] # monkey
    
    ######################################################################    
    ######################################################################
        
    # Convert viewmat data into 4x4 transformation matrix
    viewMat_fr_4x4 = [np.reshape(mat,[4,4]).T for mat in rawDF.viewMat.values]
    
    
    # Convert dataIn from eye centered coordinates into head centered coordinates
    vec_fr_XYZW = eyeToHead(sessionDict,dataIn,eyeString,'dataInHead')
    
    # Take the dot product of vec_fr_XYZW and viewMat_fr_4x4
    dataOut_fr_XYZ = np.array([np.dot(viewMat_fr_4x4[fr],vec_fr_XYZW.values[fr])
                              for fr in range(len(vec_fr_XYZW['dataInHead'].values))])
    
    # Discard the 4th column
    dataOut_fr_XYZ = dataOut_fr_XYZ[:,:3]
    
    # Turn it into a dataframe
    dataOutDf = pd.DataFrame(dataOut_fr_XYZ)

    # Rename the columns
    mIndex = pd.MultiIndex.from_tuples([(dataOutLabel,'X'),(dataOutLabel,'Y'),(dataOutLabel,'Z')])
    dataOutDf.columns = mIndex
    
    return dataOutDf 

def headToWorld(sessionDict,dataIn,dataOutLabel):
    '''
    This function takes XYZ data in head centered coordinates (XYZ) and transforms it into world centered coordinates.

    - rawDF must be the raw dataframe containing the transform matrix for the mainview

    - dataIn may be:
        - a dataframe of XYZ data
        - a 3 element list of XYZ data

    - eyeString is a string indicating which FOR contains dataIn, and may be of the values ['cyc','right','left'] 

    Returns:  A multiindexed dataframe with {(label,X),(label,Y),and (label,Z)}

    '''

    return eyeToWorld(sessionDict,dataIn,'cyc',dataOutLabel)

###########################################################################
###########################################################################

def eyeOnScreenToMetricEyeOnScreen(sessionDict,dFIn,dataOutColumnLabel):
    '''
    Convert pixel coordinates
    to metric locations on a screen at sessionDict['analysisParameters']['averageEyeToScreenDistance']

    0.126,0.071 = Screen size in meters according to SMI manual
    averageEyetoScreenDistance = 0.0725
    Note that the output is in meters, and not normalized gaze vector
    '''
    x_pixel = dFIn['X']
    y_pixel = dFIn['Y']
    z = []


    if(  sessionDict['analysisParameters']['hmd'].upper() == 'DK2' ):
        
        resolution_XY = sessionDict['analysisParameters']['hmdResolution']
        pixelSize_XY = sessionDict['analysisParameters']['hmdScreenSize']

        x = (pixelSize_XY[0]/resolution_XY[0])*np.subtract(x_pixel, resolution_XY[0]/2.0)
        y = (pixelSize_XY[1]/resolution_XY[1])*np.subtract(resolution_XY[1]/2.0, y_pixel) # This line is diffetent than the one in Homography.py(KAMRAN)
        z = np.zeros(len(x_pixel))
        averageEyetoScreenDistance = sessionDict['analysisParameters']['averageEyetoScreenDistance'] 
        z = z + averageEyetoScreenDistance
        print('*** eyeOnScreenToMetricEyeOnScreen(): For Dk2, using [\'analysisParameters\'][\'averageEyetoScreenDistance\']***')
    
    else:
        raise AssertionError('Currently only works for the Oculus DK2')


    dFOut = []
    dFOut = pd.DataFrame({(dataOutColumnLabel, 'X'):x,
            (dataOutColumnLabel, 'Y'):y,
            (dataOutColumnLabel, 'Z'):z,})

    return dFOut

def metricEyeOnScreenToPixels(sessionDict,dFIn,dataOutColumnLabel):
    '''
    Convert metric locations on a screen at sessionDict['analysisParameters']['averageEyeToScreenDistance']
    to pixel coordinates
    '''

    if( sessionDict['analysisParameters']['hmd'].upper() == 'DK2' is False ):
            raise AssertionError('Currently only works for the Oculus DK2')

    x = dFIn['X']
    y = dFIn['Y']
        
    if( sessionDict['analysisParameters']['hmd'].upper() == 'DK2'):
         
        resolution_XY = sessionDict['analysisParameters']['hmdResolution']
        pixelSize_XY = sessionDict['analysisParameters']['hmdScreenSize']
        x_pixel = (resolution_XY[0]/pixelSize_XY[0])*np.add(x, pixelSize_XY[0]/2.0)
        y_pixel = (resolution_XY[1]/pixelSize_XY[1])*np.add(y, pixelSize_XY[1]/2.0)
        
    dFOut = []
    dFOut = pd.DataFrame({(dataOutColumnLabel, 'X'):x_pixel,
            (dataOutColumnLabel, 'Y'):y_pixel})

    return dFOut


def metricEyeOnScreenToEyeInHead(sessionDict,dFIn,dataOutColumnLabel):
	'''
	Really, this just converts metricEyeONScreen into a normalized eye-in-head vector.
	'''
	#tempDF = eyeOnScreenToMetricEyeOnScreen(dFIn,dataInColumnLabel,'filtMetricEyeOnScreen','DK2')

	# Normalize 

	normXYZ = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for 
	                                   XYZ in dFIn.values],dtype=np.float)

	dFOut = pd.DataFrame(normXYZ)

	dFOut = dFOut.rename(columns={0: (dataOutColumnLabel,'X'), 1:(dataOutColumnLabel,'Y'), 2: (dataOutColumnLabel,'Z')})


	return dFOut


def filterEyeOnScreen(sessionDict,dfIn,dataOutLabel):

	gazeFilter = sessionDict['analysisParameters']['gazeFilter']
	filterParameters = sessionDict['analysisParameters']['filterParameters']

	rawDF = sessionDict['raw']
	procDF = sessionDict['processed']
	dFOut = []

	dFOut = pd.DataFrame()

	if( gazeFilter == 'median' ):

	    assert len(filterParameters) == 1, "filterParameters of length %i, expected len of 1 for median filter." % len(filterParameters)

	    dFOut['X'] = dfIn['X'].rolling(filterParameters[0], min_periods = 0).median()
	    dFOut['Y'] = dfIn['Y'].rolling(filterParameters[0], min_periods = 0).median()

	elif( gazeFilter == 'average' ):

	    assert len(filterParameters) == 1, "filterParameters of length %i, expected len of 1 for average filter." % len(filterParameters)

	    dFOut['X'] = dfIn['X'].rolling(filterParameters[0], min_periods = 0).mean()
	    dFOut['Y'] = dfIn['Y'].rolling(filterParameters[0], min_periods = 0).mean()

	elif( gazeFilter == 'medianAndAverage' ):
	    
	    'First median, then average'
	    
	    assert len(filterParameters) == 2, "filterParameters of length %i, expected len of 2 for median + average filter." % len(filterParameters)
	    
	    dFOut['X'] = dfIn['X'].rolling(filterParameters[0], min_periods = 0).median()
	    dFOut['Y'] = dfIn['Y'].rolling(filterParameters[0], min_periods = 0).median()

	    dFOut['X'] = dfIn['X'].rolling(filterParameters[0], min_periods = 0).mean()
	    dFOut['Y'] = dfIn['Y'].rolling(filterParameters[0], min_periods = 0).mean()

	else:

	    raise AssertionError('Invalid filter type.  Types accepted:  {\"median\",\"average\"}')

	dFOut.columns = pd.MultiIndex.from_tuples([(dataOutLabel,'X'),(dataOutLabel,'Y')])

	return dFOut


def findColumn(dF,label):
    '''
    Searches through first level of a multi indexed dataframe
    for column labels that contain the 'label' passed in

    '''
    colIndices = []
    colNames = []

    for cIdx in range(len(dF.columns.levels[0])):

            if label in dF.columns.levels[0][cIdx]:

                colIndices.append(cIdx)
                colNames.append(dF.columns.levels[0][cIdx])
    
    return colNames

def columnExists(dF,label):
    '''
    Returns true if columns is found.
    False if not.

    '''
    colIndices = []
    colNames = []

    for cIdx in range(len(dF.columns.levels[0])):

            if label in dF.columns.levels[0][cIdx]:

                colIndices.append(cIdx)
                colNames.append(dF.columns.levels[0][cIdx])
    
    #return colIndices,colNames

    if( len(colIndices) > 0 ):
        return True
    else:
        return False

 

def loadSessionDict(analysisParameters,startFresh = False,loadProcessed = False):
    '''
    If startFresh is False, attempt to read in session dict from pickle.
    If pickle does not exist, or startFresh is True, 
        - read session dict from raw data file
        - create secondary dataframes
    '''

    filePath = analysisParameters['filePath']
    fileName = analysisParameters['fileName']
    expCfgName = analysisParameters['expCfgName']
    sysCfgName = analysisParameters['sysCfgName']

    # if loadProcessed, try to load the processed dataframe
    if( startFresh is False and loadProcessed is True):

        try:
            print('***loadSessionDict: Loading preprocessed data for ' + str(fileName) + ' ***')
            processedDict = pd.read_pickle(filePath + fileName + '-proc.pickle')
            return processedDict
        except:
            raise Warning('loadProcessedDict: Preprocessed data not available')

    # if loadProcessed failed or not true
    # ..if startFresh is True, load raw data and crunch on that
    # if startFresh is False, try to load the pickle
    if( startFresh is True):
        sessionDict = createSecondaryDataframes(filePath,fileName,expCfgName,sysCfgName)
    else:
        try:
            sessionDict = pd.read_pickle(filePath + fileName + '.pickle')
        except:
            sessionDict = createSecondaryDataframes(filePath,fileName,expCfgName,sysCfgName)
            pd.to_pickle(sessionDict, filePath + fileName + '.pickle')

    sessionDict['analysisParameters'] = analysisParameters

    pd.to_pickle(sessionDict, filePath + fileName + '.pickle')
    return sessionDict  



def createSecondaryDataframes(filePath,fileName,expCfgName,sysCfgName):
    '''
    Separates practice and calibration trials from main dataframe.
    Reads in exp and sys config.
    '''
    sessionDf = pp.readPerformDict(filePath + fileName + ".dict")

    [sessionDf, calibDf] = seperateCalib(sessionDf)

    expConfig =  createExpCfg(filePath + expCfgName)
    sysConfig =  createSysCfg(filePath + sysCfgName)
    practiceBlockIdx = [idx for idx, s in enumerate(expConfig['experiment']['blockList']) if s == 'practice']

    [sessionDf, practiceDf] =  seperatePractice(sessionDf,practiceBlockIdx)

    sessionDf = sessionDf.reset_index()
    sessionDf = sessionDf.rename(columns = {'index':'frameNumber'})

    #sessionDf.originalTrialNumber = sessionDf.trialNumber
    #blockNum = (sessionDf['blockNumber']+1)*200
    #newTrialNum = sessionDf['trialNumber'].values * sessionDf['blockNumber'].values

    ### New trial numbers
    # sessionDf.originalTrialNumber = sessionDf.trialNumber

    newTrialNum_tr = []
    
    gb = sessionDf.groupby(['trialNumber','blockNumber'])
    gbKeys = gb.groups.keys()
    count = 0
    
    for i in gbKeys:
        newTrialNum_tr.extend( np.repeat(count,[len(gb.get_group(i)['trialNumber'])]) )
        count = count+1
    
    sessionDf['trialNumber'] = np.array(newTrialNum_tr,dtype=int)

    ###
    
    trialInfoDf = expFun.initTrialInfo(sessionDf)
    
    procDataDf = expFun.initProcessedData(sessionDf)

    paddleDF   = expFun.calcPaddleBasis(sessionDf)
    procDataDf = pd.concat([paddleDF,procDataDf],axis=1)

    sessionDict = {'raw': sessionDf, 'processed': procDataDf, 'calibration': calibDf, 'practice': practiceDf, 
    'trialInfo': trialInfoDf,'expConfig': expConfig,'sysCfg': sysConfig}

    return sessionDict


### Save calibration frames in a separate dataframe

def excludeTrialType(sessionDict,typeString):

    sessionDictCopy = sessionDict.copy()

    gbTrialType = sessionDictCopy['trialInfo'].groupby('trialType')
    newDf = gbTrialType.get_group(typeString)
    sessionDictCopy['trialInfo'] = sessionDictCopy['trialInfo'].drop(gbTrialType.get_group(typeString).index)
    sessionDictCopy['trialInfo'] = sessionDictCopy['trialInfo'].reset_index()
    #sessionDict['trialInfo' + dictSublabel] = newDf.reset_index()


    sessionDictCopy['processed']['trialType'] = sessionDictCopy['raw']['trialType']
    gbProc = sessionDictCopy['processed'].groupby('trialType')
    newDf = gbProc.get_group(typeString)
    sessionDictCopy['processed'] = sessionDictCopy['processed'].drop(gbProc.get_group(typeString).index)
    sessionDictCopy['processed']=sessionDictCopy['processed'].reset_index()
    #sessionDict['proc' + dictSublabel] = newDf.reset_index()

    gbRaw = sessionDictCopy['raw'].groupby('trialType')
    newDf = gbRaw.get_group(typeString)
    sessionDictCopy['raw'] = sessionDictCopy['raw'].drop(gbRaw.get_group(typeString).index)
    sessionDictCopy['raw'] = sessionDictCopy['raw'].reset_index()
    #sessionDict['raw' + dictSublabel] = newDf.reset_index()
    
    return sessionDictCopy



# def seperateTrialType(sessionDf,trialTypeID):
    
#     trialsDf = pd.DataFrame()

#     for ttId in trialTypeID:

#         thisTrialTypeDF = sessionDf[sessionDf['trialType']==ttId]
#         trialsDf = pd.concat([trialsDf,thisTrialTypeDF],axis=0)
#         sessionDf = sessionDf.drop(thisTrialTypeDF.index)

#     return sessionDf, trialsDf

def seperateCalib(sessionDf):
    calibFrames = sessionDf['trialNumber']>999
    calibDf = sessionDf[calibFrames]
    sessionDf = sessionDf.drop(sessionDf[calibFrames].index)
    return sessionDf, calibDf

def seperatePractice(sessionDf,practiceBlockIdx):
    
    practiceDf = pd.DataFrame()
    
    for bIdx in practiceBlockIdx:
    	#print 'Seperating practice block: ' + str(bIdx)    
	thisPracticeBlockDF = sessionDf[sessionDf['blockNumber']==bIdx]
	practiceDf = pd.concat([practiceDf,thisPracticeBlockDF],axis=0)
	sessionDf = sessionDf.drop(thisPracticeBlockDF.index)
        
    return sessionDf, practiceDf


def findFirstZeroCrossing(vecIn):
    '''
    This will return the index of the first zero crossing of the input vector
    '''
    return np.where(np.diff(np.sign(vecIn)))[0][0]

def findFirst(dataVec,targetVal):
    '''
    Reports the first occurance of targetVal in dataVec.
    If no occurances found, returns None
    '''
    return next((fr for fr, eF in enumerate(dataVec) if eF == targetVal),False)



# def calculateStatsForGazeError(processedDataFrame, trialInfoDataFrame, offsetFrame):
#     trialStartIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'trialStart'].index.tolist()
#     ballOffIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballRenderOff'].index.tolist()
#     ballOnIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballRenderOn'].index.tolist()
#     ballOnPaddleIdx = processedDataFrame[processedDataFrame['eventFlag'] == 'ballOnPaddle'].index.tolist()
#     gazeError = processedDataFrame['gazeBallError_fr_XYZ'].values
#     angleList = processedDataFrame['gazeAngularError_fr_degree'].values
#     distanceAtBallOn = np.zeros((1,3))
#     distanceAfterBallOn = np.zeros((1,3))
#     distanceAtBallOff = np.zeros((1,3))
#     distanceAfterBallOff = np.zeros((1,3))
#     distanceAtBallOnPaddle = np.zeros((1,3))
#     errorAtBallOn = []
#     errorAtBallOff = []
#     errorAtBallOnPaddle = []
#     tempDict = {}
#     counter = 0
#     for i in range(max(processedDataFrame['trialNumber'].values)):
#         distanceAtBallOn = np.vstack((distanceAtBallOn, np.array(processedDataFrame.rotatedGazePoint_fr_XYZ.values[ballOnIdx[i]] - processedDataFrame.rotatedBallOnScreen_fr_XYZ.values[ballOnIdx[i]] )))
#         distanceAfterBallOn = np.vstack((distanceAfterBallOn, np.array(processedDataFrame.rotatedGazePoint_fr_XYZ.values[ballOnIdx[i] + offsetFrame] - processedDataFrame.rotatedBallOnScreen_fr_XYZ.values[ballOnIdx[i] + offsetFrame] )))
#         distanceAtBallOff= np.vstack((distanceAtBallOff, np.array(processedDataFrame.rotatedGazePoint_fr_XYZ.values[ballOffIdx[i]] - processedDataFrame.rotatedBallOnScreen_fr_XYZ.values[ballOffIdx[i]] )))
#         distanceAfterBallOff = np.vstack((distanceAfterBallOff, np.array(processedDataFrame.rotatedGazePoint_fr_XYZ.values[ballOffIdx[i] + offsetFrame] - processedDataFrame.rotatedBallOnScreen_fr_XYZ.values[ballOffIdx[i] + offsetFrame] )))
#         processedDataFrame
#         try :
#             tempDict['distanceAtBallOnPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'] = np.vstack((tempDict['distanceAtBallOnPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'], np.array([gazeError[ballOnIdx[i]], i])))
#         except KeyError:
#             tempDict['distanceAtBallOnPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'] = np.array([gazeError[ballOnIdx[i]], i])
#         try : 
#             tempDict['distanceAtBallOnPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'] = np.vstack((tempDict['distanceAtBallOnPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'], np.array([gazeError[ballOnIdx[i]], i])))
#         except KeyError:
#             tempDict['distanceAtBallOnPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'] = np.array([gazeError[ballOnIdx[i]], i])
#         try :
#             tempDict['distanceAfterBallOnPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'] = np.vstack((tempDict['distanceAfterBallOnPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'], np.array([gazeError[ballOnIdx[i] + offsetFrame], i])))
#         except KeyError:
#             tempDict['distanceAfterBallOnPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'] = np.array([gazeError[ballOnIdx[i] + offsetFrame], i])
#         try : 
#             tempDict['distanceAfterBallOnPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'] = np.vstack((tempDict['distanceAfterBallOnPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'], np.array([gazeError[ballOnIdx[i] + offsetFrame], i])))
#         except KeyError:
#             tempDict['distanceAfterBallOnPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'] = np.array([gazeError[ballOnIdx[i] + offsetFrame], i])
#         try :
#             tempDict['distanceAtBallOffPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'] = np.vstack((tempDict['distanceAtBallOffPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'], np.array([gazeError[ballOffIdx[i]], i])))
#         except KeyError:
#             tempDict['distanceAtBallOffPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'] = np.array([gazeError[ballOffIdx[i]], i])
#         try :
#             tempDict['distanceAtBallOffPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'] = np.vstack((tempDict['distanceAtBallOffPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'], np.array([gazeError[ballOffIdx[i]], i])))
#         except KeyError:
#             tempDict['distanceAtBallOffPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'] = np.array([gazeError[ballOffIdx[i]], i])
#         try :
#             tempDict['distanceAfterBallOffPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'] = np.vstack(( tempDict['distanceAfterBallOffPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'], np.array([gazeError[ballOffIdx[i] + offsetFrame], i])))
#         except KeyError:
#             tempDict['distanceAfterBallOffPreBD'+str(int(1000*trialInfoDataFrame['preBlankDur'].values[i]))+'ms'] = np.array([gazeError[ballOffIdx[i] + offsetFrame], i])
#         try :
#             tempDict['distanceAfterBallOffPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'] = np.vstack((tempDict['distanceAfterBallOffPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'], np.array([gazeError[ballOffIdx[i] + offsetFrame], i])))
#         except KeyError:
#             tempDict['distanceAfterBallOffPostBD'+str(int(1000*trialInfoDataFrame['postBlankDur'].values[i]))+'ms'] = np.array([gazeError[ballOffIdx[i] + offsetFrame], i])
#         if ( trialInfoDataFrame['ballCaughtQ'].values[i] == True ):
#             distanceAtBallOnPaddle = np.vstack((distanceAtBallOnPaddle, np.array(gazeError[ballOnPaddleIdx[counter]])))
#             errorAtBallOnPaddle.append(angleList[ballOnPaddleIdx[counter]])
#             counter = counter + 1
#         errorAtBallOn.append(angleList[ballOnIdx[i]])
#         errorAtBallOff.append(angleList[ballOffIdx[i]])
#     distanceAtBallOn = np.delete(distanceAtBallOn, 0,0)
#     distanceAfterBallOn = np.delete(distanceAfterBallOn, 0,0)
#     distanceAtBallOff = np.delete(distanceAtBallOff, 0,0)
#     distanceAfterBallOff = np.delete(distanceAfterBallOff, 0,0)
#     distanceAtBallOnPaddle = np.delete(distanceAtBallOnPaddle, 0,0)
#     return [tempDict, distanceAtBallOn, distanceAtBallOff, distanceAfterBallOn, distanceAfterBallOff, distanceAtBallOnPaddle]



def createExpCfg(expCfgPathAndName):

    """
    Parses and validates a config obj
    Variables read in are stored in configObj

    """

    print("Loading experiment config file: " + expCfgPathAndName)
    
    from os import path
    filePath = path.dirname(path.abspath(expCfgPathAndName))

    # This is where the parser is called.
    expCfg = ConfigObj(expCfgPathAndName, configspec=filePath + '/expCfgSpec.ini', raise_errors = True, file_error = True)

    validator = Validator()
    expCfgOK = expCfg.validate(validator)
    if expCfgOK == True:
        print("Experiment config file parsed correctly")
    else:
        print('Experiment config file validation failed!')
        res = expCfg.validate(validator, preserve_errors=True)
        for entry in flatten_errors(expCfg, res):
        # 1each entry is a tuple
            section_list, key, error = entry
            if key is not None:
                section_list.append(key)
            else:
                section_list.append('[missing section]')
            section_string = ', '.join(section_list)
            if error == False:
                error = 'Missing value or section.'
            print(section_string, ' = ', error)
        sys.exit(1)
    if expCfg.has_key('_LOAD_'):
        for ld in expCfg['_LOAD_']['loadList']:
            print('Loading: ' + ld + ' as ' + expCfg['_LOAD_'][ld]['cfgFile'])
            curCfg = ConfigObj(expCfg['_LOAD_'][ld]['cfgFile'], configspec = expCfg['_LOAD_'][ld]['cfgSpec'], raise_errors = True, file_error = True)
            validator = Validator()
            expCfgOK = curCfg.validate(validator)
            if expCfgOK == True:
                print("Experiment config file parsed correctly")
            else:
                print('Experiment config file validation failed!')
                res = curCfg.validate(validator, preserve_errors=True)
                for entry in flatten_errors(curCfg, res):
                # each entry is a tuple
                    section_list, key, error = entry
                    if key is not None:
                        section_list.append(key)
                    else:
                        section_list.append('[missing section]')
                    section_string = ', '.join(section_list)
                    if error == False:
                        error = 'Missing value or section.'
                    print(section_string, ' = ', error)
                sys.exit(1)
            expCfg.merge(curCfg)

    return expCfg


def createSysCfg(sysCfgPathAndName):
    """
    Set up the system config section (sysCfg)
    """

    # Get machine name
    #sysCfgName = platform.node()+".cfg"
    
    
    

    print("Loading system config file: " + sysCfgPathAndName)

    # Parse system config file
    from os import path
    filePath = path.dirname(path.abspath(sysCfgPathAndName))
    
    sysCfg = ConfigObj(sysCfgPathAndName , configspec=filePath + '/sysCfgSpec.ini', raise_errors = True)

    validator = Validator()
    sysCfgOK = sysCfg.validate(validator)

    if sysCfgOK == True:
        print("System config file parsed correctly")
    else:
        print('System config file validation failed!')
        res = sysCfg.validate(validator, preserve_errors=True)
        for entry in flatten_errors(sysCfg, res):
        # each entry is a tuple
            section_list, key, error = entry
            if key is not None:
                section_list.append(key)
            else:
                section_list.append('[missing section]')
            section_string = ', '.join(section_list)
            if error == False:
                error = 'Missing value or section.'
            print(section_string, ' = ', error)
        sys.exit(1)
    return sysCfg

# def plotMyData_Scatter3D(data, label, color, marker, axisLabels):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data[:,0], data[:,2], data[:,1], label = label, c = color, marker = marker)
#     ax.set_xlabel(axisLabels[0])
#     ax.set_ylabel(axisLabels[1])
#     ax.set_zlabel(axisLabels[2])
#     legend = plt.legend(loc=[1.,0.4], shadow=True, fontsize='small')# 'upper center'
#     plt.show()

# def plotMyData_2D(data, title, label, color, marker, axisLabels, dataRange = None):
#     fig1 = plt.figure()
#     plt.plot(data[:,0], data[:,1], label = label, c = color, marker = marker)
#     plt.xlabel(axisLabels[0])
#     plt.ylabel(axisLabels[1])
#     plt.title(title)
#     plt.grid(True)
#     legend = plt.legend(loc=[1.,0.4], shadow=True, fontsize='small')# 'upper center'
#     plt.show()

def dotproduct( v1, v2):
    r = sum((a*b) for a, b in zip(v1, v2))
    return r

def length(v):
    return np.sqrt(dotproduct(v, v))

def vectorAngle( v1, v2):
    r = (180.0/np.pi)*np.arccos((dotproduct(v1, v2)) / (length(v1) * length(v2)))#np.arccos((np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))#
    return r

# def calculateGazeBallOnScreen(rawDataFrame, processedDataFrame, trialID = None):
#     minRange = 0
#     if (trialID is None):
#         rawTempDataFrame = rawDataFrame
#         processedTempDataFrame = processedDataFrame
#         maxRange = len(rawTempDataFrame)
#     else:
#         rawGBTrial = rawDataFrame.groupby(['trialNumber'])
#         processedGBTrial = processedDataFrame.groupby(['trialNumber'])
#         startFrame = 0
#         maxRange = len(rawGBTrial.get_group(trialID))
#         numberOfFrames = len(rawGBTrial.get_group(trialID))
#         rawTempDataFrame = rawGBTrial.get_group(trialID)
#         processedTempDataFrame = processedGBTrial.get_group(trialID)
#     print 'Trial ID = ', trialID, 'FrameCount = ', maxRange
#     cycEyePosition_X = rawTempDataFrame['viewPos']['X'].values
#     cycEyePosition_Y = rawTempDataFrame['viewPos']['Y'].values
#     cycEyePosition_Z = rawTempDataFrame['viewPos']['Z'].values
#     headQuat_X = rawTempDataFrame['viewQuat']['X'].values
#     headQuat_Y = rawTempDataFrame['viewQuat']['Y'].values
#     headQuat_Z = rawTempDataFrame['viewQuat']['Z'].values
#     headQuat_W = rawTempDataFrame['viewQuat']['W'].values
#     ballPosition_X = rawTempDataFrame['ballPos']['X'].values
#     ballPosition_Y = rawTempDataFrame['ballPos']['Y'].values
#     ballPosition_Z = rawTempDataFrame['ballPos']['Z'].values
#     cycPOR_X = processedTempDataFrame['avgFilt3_cycEyeOnScreen']['X'].values
#     cycPOR_Y = processedTempDataFrame['avgFilt3_cycEyeOnScreen']['Y'].values
#     metricCycPOR_X = []
#     metricCycPOR_Y = []
#     metricCycPOR_Z = np.zeros(maxRange)
#     constantValue = 1.0
#     metricCycPOR_Z = metricCycPOR_Z + constantValue 
#     cameraCenterPosition = np.array([0.0,0.0,0.0]) # in HCS
#     planeNormal = np.array([0.0,0.0,1.0]) # in Both HCS and WCS
#     eyetoScreenDistance = 0.0725 # This assumes that the Eye-Screen Distance is always constant
#     screenCenterPosition = np.array([0.0,0.0,eyetoScreenDistance])
#     [metricCycPOR_X, metricCycPOR_Y] = pixelsToMetric(cycPOR_X, cycPOR_Y)
#     viewRotMat_fr_mRow_mCol = [quat2transform(q) for q in rawTempDataFrame.viewQuat.values]
#     metricCycPOR_fr_XYZ = np.array([metricCycPOR_X, metricCycPOR_Y, metricCycPOR_Z], dtype = float).T
#     calibratedPOR_fr_XYZ = np.zeros((maxRange,3))
#     H = processedDataFrame['linearHomography'].values[0].reshape((3,3))
#     for i in range(maxRange):
#         calibratedPOR_fr_XYZ[i,:] = np.dot(H, metricCycPOR_fr_XYZ[i,:])
#     calibratedPOR_fr_XYZ[:,2] = eyetoScreenDistance
#     gazePoint_fr_XYZ = np.array([ np.dot(viewRotMat_fr_mRow_mCol[fr], calibratedPOR_fr_XYZ[fr].T) 
#          for fr in range(len(calibratedPOR_fr_XYZ))])
#     ballOnScreen_fr_XYZ = np.empty([1, 3], dtype = float)
#     for i in range(maxRange):
#         lineNormal = [np.subtract(rawTempDataFrame['ballPos'][i:i+1].values, rawTempDataFrame['viewPos'][i:i+1].values)]
#         rotatedNormalPlane = np.array( np.dot(viewRotMat_fr_mRow_mCol[i], planeNormal))
#         rotatedScreenCenterPosition = np.dot(viewRotMat_fr_mRow_mCol[i], screenCenterPosition)
#         tempPos = findLinePlaneIntersection( cameraCenterPosition, lineNormal[0], rotatedNormalPlane, rotatedScreenCenterPosition) 
#         ballOnScreen_fr_XYZ = np.vstack((ballOnScreen_fr_XYZ, tempPos[0].T))
#     ballOnScreen_fr_XYZ = np.delete(ballOnScreen_fr_XYZ, 0, 0)
#     rotatedBallOnScreen_fr_XYZ = np.array([np.dot(np.linalg.inv(viewRotMat_fr_mRow_mCol[fr]), ballOnScreen_fr_XYZ[fr].T,) 
#                                     for fr in range(len(ballOnScreen_fr_XYZ))])
#     rotatedGazePoint_fr_XYZ = np.array([np.dot(np.linalg.inv(viewRotMat_fr_mRow_mCol[fr]), gazePoint_fr_XYZ[fr].T) 
#                                     for fr in range(len(gazePoint_fr_XYZ))])
#     print ballOnScreen_fr_XYZ.shape
#     return [gazePoint_fr_XYZ, rotatedGazePoint_fr_XYZ, ballOnScreen_fr_XYZ, rotatedBallOnScreen_fr_XYZ]

def createLine( point0, point1 ):
    unitVector = np.subtract(point0, point1)/length(np.subtract(point0, point1))
    return unitVector

def findLinePlaneIntersection(point_0, line, planeNormal, point_1):
    s = point_1 - point_0
    numerator = dotproduct(s, planeNormal)
    denumerator = np.inner(line, planeNormal)
    if (denumerator == 0):
        print('No Intersection')
        return None
    d = np.divide(numerator, denumerator)
    intersectionPoint = np.multiply(d, line) + point_0
    return intersectionPoint

def findResidualError(projectedPoints, referrencePoints):
    e2 = np.zeros((projectedPoints.shape[0],2))
    for i in range(projectedPoints.shape[0]):
        temp = np.subtract(projectedPoints[i], referrencePoints[i])
        e2[i,:] = np.power(temp[0:2], 2)
    return [np.sqrt(sum(sum(e2[:])))]


def quat2transform(q):
    """
    Transform a unit quaternion into its corresponding rotation matrix (to
    be applied on the right side).
    :returns: transform matrix
    :rtype: numpy array
    """
    x, y, z, w = q
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    wz2 = 2 * w * z
    zx2 = 2 * z * x
    wy2 = 2 * w * y
    yz2 = 2 * y * z
    wx2 = 2 * w * x
    rmat = np.empty((3, 3), float)
    rmat[0,0] = 1. - yy2 - zz2
    rmat[0,1] = xy2 - wz2
    rmat[0,2] = zx2 + wy2
    rmat[1,0] = xy2 + wz2
    rmat[1,1] = 1. - xx2 - zz2
    rmat[1,2] = yz2 - wx2
    rmat[2,0] = zx2 - wy2
    rmat[2,1] = yz2 + wx2
    rmat[2,2] = 1. - xx2 - yy2
    return rmat

def print_source(function):
    
    """For use inside an IPython notebook: given a module and a function, print the source code."""
    from inspect import getsource,getmodule
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    from IPython.core.display import HTML
    
    internal_module = getmodule(function)

    return HTML(highlight(getsource(function), PythonLexer(), HtmlFormatter(full=True)))


