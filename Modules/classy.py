from __future__ import division

import sys
sys.path.append("Modules/")
import os

import pandas as pd

import numpy as np
from scipy import signal as sig

import performFun as pF
import catchE1Funs as expFun
import recalibration as rc

import bokeh.plotting as bkP
import bokeh.models as bkM
from bokeh.palettes import Spectral6
from bokeh.embed import file_html
from bokeh.resources import CDN

from plottingFuns import *

def applyClassifier(sessionDict,model,winRanges):
    
    def savGolFilter(sessionDict,polyorder = 2):
    
        sessionDict['processed'][('cycGIWAngles','azimuth')]   = sessionDict['processed'][('cycGIWDir','X')].apply(lambda x: np.rad2deg(np.arccos(x)))
        sessionDict['processed'][('cycGIWAngles','elevation')] = sessionDict['processed'][('cycGIWDir','Y')].apply(lambda y: np.rad2deg(np.arccos(y)))

        x = sessionDict['processed'][('cycGIWAngles')].values
        fixDurFrames =  int(np.floor(0.1 / sessionDict['analysisParameters']['fps']))
        delta = sessionDict['analysisParameters']['fps']
        polyorder = 2

        #print 'Filter width: %u frames' % fixDurFrames
        from scipy.signal import savgol_filter
        vel = savgol_filter(x, fixDurFrames, polyorder, deriv=1, delta=delta, axis=0, mode='interp', cval=0.0)

        sessionDict['processed'][('cycSGVel','azimuth')] = vel[:,0]
        sessionDict['processed'][('cycSGVel','elevation')] = vel[:,1]
        sessionDict['processed'][('cycSGVel','2D')] = np.sqrt(np.sum(np.power(vel,2),1))

        return sessionDict


    def classifySegment(vel_fr,model,verbose = False):

        frameRateSecs = (1/75.0) #sessionDict['analysisParameters']['fps']

        duration = frameRateSecs * len(vel_fr)
        amplitude = sum(np.multiply(win,frameRateSecs))

        #     amplitude =  vectorAngle(sessionDict['processed']['cycGIWDir'].loc[winLoc[0]],
        #                              sessionDict['processed']['cycGIWDir'].loc[winLoc[1]])

        vel = np.max(vel_fr)

        group = int(model.predict([duration,amplitude,vel]))-1

        catNames = ('saccade','fixation','pursuit')

        probs = np.array(model.predict_proba([duration,amplitude,vel])[0])

        if( verbose):

            print 'Features:'
            print '  Duration: %1.2f' %(duration)
            print '  Amplitude: %1.2f' %amplitude
            print '  Velocity: %1.2f \n' %vel

            print 'Probabilities:'
            print '  Saccade: %0.2f' %probs[0]
            print '  Fixation: %0.2f' %probs[1]
            print '  Pursuit: %0.2f \n' %probs[2]

            print '** Identified as ' + catNames[group] + ' ** \n'

        return np.array(np.hstack([group,probs]))


    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def createGazeEventsDf(sessionDict,gazeEvent_fr, minDurFr = False):

        catNames = ('saccade','fixation','pursuit')
        time_fr = np.cumsum(sessionDict['processed']['smiDeltaT'])

        gazeEventsDf = pd.DataFrame()

        for idx, eventStr in enumerate(catNames):

            gazeDf = pd.Series(gazeEvent_fr)
            starts = gazeDf.index[(gazeDf.values == idx) & np.roll((gazeDf.values != idx),1)]
            ends = gazeDf.index[(gazeDf.values == idx) & np.roll((gazeDf.values != idx),-1)]

            if(starts[0]>ends[0]):
                starts = starts[:-1]
                ends = ends[1:]

            if( minDurFr ):
                # filter those which are adequately apart
                event_onOff = np.array([[i, j] for i, j in zip(starts, ends) if j > i + minDurFr ])
                starts = event_onOff[:,0]
                ends = event_onOff[:,1]

            trialNum = np.array(sessionDict['processed']['trialNumber'].values[starts])

            eventTime_on = np.squeeze(time_fr[starts])
            eventTime_off = np.squeeze(time_fr[ends])

            thisEventDf = pd.DataFrame({ 
                (eventStr,'timeStart'): np.array(eventTime_on),
                (eventStr,'timeEnd'): np.array(eventTime_off),
                (eventStr,'frameStart'): np.array(starts),
                (eventStr,'frameEnd'): np.array(ends),
                (eventStr,'trialNumber'): np.array(trialNum) })

            gazeEventsDf = pd.concat([gazeEventsDf,thisEventDf],axis=1)

        return gazeEventsDf
    
    # Apply savgol filter
    sessionDict = savGolFilter(sessionDict,polyorder = 2)
    
    vel_fr = np.array(sessionDict['processed'][('cycSGVel','2D')].values,dtype=np.float)

    group_win_fr = np.zeros([len(winRanges),len(vel_fr)])
    prob_win_fr_type = np.zeros([len(winRanges),len(vel_fr),3])

    ##  Apply to data
    for idx, winSize in enumerate(winRanges):

        groupAndProbs_fr_gfsp = np.array([classifySegment(win,model) for win in rolling_window(vel_fr,winSize)],dtype=np.float)

        group_fr = groupAndProbs_fr_gfsp[:,0]
        probs_fr = groupAndProbs_fr_gfsp[:,1:]

        zeros = [0] * (len(vel_fr)-len(group_fr))
        halfLen = int(np.floor(len(zeros)))
        group_fr = np.hstack([zeros[:halfLen],group_fr,zeros[halfLen:]])

        zeros = [[0,0,0]] * (len(vel_fr)-len(probs_fr))
        halfLen = int(np.floor(np.shape(zeros)[0]/2))
        probs_fr = np.vstack([zeros[:halfLen][:],probs_fr,zeros[halfLen:][:]])
        #probs_fr = np.vstack([probs_fr,zeros])

        group_win_fr[idx,:] = group_fr
        prob_win_fr_type[idx,:,:] = probs_fr
    
    # On each frame, take the max probability for each window
    maxProb_fr_type = np.max(prob_win_fr_type,axis=0)
    
    # On each frame, take the most probable type
    gazeEvent_fr = np.argmax(maxProb_fr_type, axis=1)
        
    return (createGazeEventsDf(sessionDict,gazeEvent_fr), gazeEvent_fr)
