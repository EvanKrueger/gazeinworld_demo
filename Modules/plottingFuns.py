from __future__ import division

import sys
sys.path.append("Modules/")
import os

import pandas as pd

import numpy as np
from scipy import signal as sig

from performFun import *
from catchE1Funs import *

import bokeh.plotting as bkP
import bokeh.models as bkM
from bokeh.palettes import Spectral6
from bokeh.embed import file_html
from bokeh.resources import CDN

##################################################################
##################################################################
#### Plotting functions

# def plotFilteredGazeData(rawDataFrame, processedDataFrame):
    
#     from bokeh.io import vplot
#     from bokeh.plotting import figure, output_file, show
    
#     frameRange = range(20000,21000)
    
#     x0 = rawDataFrame['cycEyeOnScreen']['X'][frameRange].values
#     x1 = processedDataFrame['medFilt3_cycEyeOnScreen']['X'][frameRange].values
#     x2 = processedDataFrame['avgFilt3_cycEyeOnScreen']['X'][frameRange].values
#     x3 = processedDataFrame['avgFilt5_cycEyeOnScreen']['X'][frameRange].values
#     y0 = rawDataFrame['cycEyeOnScreen']['Y'][frameRange].values
#     y1 = processedDataFrame['medFilt3_cycEyeOnScreen']['Y'][frameRange].values
#     y2 = processedDataFrame['avgFilt3_cycEyeOnScreen']['Y'][frameRange].values
#     y3 = processedDataFrame['avgFilt5_cycEyeOnScreen']['Y'][frameRange].values
#     T = range(len(x1))
#     dataColor = ['red','green', 'blue', 'yellow']
#     dataLegend = ["Raw", "Med3", "Med5", "Med7"]
#     p1 = figure(plot_width=500, plot_height=500)
#     p1.multi_line(xs=[T,T,T,T], ys=[x0, x1, x2, x3],
#                  color=dataColor)
#     p2 = figure(plot_width=500, plot_height=500)
#     p2.multi_line(xs=[T,T,T,T], ys=[y0, y1, y2, y3],
#                  color=dataColor)
#     p = vplot(p1,p2)
#     show(p)

def viewGazeEvents(sessionDict,gazeEventsDf,trialNum,legendLabels=None,yLims = [0,300],
               events_fr=None,trialsStarts_tr=None,plotHeight=500,plotWidth = 1000):
    
    ''' 
    Creates a time-series plot of gaze data with Bokeh.
    dataFrame = a dataframe with field ['frameTime'], ['eventFlag'], and ['trialNumber'] 
    yLabel = A label for the Y axis. 
    yDataList = A list of vectors to be plotted on the Y axis as a line
    legendLabels = A list of names for data plotted on the Y axis
    yMax = Height of Y axidafdataFrames
    markEvents= Show vertical lines with labels at events in dataFrame['eventFlag']
    markTrials=Show vertical lines with labels at start of each trial
    '''

    ######################################################################################################
    ## Gather data
    
    gbRaw = sessionDict['raw'].groupby(['trialNumber']).get_group(trialNum)
    gbProc = sessionDict['processed'].groupby(['trialNumber']).get_group(trialNum)
    #frametime_fr = sessionDict['raw']['frameTime'].values
    
    trialFrames = range(gbProc.index[0],gbProc.index[-1])
    
    from bokeh.palettes import Spectral6
    
    if( legendLabels and isinstance(legendLabels, list) is False):
        raise TypeError('legendLabels should be a list of lists.  Try [yLabelList].')
        
    #### Setup figure

    yRange = bkM.Range1d(yLims[0],yLims[1])

    p = bkP.figure(plot_width =plotWidth, plot_height=plotHeight,tools="xpan,reset,save,xwheel_zoom,resize,tap",
                   y_range=[0,500], 
                   x_range=[np.min(gbRaw['frameTime']),np.max(gbRaw['frameTime'])],
                   x_axis_label='time (s)', y_axis_label='GIW velocity')

    p.ygrid.grid_line_dash = [6, 4]

    #p.x_range = bkM.Range1d(dataFrame['frameTime'].values[0], dataFrame['frameTime'].values[0]+2)
    #p.x_range = bkM.Range1d(np.min(frametime_fr), np.min(frametime_fr)+2)
    p.y_range = yRange
    
    ######################################################################################################
    ## Plot gaze velocity(s)

    eventTypes = ('saccade','fixation','pursuit')
    eventColors = ['green','red','gold']
    
    p.line(gbRaw['frameTime'],gbProc[('cycSGVel','2D')],line_width=4, alpha=.7,color='gray',line_dash=[3,3])

    for eventIdx, eventstr in enumerate(eventTypes):
        
        if( columnExists(gazeEventsDf,eventstr) ):
            
            gbGazeEvent = gazeEventsDf[eventstr].dropna().groupby('trialNumber')
            
            # Handle nan values
            eventStartFr = np.array(gbGazeEvent.get_group(trialNum)['frameStart'].values,dtype=np.int)
            eventEndFr = np.array(gbGazeEvent.get_group(trialNum)['frameEnd'].values,dtype=np.int)

            event_onOff = zip(eventStartFr,eventEndFr)

            #frametime_fr = sessionDict['raw']['frameTime'].values
            eyeframetime_fr = sessionDict['raw']['frameTime'].values
            #eyeframetime_fr = sessionDict['raw']['frameTime'].values
            frameTime_event_frNum = [eyeframetime_fr[range(onOff[0],onOff[1])] for onOff in event_onOff]

            sgVel_fr = sessionDict['processed'][('cycSGVel','2D')]
            velData_event_frNum = [sgVel_fr.values[range(onOff[0],onOff[1])] for onOff in event_onOff]

            p.multi_line(frameTime_event_frNum,velData_event_frNum,line_width=5,color=eventColors[eventIdx])  
        
    ######################################################################################################
    ### Annotate events
    
    showHighBox = False
    
    #if( type(events_fr) is pd.Series ):
    if( events_fr.any() ):
        
        showHighBox = True
        frametime_fr = gbRaw['frameTime'].values
        X = frametime_fr[np.where(events_fr>2)]
        Y = [yLims[1]*.9]*len(X)
        text = [str(event) for event in events_fr[np.where(events_fr>2)]]

        p.text(X,Y,text,text_font_size='8pt',text_font='futura')
        
        ### Vertical lines at events
        X = [ [X,X] for X in frametime_fr[np.where(events_fr>2)]]
        p.multi_line(X,[[yLims[0],yLims[1]*.9]] * len(X),color='red',alpha=0.6,line_width=2)

    if( trialsStarts_tr):
        
        showHighBox = True
        
        ### Annotate trial markers
        X = [trialStart+0.02 for trialStart in trialsStarts_tr]
        Y = [yLims[1]*.95] * len(trialsStarts_tr)
        text = [  'Tr ' + str(trIdx) for trIdx,trialStart in enumerate(trialsStarts_tr)]
        p.text(X,Y,text,text_font_size='10pt',text_font='futura',text_color='red')

        ### Vertical lines at trial starts
        X = [  [trialStart]*2 for trialStart in trialsStarts_tr]
        Y = [[yLims[0],yLims[1]*.9]] * len(trialsStarts_tr)
        p.multi_line(X,Y,color='red',alpha=0.6,line_width=4)

    if( showHighBox ):
        
        high_box = bkM.BoxAnnotation(plot=p, bottom = yLims[1]*.9, 
                                     top=yLims[1], fill_alpha=0.7, fill_color='green', level='underlay')
        p.renderers.extend([high_box])
        
    return p




def timeSeries( frametime_fr=None, yDataList=None,yLabel=None,legendLabels=None,yLims = [0,300],
               events_fr=None,trialsStarts_tr=None,plotHeight=500,plotWidth = 1000):
    ''' 
    Creates a time-series plot of gaze data with Bokeh.
    dataFrame = a dataframe with field ['frameTime'], ['eventFlag'], and ['trialNumber'] 
    yLabel = A label for the Y axis. 
    yDataList = A list of vectors to be plotted on the Y axis as a line
    legendLabels = A list of names for data plotted on the Y axis
    yMax = Height of Y axidafdataFrames
    markEvents= Show vertical lines with labels at events in dataFrame['eventFlag']
    markTrials=Show vertical lines with labels at start of each trial
    '''
    from bokeh.palettes import Spectral6
    
    if( isinstance(yDataList, list) is False):
        raise TypeError('yDataList should be a list of lists.  Try [yData].')
    
    if( legendLabels and isinstance(legendLabels, list) is False):
        raise TypeError('legendLabels should be a list of lists.  Try [yLabelList].')
        
    #### Setup figure

    yRange = bkM.Range1d(yLims[0],yLims[1])

    p = bkP.figure(plot_width =plotWidth, plot_height=plotHeight,tools="xpan,reset,save,xwheel_zoom,resize,tap",
                   y_range=[0,500], 
                   x_range=[np.min(frametime_fr),np.max(frametime_fr)],
                   x_axis_label='time (s)', y_axis_label=yLabel)

    p.ygrid.grid_line_dash = [6, 4]

    #p.x_range = bkM.Range1d(dataFrame['frameTime'].values[0], dataFrame['frameTime'].values[0]+2)
    p.x_range = bkM.Range1d(np.min(frametime_fr), np.min(frametime_fr)+2)
    p.y_range = yRange

    ### Vertical lines at trial starts
    if( trialsStarts_tr ):
        
        X = [[startIdx]*2 for startIdx in trialsStarts_tr] #dataFrame.groupby('trialNumber')]
        Y = [[yLims[0],yLims[1]]] * len(trialsStarts_tr)
        p.multi_line(X,Y,color='red',alpha=0.6,line_width=2)

    ######################################################################################################
    ## Plot gaze velocity(s)
            
    for yIdx, yData in enumerate(yDataList):
        
        if(legendLabels and len(legendLabels) >= yIdx):
            p.line(frametime_fr,yData,line_width=3, alpha=.7,color=Spectral6[yIdx],legend=legendLabels[yIdx]) 
        else:
            p.line(frametime_fr,yData,line_width=3, alpha=.7,color=Spectral6[yIdx]) 

        #p.line(dataFrame['frameTime'].values,dataFrame['cycGIWVelocityRAW'].values,line_width=3,alpha=.7,color='green',legend="raw")
            

    ######################################################################################################
    ### Annotate events
    
    showHighBox = False
    
    #if( type(events_fr) is pd.Series ):
    if( events_fr.any() ):
        
        showHighBox = True
        X = frametime_fr[np.where(events_fr>2)]+.01
        Y = [yLims[1]*.9]*len(X)
        text = [str(event) for event in events_fr[np.where(events_fr>2)]]

        p.text(X,Y,text,text_font_size='8pt',text_font='futura')
        
        ### Vertical lines at events
        X = [ [X,X] for X in frametime_fr[np.where(events_fr>2)]]
        p.multi_line(X,[[yLims[0],yLims[1]*.9]] * len(X),color='red',alpha=0.6,line_width=2)

    if( trialsStarts_tr):
        
        showHighBox = True
        
        ### Annotate trial markers
        X = [trialStart+0.02 for trialStart in trialsStarts_tr]
        Y = [yLims[1]*.95] * len(trialsStarts_tr)
        text = [  'Tr ' + str(trIdx) for trIdx,trialStart in enumerate(trialsStarts_tr)]
        p.text(X,Y,text,text_font_size='10pt',text_font='futura',text_color='red')

        ### Vertical lines at trial starts
        X = [  [trialStart]*2 for trialStart in trialsStarts_tr]
        Y = [[yLims[0],yLims[1]*.9]] * len(trialsStarts_tr)
        p.multi_line(X,Y,color='red',alpha=0.6,line_width=4)

    if( showHighBox ):
        
        high_box = bkM.BoxAnnotation(plot=p, bottom = yLims[1]*.9, 
                                     top=yLims[1], fill_alpha=0.7, fill_color='green', level='underlay')
        p.renderers.extend([high_box])
        
    return p

    
def plotCalibrationError(sessionDict,gazeDataDf,calibPointDf,calibSessionIdx=0,outFilepath=False):

    #frameCount = range(len(sessionDict['calibration']['cycEyeOnScreen']))

    numberOfCalibrationSession =  (max(sessionDict['calibration'].trialNumber.values)%1000)//100

    trialOffset = 1000;
    frameOffset = 0

    numberOfCalibrationPoints = 27
    numberOfCalibrationSession = 0

    # Calculate a homography for each session
    i = calibSessionIdx

    startTrialNumber = trialOffset + 100*i 
    endTrialNumber = trialOffset + 100*i + numberOfCalibrationPoints - 1

    gbCalibSession = sessionDict['calibration'].groupby(['trialNumber'])
    firstCalibrationSession = gbCalibSession.get_group(startTrialNumber)
    lastCalibrationSession = gbCalibSession.get_group(endTrialNumber)

    numberOfCalibrationFrames = max(lastCalibrationSession.index) - min(firstCalibrationSession.index) + 1
    dataRange = range(frameOffset, frameOffset + numberOfCalibrationFrames)

    #####
    p = bkP.figure(plot_width=500, plot_height=500)

    xs = gazeDataDf['X'].values[dataRange]
    ys = gazeDataDf['Y'].values[dataRange]
    p.scatter(xs, ys, marker="cross", size=10,line_color="orange", fill_color="orange", alpha=0.5,name='POR')
      
    xs = calibPointDf['X'].values[dataRange]
    ys = calibPointDf['Y'].values[dataRange]
    p.scatter(xs, ys, marker="circle", size=10,line_color="green", fill_color="green", alpha=0.5,name='POR')

    return p

def plotGroupInteraction_pre_post(betweenSubStatsDf,xVar,yVar, lineVar,outFilePath=False):

    xs = np.unique(betweenSubStatsDf[xVar].round(decimals =2))
    lineNames = np.array(np.unique(betweenSubStatsDf[lineVar]),dtype=np.str)

    p = bkP.figure(width=800, height=400,tools = "pan,reset,resize,hover,wheel_zoom,save")

    off = [-.01, 0, .01]
    clist = ['blue','orange','green']

    j = 0
    gbLine = betweenSubStatsDf.groupby(lineVar)
    for grIdx,gr in gbLine:
        ys = gr[yVar]['mean']
        yerr = gr[yVar]['<lambda>']    
        p.line( xs+off[j], ys ,line_width=3,color=clist[j],legend = lineVar + ' ' +  lineNames[j])
        errorbar(p,xs+off[j], ys,yerr=yerr, color=clist[j], point_kwargs={'line_width':3,'size': 10},error_kwargs={'line_width': 3})
        j = j+1

    p.xaxis.axis_label = xVar
    p.yaxis.axis_label = ''.join(yVar)

    if( outFilePath ):

        bkP.output_file(outFilePath + yVar + '_' + xVar +'_' + lineVar + '.html') 
        bkP.save(p)

    else:

        return p



def plotCatchRateInteraction(sessionDict, outFilePath = False):

    trialInfoDF = sessionDict['trialInfo']
    yVar = 'ballCaughtQ'
    xVar = 'preBlankDur'
    lineVar = 'postBlankDur'

    lineNames = np.array(np.unique(trialInfoDF[lineVar].round(decimals =2)),dtype=np.str)
    groupedByPre =  trialInfoDF.groupby([lineVar,xVar])

    mean_cond = []
    std_cond  = []

    for gNum, gr in groupedByPre:
    
        mean_cond.append(100 * np.sum(gr[yVar]) / len(gr[yVar]) )
    
    numV1 = len(np.unique(np.array(groupedByPre.groups.keys())[:,0]))
    numV2 = len(np.unique(np.array(groupedByPre.groups.keys())[:,1]))
    mean_pre_post = np.reshape(mean_cond,[numV1,numV2])

    xs = [np.unique(trialInfoDF[xVar].round(decimals =2))]*3
    ys = [np.array(xyz,dtype=np.float) for xyz in mean_pre_post]

    p = bkP.figure(width=800, height=400,tools = "pan,reset,resize,hover")

    off = [-.01, 0, .01]
    clist = ['blue','orange','green']

    for j in range(3):
        p.line( xs[j]+off[j], ys[j] ,line_width=3,color=clist[j])#,legend = lineVar + ' ' +  lineNames[j])
        p.circle(xs[j]+off[j], ys[j], fill_color=clist[j], size=8)

    p.xaxis.axis_label = xVar
    p.yaxis.axis_label = ''.join(yVar)

    p.y_range = bkM.Range1d(0,100)

    if( outFilePath ):
        bkP.output_file(outFilePath + 'plotCatchRateInteraction.html') 
        bkP.save(p)
    else:
        return p

def plotCatchingErrorScatter(sessionDict,scatterVar,outFilePath=False):
    '''
    Plots the X and Y data of scatterVar 
    Assumes data is already normalized to the XY position of the paddle 
    '''
    rawDF = sessionDict['raw']
    procDF = sessionDict['processed']
    trialInfoDF = sessionDict['trialInfo']

    caughtGroups = trialInfoDF.groupby('ballCaughtQ')
    missedTrials = caughtGroups.get_group(False)
    caughtTrials = caughtGroups.get_group(True)

    paddleRadius =   getPaddleRadius(sessionDict)
    ballRadius =  0.09/2.0 #   .getBallRadius(sessionDict)

    TOOLS = "pan,wheel_zoom,hover,save"

    p = bkP.figure(plot_width=400, plot_height=400,tools=TOOLS)

    ##############################
    ## Missed balls

    missedSource = bkM.ColumnDataSource(
        data=dict(
            trialNum = missedTrials.index,
            preBlankDur = missedTrials.preBlankDur.values,
            blankDur = missedTrials.blankDur.values,
            postBlankDur = missedTrials.postBlankDur.values,
            timeToPaddle = [str("%.2f" % x) for x in missedTrials['timeToPaddle']],
            X = missedTrials[scatterVar]['X'].values,
            Y = missedTrials[scatterVar]['Y'].values
        )
    )

    missedBalls  = p.circle('X',
                  'Y', color="red",source=missedSource)
        
    missedBallGlyph = missedBalls.glyph
    missedBallGlyph.radius = ballRadius
    missedBallGlyph.fill_alpha = 0.2
    missedBallGlyph.line_color = "red"
    missedBallGlyph.line_alpha = 0.2
    missedBallGlyph.line_width = 2

    ##############################
    ## Caught balls


    caughtSource = bkM.ColumnDataSource(
        data=dict(
            trialNum = caughtTrials.index,
            preBlankDur = caughtTrials.preBlankDur.values,
            blankDur = caughtTrials.blankDur.values,
            postBlankDur = caughtTrials.postBlankDur.values,
            timeToPaddle = [str("%.2f" % x) for x in caughtTrials['timeToPaddle']],
            X = caughtTrials[scatterVar]['X'].values,
            Y = caughtTrials[scatterVar]['Y'].values
        )
    )

    
    caughtBalls  = p.circle('X','Y',color="green",source=caughtSource)

    caughtBallGlyph = caughtBalls.glyph
    caughtBallGlyph.radius = ballRadius
    caughtBallGlyph.fill_alpha = 0.2
    caughtBallGlyph.line_alpha = 0.2
    caughtBallGlyph.line_color = "green"
    caughtBallGlyph.line_width = 2

    ##############################
    ## Paddle

    paddle = p.circle(0,0, color="yellow", alpha=0.5)
    paddleGlyph = paddle.glyph
    paddleGlyph.radius = paddleRadius
    paddleGlyph.fill_alpha = 0.2
    paddleGlyph.line_color = "yellow"
    paddleGlyph.line_dash = [6, 3]
    paddleGlyph.line_width = 2
    paddleGlyph.line_color = "firebrick"


    p.y_range = bkM.Range1d(-.75,.75)
    p.x_range = bkM.Range1d(-.75,.75)

    p.axis.axis_label = 'meters'

    from collections import OrderedDict

    hover = p.select(dict(type=bkM.HoverTool))
    hover.tooltips = OrderedDict([
        ("trialNum", "@trialNum"),
        ("preBlankDur", "@preBlankDur"),
        ("blankDur", "@blankDur"),
        ("postBlankDur", "@postBlankDur"),
        ("timeToPaddle", "@timeToPaddle"),
    ])

    if( outFilePath):

        bkP.output_file(outFilePath + scatterVar + 'Scatter.html') 
        bkP.save(p)

    else:

        return p


def plotGazeErrorInteractions(sessionDict,timeOffset=0,columnName = 'cycGIWtoBallAngle_ballOn_0.00', 
    yRange = [-10,20], xVar = 'preBlankDur',lineVar = 'postBlankDur',outFilePath = False):

    blankForNFrames = np.round( timeOffset / sessionDict['analysisParameters']['fps'])
    plotFr_tr = np.array(np.add(blankForNFrames,sessionDict['trialInfo']['ballOnFr']),dtype=np.int)

    figs = []

    #### Gaze error interactions
    for subLevelNameStr in list(sessionDict['processed']['cycGIWtoBallAngle'].columns):

        yVar = columnName,subLevelNameStr
        cycToBallAng_fr = sessionDict['processed']['cycGIWtoBallAngle'][subLevelNameStr][plotFr_tr]

        p = plotInteraction(sessionDict['trialInfo'],xVar,yVar,lineVar)
        p.y_range = bkM.Range1d(yRange[0],yRange[1])
        figs.append(p)

    if(outFilePath):
        bkP.output_file(outFilePath + 'gazeInteractions.html') 
        bkP.save(bkM.layouts.Column(*figs))
    else:
        return bkM.layouts.Column(*figs)
    

def plotGazeErrorScatterArray(sessionDict,figsBefore,figsAfter,outFilePath,axisSpan=80):
    
    numFigs = figsBefore + figsAfter

    blankDur = sessionDict['expConfig']['trialTypes']['default']['blankDur']
    blankForNFrames = ( blankDur / (1./75))

    spanBetweenFigs = np.floor(blankForNFrames/(figsBefore-1))

    figOffset_fig = np.dot(range(0,figsBefore),spanBetweenFigs)

    for fIdx in range(figsAfter):
        figOffset_fig = np.append(figOffset_fig,figOffset_fig[-1]+spanBetweenFigs)

        figs = []

    for figOffset in figOffset_fig:

        plotFr_tr = np.add(figOffset,sessionDict['trialInfo']['ballOffFr'])
        plotData = sessionDict['processed']['cycGIWtoBallAngle']
        figs.append(plotGazeErrorScatter(sessionDict,plotData,plotFr_tr))

        offsetDurationStr = '%1.2f' % (figOffset * (1./75))
        figs[-1].title.text_font_size = '10pt'
        figs[-1].title.text = 'dt = ' + offsetDurationStr + 's'


    h = bkM.layouts.Row(*figs)

    if( outFilePath):

        bkP.output_file(outFilePath + 'gazeErorScatter.html') 
        bkP.save(h)

    else:
        return h


def plotGazeErrorScatter(sessionDict,scatterData,plotFr_tr,axisSpan = 80):
    
    p = bkP.figure(x_range=(-axisSpan/2.0, axisSpan/2.0), y_range=(-axisSpan/2.0, axisSpan/2.0), 
                    plot_width=400, plot_height=400)

    y_box = bkM.BoxAnnotation(plot=p, bottom=-1, top=1, fill_alpha=0.1, fill_color='black')
    x_box = bkM.BoxAnnotation(plot=p, left=-1, right=1, fill_alpha=0.1, fill_color='black') 
    p.renderers.extend([x_box, y_box])

    x = scatterData['X_worldUp' ][plotFr_tr]
    y = scatterData['Y_worldUp'][plotFr_tr]

    scatterLoc = p.scatter(x, y, marker="circle", size=10,
                  line_color="navy", fill_color="orange", alpha=0.5,name='gaze')


    return p



def plotCatchingErrorInteractions(sessionDict,outFilePath=False):
    
    figs = []

    #### Catching error interactions
    yVar = 'catchingError','2D'
    xVar = 'preBlankDur'
    lineVar = 'postBlankDur'
    p1 = plotInteraction(sessionDict['trialInfo'],xVar,yVar,lineVar)
    p1.y_range = bkM.Range1d(0,.75)
    figs.append(p1)
    
    #### Catching error interactions
    yVar = 'catchingError','X'
    xVar = 'preBlankDur'
    lineVar = 'postBlankDur'
    p2 = plotInteraction(sessionDict['trialInfo'],xVar,yVar,lineVar)
    p2.y_range = bkM.Range1d(0,.75)
    figs.append(p2)

    #### Catching error interactions
    yVar = 'catchingError','Y'
    xVar = 'preBlankDur'
    lineVar = 'postBlankDur'
    p3 = plotInteraction(sessionDict['trialInfo'],xVar,yVar,lineVar)
    p3.y_range = bkM.Range1d(0,.75)
    figs.append(p3)

    h = bkM.layouts.Column(*figs)

    if( outFilePath):
        bkP.output_file(outFilePath + 'catchingErrorInteractions.html') 
        bkP.save(h)
    else:
        return h


def plotPredictiveCatchingErrorInteractions(sessionDict,outFilePath):
    figs = []

    #### Catching error interactions
    yVar = 'predictiveCatchingError','2D'
    xVar = 'preBlankDur'
    lineVar = 'postBlankDur'
    p1 = plotInteraction(sessionDict['trialInfo'],xVar,yVar,lineVar)
    p1.y_range = bkM.Range1d(0,.75)


    #### Catching error interactions
    yVar = 'predictiveCatchingError','X'
    xVar = 'preBlankDur'
    lineVar = 'postBlankDur'
    p2 = plotInteraction(sessionDict['trialInfo'],xVar,yVar,lineVar)
    p2.y_range = bkM.Range1d(0,.75)

    #### Catching error interactions
    yVar = 'predictiveCatchingError','Y'
    xVar = 'preBlankDur'
    lineVar = 'postBlankDur'
    p3 = plotInteraction(sessionDict['trialInfo'],xVar,yVar,lineVar)
    p3.y_range = bkM.Range1d(0,.75)
    
    figs = [p1,p2,p3]

    h = bkM.layouts.Column(*figs)
       
    if( outFilePath):

        bkP.output_file(outFilePath + 'predictiveCatchErrorInteractions.html') 
        bkP.save(h)

    else:
        return h


# def plotCatchingErrFigStack(sessionDict,xVar,yVar):
    
#   xVar = 'postBlankDur'
#   yVar = 'catchingError','2D'

#   p1 =  plotMainEffect(sessionDict['trialInfo'],xVar,yVar)
#   p1.y_range = bkM.Range1d(-0.25,0.75)

#   yVar = 'catchingError','X'

#   p2 = plotMainEffect(sessionDict['trialInfo'],xVar,yVar)
#   p2.y_range = bkM.Range1d(-0.25,0.75)

#   yVar = 'catchingError','Y'

#   p3 = plotMainEffect(sessionDict['trialInfo'],xVar,yVar)
#   p3.y_range = bkM.Range1d(-0.25,0.75)

#   return bkP.vplot(p1,p2,p3)

def plotCatchingErrorMainEffects(sessionDict):
    
    #### Catching error by postblankDur
    xVar = 'postBlankDur'
    yVar = 'catchingError','2D'
    bkP.output_file(outFilePath + 'catchingError_postBlank.html') 
    bkP.save(plotCatchingErrFigStack(sessionDict,xVar,yVar))

    #### Catching error by preblankDur
    xVar = 'preBlankDur'
    yVar = 'catchingError','2D'
    bkP.output_file(outFilePath + 'catchingError_preBlank.html') 
    bkP.save(plotCatchingErrFigStack(sessionDict,xVar,yVar))



def errorbar(fig, x, y, xerr=None, yerr=None, color='red', point_kwargs={}, error_kwargs={}):

    fig.circle(x, y, color=color, **point_kwargs)


    if xerr is not None:
      x_err_x = []
      x_err_y = []
      for px, py, err in zip(x, y, xerr):
          x_err_x.append((px - err, px + err))
          x_err_y.append((py, py))
      fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

    if yerr is not None:
      y_err_x = []
      y_err_y = []
      for px, py, err in zip(x, y, yerr):
          y_err_x.append((px, px))
          y_err_y.append((py - err, py + err))
      fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)

def plotInteraction(trialInfoDF,xVar,yVar, lineVar,outFilePath=False):

    lineNames = np.array(np.unique(trialInfoDF[lineVar].round(decimals =2)),dtype=np.str)
    groupedByPre =  trialInfoDF.groupby([lineVar,xVar])

    mean_cond = []
    std_cond  = []

    for gNum, gr in groupedByPre:
        mean_cond.append(np.nanmean(np.array(gr[yVar].values,dtype=np.float)))
        std_cond.append(np.nanstd(np.array(gr[yVar].values,dtype=np.float)))

    numV1 = len(np.unique(np.array(groupedByPre.groups.keys())[:,0]))
    numV2 = len(np.unique(np.array(groupedByPre.groups.keys())[:,1]))
    mean_pre_post = np.reshape(mean_cond,[numV1,numV2])

    mean_pre_post = np.reshape(mean_cond,[numV1,numV2])
    std_pre_post = np.reshape(std_cond,[numV1,numV2])

    xs = [np.unique(trialInfoDF[xVar].round(decimals =2))]*3
    ys = [np.array(xyz,dtype=np.float) for xyz in mean_pre_post]
    yerr = [np.array(xyz,dtype=np.float) for xyz in std_pre_post]

    p = bkP.figure(width=800, height=400,tools = "pan,reset,resize,hover")

    off = [-.01, 0, .01]
    clist = ['blue','orange','green']

    for j in range(3):
       p.line( xs[j]+off[j], ys[j] ,line_width=3,color=clist[j],legend = lineVar + ' ' +  lineNames[j])
       errorbar(p,xs[j]+off[j],ys[j], yerr=yerr[j],color=clist[j], point_kwargs={'line_width':3,'size': 10}, 
        error_kwargs={'line_width': 3})

    p.xaxis.axis_label = xVar
    p.yaxis.axis_label = ''.join(yVar)

    if( outFilePath ):

        bkP.output_file(outFilePath + yVar + '_' + xVar +'_' + lineVar + '.html') 
        bkP.save(p)

    else:

        return p
    


def plotMainEffect(dF,xVar,yVar, plotTitle = False):

    groupedBy =  dF.groupby(xVar)

    pre_mean = []
    pre_std = []

    for gNum, gr in groupedBy:
        pre_mean.append(np.mean(gr[yVar].values))
        pre_std.append(np.std(gr[yVar].values))

    xs = np.unique(dF[xVar].round(decimals =2))
    yerr = pre_std
    ys = pre_mean

    # plot the points
    p = bkP.figure(width=800, height=400)
    p.line(xs, ys, color='orange',line_width=2)
    errorbar(p,xs,ys, yerr=yerr,point_kwargs={'size': 10}, error_kwargs={'line_width': 3})

    p.xaxis.axis_label = xVar
    p.yaxis.axis_label = ''.join(yVar)

    if( plotTitle ):
        p.title.text = plotTitle
        
    return p
