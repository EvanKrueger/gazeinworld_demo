
# Framework
import numpy as np

class Metric(object):
    """
    Class operates on recorded data, an algorithm, stimulus and labelled data
    to quantify the performance of the algorithm. This can be done in many ways
    and results are stored in MetricResults-derived classes.
    """
    def __init__(self):
        super(Metric, self).__init__()
        
        # Results are stored per data set
        self.results = []
        
        # It is sometimes needed to associate a metric with a certain algorithm
        # so here is a place where its name can be stored.
        self.algName = str()
        
    #
    # Use these functions when communicating with the class
    #
    
    def associateWithAlgorithmName(self, algName):
        self.algName = algName
        
    def getAssociateAlgorithmName(self):
        return self.algName
    
    def analyseAndStore(self, engine):
        self.results.append(self.analyse(engine))
    
    def getResults(self): #, recordings=None):
        # if recordings == None:
        #     recordings = self.results.keys()
        return self.arrangeResults()#recordings)


    #
    # Override the following functions in your metric
    #
    
    def analyse(self, engine):
        '''
        Analyses one recording of data. Results are returned in a
        datatype free of choice (dictionary is handy).
        '''
        return {}
        
    def arrangeResults(self): #, recordings):
        '''
        This class takes all the analysed data and generates a list of MetricResults.
        The implementation side can choose how to render this list.
        
        This function is needed for two reasons:
        1. How to translate the stored data into MetricResults objects.
        2. It specifies how to combine data from several experiments, this
           can be done by means, concatenation, maximum value, etc.
        '''
        return []
        
class MetricResults(object):
    # Metric types
    def __init__(self, metrictype, name, data):
        self.metrictype = metrictype
        self.name = name
        self.data = data
        self.algName = ""
        
    def getType(self):
        return self.metrictype    
    
    def setAlgorithmName(self, algName):
        self.algName = algName
    
    def getAlgorithmName(self):
        return self.algName
    
    def getName(self):
        return self.name
    
    def getData(self):
        return self.data
        
        
class MetricResultsGroup(object):
    '''
    Essentially this is a group of metric results to represent one metric
    with results from several algorithms. This is what is being passed to
    a MetricResultsRenderer so it can decide how to display the different
    algorithms (in a table, graph, etc.)
    
    The primary reason for this being a class and not just kept as a list
    is because it gives needed structure to what is being passed around.
    '''
    def __init__(self, metricType):
        super(MetricResultsGroup, self).__init__()
        self.metricType = metricType
        self.metricResultsList = []
    
    def addMetricResults(self, metricResults):
        self.metricResultsList.append(metricResults)
        
    def getMetricType(self):
        return self.metricType
        
    def getResultsList(self):
        return self.metricResultsList
        
    def getName(self):
        if len(self.metricResultsList) == 0:
            return str()
        else:
            return self.metricResultsList[0].getName()
        
        
        
METRIC_RENDERERS = {}
def GetMetricRenderer(metricResultsGroup):
    assert isinstance(metricResultsGroup, MetricResultsGroup)
    '''Construct a metric renderer for the given results object'''
    return METRIC_RENDERERS[metricResultsGroup.getMetricType()](metricResultsGroup)

class MetricResultsRenderer(object):
    """This stores a description of how to render an analysis results"""
    def __init__(self, metricResultsGroup):
        super(MetricResultsRenderer, self).__init__()
        self.metricResultsGroup = metricResultsGroup # MetricResultsGroup object
        
#    def iterateResults(self):
#        '''Iterate the results for the different algorithms'''
#        return self.results

    def getResultsList(self):
        return self.metricResultsGroup.getResultsList()
    
    def createWidget(self, parent):
        pass

class RegisterMetricResultsRenderer(object):
    ''' Registers a handler for the type 'vartype' '''
    def __init__(self, metrictype):
        self.metrictype = metrictype

    def __call__(self, handler):
        METRIC_RENDERERS[self.metrictype] = handler
        return handler


#
#   PyQt specific Metric Results handlers. Possibly move to PyQt/ folder TODO
#

from PyQt4 import QtGui, QtCore

@RegisterMetricResultsRenderer('table')
class __handler_table(MetricResultsRenderer):
    def createWidget(self, parent):
        widget = QtGui.QTableWidget(parent)
        
        cols = len(self.getResultsList()[0].getData())
        rows = len(self.getResultsList())

        widget.setColumnCount(cols)
        widget.setRowCount(rows)


        for i, key in enumerate(self.getResultsList()[0].getData()):
            item = QtGui.QTableWidgetItem()
            item.setText(key)
            widget.setHorizontalHeaderItem(i, item)
            
        for row, res in enumerate(self.getResultsList()):
            item = QtGui.QTableWidgetItem()
            item.setText(res.getAlgorithmName())
            widget.setVerticalHeaderItem(row, item)
            
            for col, tup in enumerate(res.getData().items()):
                (key, value) = tup
                widget.setItem(row, col, QtGui.QTableWidgetItem(str(value)))

        return widget
        
import matplotlib
import matplotlib.pyplot
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvasStealsWheelEvent
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
    
class FigureCanvas(FigureCanvasStealsWheelEvent):
    '''
    This makes scrolling work above the canvases, there might be a setting
    for this, but I couldn't find it, so here is a straightforward solution.
    Normally scrolling is used for zooming and such, but not in our case.
    '''
    def wheelEvent( self, event ):
        event.ignore()
    
@RegisterMetricResultsRenderer('graph')
class __handler_graph(MetricResultsRenderer):
    '''Creates a graph widget using matplotlib'''
    def createWidget(self, parent):
        
        dpi = 100
        fig = Figure((5.0, 4.0), dpi=dpi)
        fig.subplots_adjust(top=1.0, bottom=0.0)
        canvas = FigureCanvas(fig)
        canvas.setMouseTracking(False)
        
        canvas.setMinimumHeight(300)
        
        axes = fig.add_subplot(111)     

        return canvas
        
@RegisterMetricResultsRenderer('histogram')
class __handler_histogram(MetricResultsRenderer):
    '''Creates a histogram using matlibplot'''
    def createWidget(self, parent):
        dpi = 100
        alg_count = len(self.getResultsList())

        fig = Figure((5.0, 4.0), dpi=dpi, facecolor=(1, 1, 1))
        buf = 0.15/float(alg_count)
        fig.subplots_adjust(top=1.0-buf, bottom=buf, left=0.1, right=0.95, hspace=0.4)
        canvas = FigureCanvas(fig)
        
        canvas.setMinimumHeight(300*alg_count)

        for i, res in enumerate(self.getResultsList()):
            axes = fig.add_subplot(alg_count, 1, i+1)
            data = res.getData()
        
            if data['data']:
                axes.hist(data['data'], data['bins'])
            axes.set_title(res.getAlgorithmName())
            axes.set_xlabel(data['xlabel'], size="small")
            axes.set_ylabel(data['ylabel'], size="small")

        return canvas
        
@RegisterMetricResultsRenderer('barchart')
class __handler_barchart(MetricResultsRenderer):
    '''Creates a barchart using matlibplot'''
    def createWidget(self, parent):
        dpi = 100
        alg_count = len(self.getResultsList())
        start = 0.1
        width = 0.8/2
        colours = 'rbygp'

        fig = Figure((5.0, 4.0), dpi=dpi, facecolor=(1, 1, 1))
        fig.subplots_adjust(top=0.8, bottom=0.1)
        canvas = FigureCanvas(fig)
        
        #canvas.setMinimumHeight(300*alg_count)
        canvas.setMinimumHeight(300)
        
        axes = fig.add_subplot(111)     

        info = {}
        xticks = []
        for i, res in enumerate(self.getResultsList()):
            data = res.getData()
            
            for key, value in data.items():
                if key not in info:
                    info[key] = []
                info[key].append(value)
            
            xticks += [res.getAlgorithmName()]
        
        ind = np.arange(alg_count)
        rects = []
        for i, (key, values) in enumerate(info.items()):
            r = axes.bar(ind+start+i*width, values, width, color=colours[i])
            rects.append(r[0])
         
        #axes.set_ylabel('Value')
        axes.set_xticks(ind+0.5)
        axes.set_xticklabels(xticks) 
        assert len(rects) == len(info)
        axes.legend(rects, info.keys(), 
            bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=2, mode="expand", borderaxespad=0.)

        return canvas