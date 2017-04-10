
# Framework
import sys, os
from copy import copy, deepcopy

# Project
from Engine import Engine
from Metric import Metric, MetricResultsGroup

#
#   This part is coded quite quickly and will need some refactory if 
#   code need to be maintained.
# 

class Analyser(object):    
    """
    The Analyser performs analysis on data sets using metrics.
    
    This class is separate from the Engine since it can use
    several engines (one for each data set) in its analysis
    """
    def __init__(self): #, algParams=None
        super(Analyser, self).__init__()
        
    def setParameters(params):
        self.algParams = params
    
    def analyse(self, referenceEngine, algs, metrics, datapath=None):
        '''
        The engine is attached to give some settings, such as the
        settings of the algorithms. if datapath is None it will also
        be used as data source. This function will construct its own
        engine objects though, this engine object will remain unchanged.
        
        The algs should be ready objects for the algorithms with their
        parameters set.
        '''
        # TODO Reset metrics?
        
        self.metricDict = {}
        
        #engine = Engine()
        #engine.stimulus = referenceEngine.stimulus
            #engine.data = data
                    
            # TODO - Probably change around the order of these
            # for alg in algs:
            #     engine.alg = alg
            #     engine.process()

        for metric in metrics:

            metric_list = []

            for alg in algs:
                # TODO - calls this more than it has to
                
                # Deepcopy is important, otherwise it will handle the same data
                #m = deepcopy(metric)
                name = metric.__class__.__name__
                # if name in self.metricDict:
                #     m = self.metricDict[name]
                # else:
                #     m = deepcopy(metric)
                m = deepcopy(metric)
                
                for data_i, engine in enumerate(self.engineIterator(referenceEngine, datapath)):
                    engine.alg = alg
                    engine.process()
                    m.analyseAndStore(engine)
                    algname = engine.getFilterName()
                    m.associateWithAlgorithmName(algname)
                
                #m = metric
                metric_list.append(m)
                
            self.metricDict[metric.__class__.__name__] = metric_list
                
    def getMetricNames(self):
        return self.metricDict.keys()
                
    def getResultsGroups(self, metricName):
        '''
        This generator iterates the results by returning a tuple
        with the name and the reults (str, MetricResults)
        '''
        resultsGroups = {}
        
        for metric in self.metricDict[metricName]:
            resultsList = metric.getResults()
            for i, results in enumerate(resultsList):
                # Set the algorithm name
                results.setAlgorithmName(metric.getAssociateAlgorithmName())
                if i not in resultsGroups:
                    resultsGroups[i] = MetricResultsGroup(results.getType())
                    
                resultsGroups[i].addMetricResults(results)
 
        return resultsGroups.values()
            
    def engineIterator(self, engine, datapath):
        '''
        Iterates all data objects in the analysis
        
        If datapath is None, then only the data in the engine object
        is used. If datapath is a path, all NPZ files in the path
        is iterated, otherwise datapath should point at a single NPZ file
        '''
        
        if datapath == None:
            yield engine
            return
        
        if os.path.isdir(datapath):
            files = []
            for fn in os.listdir(datapath):
                name, ext = os.path.splitext(fn)
                if ext.lower() == '.npz':
                    files.append(os.path.join(datapath, fn))
        else:
            files = [datapath]
            
        #retEngine = deepcopy(engine)
        retEngine = Engine()
        retEngine.stimulus = engine.stimulus
        retEngine.filtering = engine.filtering
        for path in files:
            retEngine.loadDataNPZ(path)
            yield retEngine