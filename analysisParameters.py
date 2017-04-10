import os

def loadParameters(fileTime):

    analysisParameters = dict()

    analysisParameters['gazeFilter'] = 'medianAndAverage'
    analysisParameters['filterParameters'] = [3,3]
    analysisParameters['expCfgName'] = "gd_pilot.cfg"
    analysisParameters['sysCfgName'] = "PERFORMVR.cfg"
    analysisParameters['hmd'] = 'DK2'
    analysisParameters['fps'] = (1.0/75.0)
    analysisParameters['hmdResolution'] = [1920.0,1080.0]
    analysisParameters['hmdScreenSize'] = [0.126,0.071]
    analysisParameters['averageEyetoScreenDistance'] = 0.0725
    analysisParameters['outlierThresholdSD'] = 2
    analysisParameters['removedOutliersFrom'] = []
    analysisParameters['fileName'] = "exp_data-" + fileTime
    analysisParameters['fileTime'] = fileTime
    analysisParameters['numberOfCalibrationPoints'] = 27

    containingDir = os.getcwd().split('/')[-1]
    
    if( os.getcwd().split('/')[-2] == 'flipAnalysis' ): #containingDir == 'analyzeE1' or containingDir == 'eventLabeller' ):
        
        print '**** Loading 1 **** '

        analysisParameters['filePath'] = "Data/" + fileTime + "/"

        if not os.path.exists('Figures/' + fileTime):
            os.makedirs('Figures/' + fileTime)

        analysisParameters['figOutDir'] = 'Figures/' + fileTime + '/'
        
        analysisParameters['eventClassifierLoc'] = 'Modules/SVMClassifier.pickle'

    else:

        print '**** Loading 2 **** '
        
        analysisParameters['filePath'] = "../Data/" + fileTime + "/"

        if not os.path.exists('../Figures/' + fileTime):
            os.makedirs('../Figures/' + fileTime)

        analysisParameters['figOutDir'] = '../Figures/' + fileTime + '/'
        
        analysisParameters['eventClassifierLoc'] = '../Modules/SVMClassifier.pickle'

    return analysisParameters

