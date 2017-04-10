
# Framworks
import re
import sys
import os
import copy
import numpy as np

# Project
from Filter import Filter
from Stimulus import Stimulus

# This class is the main file in the Eye Studio
class Engine:
    class Fixation(object):
        def __init__(self, start, end, pos):
            self.start = start
            self.end = end
            self.pos = pos
            
        def __repr__(self):
            return "Fixation(%s, %s, %s)" % (self.start, self.end, self.pos)
            
        def getDuration(self):
            return self.end - self.start
    
    def __init__(self, alg=None):
        self.data = None
        
        self.pos = (0,0) # Stores current position
        self.gazetype = None # Stores the result: smooth pursuit, fixation or saccade
        self.curTime = 0 # [ms]
        self.loop_count = 0
        self.resolution = (1,1)
        self.imports = {}

        #if instanceof(filter, Filter)
        self.alg = alg
        self.state = 0
        
        # Results are stored per algorithm with a dictionary with 
        # the filter name as argument.
        self.results = {}
        self.manualPositions = {}
        self.frames = {}
        self.fixations = []
        self.filtering = None # TODO Probably remove
        
        # This is similar to results, but there can only be one labelling
        # per file. This is the labelling that will be used in training
        # or analysis and should be considered being the correct one.
        self.labels = None
        self.labelframes = None
        
        # Stimulus object
        self.stimulus = None
        
        # Temp
        self.stimulus = Stimulus()
        #self.stimulus.loadFromTextFile('/Users/Gustav/Dropbox/Ollanta+Gustav/tobii/EyeStudio/test.stim.txt')
        #self.stimulus.initPositions()
        
    def updateLabels(self):
        self.labelframes = self.findFixations(self.labels)

    def setLabelsFromResults(self, key):
        try:
            self.labels = copy.copy(self.results[key])
            self.updateLabels()
            return True
        except:
            return False

    def hasLabelling(self):
        return self.labels is not None

    def getLabel(self, i):
        return self.labels[i]
        
    def getState(self, i, key):
        return self.results[key][i]

    def setFilter(self, alg):
        self.alg = alg

    # This will import the filter files if they haven't already
    def setFilterByName(self, name):
        def showTraceback():
            import traceback
            feedback = traceback.format_exc()
            print>>sys.stderr, "------------------------------------------"
            print>>sys.stderr, feedback
        
        if name not in self.imports:
            # Load filter file and see if it loads alright.
            try:
                self.imports[name] = __import__('Filters.%s' % (name), globals(), locals(), [name], -1)
            except Exception, e:
                showTraceback()
                self.alg = None
                return False
        
        # Set filter
        try:
            self.setFilter(eval("self.imports[name].%s()" % name))
        except Exception, e:
            showTraceback()
            self.alg = None
            return False
            
        return True
        
    def setLabelRange(self, fr, to, label, ordinal=None):
        '''Sets the label of a range, if to is None, then loop_count is used'''
        if to == None:
            to = self.loop_count

        if fr == None or fr == to:
            return
            
        if fr > to:
            fr, to = to, fr
         
         
        updateOrdinals = ordinal == None
        if updateOrdinals:
            if fr == 0:
                ordinal = 0
            else:
                ordinal = self.labels[fr-1,1]
                if self.labels[fr-1,0] != label:
                    ordinal += 1
                
        for i in xrange(fr, to):
            self.labels[i,0] = label
            self.labels[i,1] = ordinal
            
        # Update the ordinal numbers
        start = ordinal
        if self.labels[to-1,0] != self.labels[to,0]:
            start += 1
        
        delta = start - self.labels[to,1]
        if updateOrdinals:
            for i in xrange(to, len(self.labels)):
                self.labels[i,1] += delta
            
        self.updateLabels()
        
        
        
    def getNamesOfFiltersWithResults(self):
        return self.results.keys()

    def loadData(self, datafile):
        ext = os.path.splitext([datafile])[1]
        if ext == '.tsv':
            return self.loadDataTSV(datafile)
        elif ext == '.npz':
            return self.loadDataNPZ(datafile)
            
        
    def loadDataTSV(self, datafile):
        ''' Loads a Tobii CSV file '''

        re_size = re.compile(r'Recording resolution:\s*(\d+)\s*x\s*(\d+)')
        re_screen = re.compile(r'Screen size.*?:\s*([\d.]+)\s*x\s*([\d.]+)')

        self.screenSize = None
        self.resolution = None

		# TODO This should handle *any* TSV file.
        re_header = re.compile(r'Timestamp.*')
        
        f = open(datafile, 'r')

        # Every element is stored as (time, x, y, head distance)
        self.data = np.array([]).reshape(0, 4)

        reading_data = False
        for line in f:
            if reading_data:
                # If decimal comma is used, convert to point
                line = line.replace(',', '.')
                values = line.split()
                if len(values) == 1: # Blink, no data
                    pass
                else:
                    pos = map(float, (values[0], values[1], values[2], values[3]))
                    self.data = np.vstack([self.data, np.hstack(pos)])
            else:
                res = re_size.match(line)
                if res:
                    self.resolution = map(int, (res.group(1), res.group(2)))
                    continue
                
                line2 = line.replace(',', '.')
                res = re_screen.match(line2)
                if res:
                    self.screenSize = map(float, (res.group(1), res.group(2)))
                    continue

                res = re_header.match(line)
                if res:
                    reading_data = True
                    continue
                    
        if not self.screenSize or not self.resolution:
            raise Exception('TSV file must specify screen size and resolution')
                    
        # Converting data from pixels to centimetres.
        for i in xrange(2):
            self.data[:,i+1] *= self.screenSize[i]/self.resolution[i]
        
        # Create an empty vector for the labelling
        self.resetLabels()
        
        if self.hasStimulus():
            self.stimulus.initPositions(self.data, self.screenSize)        
        
        #self.data = loadtxt(datafile, unpack=True, delimiter=',')

    def resetLabels(self):
        '''Creates an empty data structure for labels'''
        c = self.dataSize()
        self.labels = np.array([None, 0]*c).reshape(c, 2)

    def loadDataNPZ(self, datafile):
        ''' Loads a Eye Studio data file '''
        infile = open(datafile, "rb")
        npz = np.load(infile)

        try:
            self.resolution = tuple(npz['resolution'])
            self.screenSize = tuple(npz['screen_size'])
        except KeyError:
            # TODO Remove these defaults
            self.resolution = (1280, 1024)
            self.screenSize = (337, 268)
            
            
        # Loda data into the model
        self.data = npz['data']
        
        # TODO Tidy up or remove

        if True:
	        blinkpoints = []
	        for i in xrange(len(self.data)):
	            x, y = self.data[i,1:3]
	            if x <= 0 or x >= self.resolution[0] or \
	               y <= 0 or y >= self.resolution[1]:
	               blinkpoints.append(i)
	            elif blinkpoints:
	                dur = self.data[blinkpoints[-1],0] - self.data[blinkpoints[0],0]
	#                print dur
	                # TODO Temporarily fix all blinks
	                dur = 0

	                if dur <= 1000 and blinkpoints[0] >= 2:
	                    # It's below 100 ms, so try to "remove" it
	                    for j in [blinkpoints[0]-1]+blinkpoints:
	                        self.data[j,1:] = self.data[blinkpoints[0]-2,1:]
	                else:
	                    # Just label them as invalid, this is done by setting them
	                    # to the following values. 
	                    for j in blinkpoints:
	                        self.data[j,1:4] = [-100, -100, 0]
                        
	                blinkpoints = []
        
	        # TODO End of blink filter
        
        # Do filtering
        if self.filtering:
            self.data = self.filtering(self.data)
        
        # Load stimulus
        if 'stimulus' in npz:
            self.stimulus = Stimulus()
            self.stimulus.setEventsFromArray(npz['stimulus'])
        
        try:
            self.labels = npz['labels']
        except:
            self.resetLabels()

        infile.close()

        self.updateLabels()
        
        if self.hasStimulus():
            self.stimulus.initPositions(self.data, self.screenSize)
        
    def saveDataNPZ(self, datafile):
        ''' Saves a Eye Studio data file '''
        outfile = open(datafile, "wb")
        
        d = dict(
            data=self.data, 
            resolution=self.resolution, 
            labels=self.labels,
            screen_size=self.screenSize,
        )

        if self.hasStimulus():
            d['stimulus'] = self.stimulus.getEventsAsArray()
        
        np.savez(outfile, **d)
        outfile.close()
        
    def process(self):
        '''
        Process the loaded data with the current algorithm, all at once.
        '''
        # The two values in this array is (state, classification ordinal)
        c = self.dataSize()
        fn = self.getFilterName()
        res = self.results[fn] = np.array([None, 0]*c).reshape(c, 2)
        manpos = self.manualPositions[fn] = np.array([None, None]*c).reshape(c, 2)

        def setFixation(index, state, ordinal=None, pos=None):
            res[index] = np.hstack([state, ordinal])
            if pos is not None:
                manpos[index] = np.hstack(pos)
        
        self.alg.process(self.data, setFixation)
        
        # Set ordinals automatically by patching together equal states
        # If the first ordinal is None, then they will all be 
        # automatically set.
        if c != 0 and res[0,1] == None:
            lastState = None
            curOrdinal = -1
            for i in xrange(c):
                state = res[i,0]
                if state != lastState:
                    curOrdinal += 1
                lastState = state
                    
                res[i,1] = curOrdinal
        
        # Rasterize
        self.frames[fn] = self.findFixations(res, manpos)

        #print ":",len(list(self.iterateStateDurations(Filter.SACCADE)))
        # TODO
        #print list(self.iterateStateDurations(Filter.SACCADE))
        
    def removeResults(self, filterName=None):
        '''
        Removes the results and thus unloads the algorithm's results
        '''
        if not filterName:
            filterName = self.getFilterName()
        
        del self.results[filterName]
        del self.frames[filterName]
        
        
    def findFixations(self, data, manualPositions=None):
        '''
        When all gaze points have been classified, this determines the positions
        of the fixations for each frame. This is needed for fast seeking and playback.
        '''

        #if not filterName:
        #    filterName = self.getFilterName()
    
        c = len(data)
        #c = self.dataSize()
        #res = self.results[filterName]
        res = data

        # Frames: (x, y, elapsed time since fixation started) 
        # Time will probably so far be ticks
        #frames = self.frames[self.getFilterName()] = np.array([0.0,0.0,-1.0]*c).reshape(c, 3)
        frames = np.array([0.0,0.0,-1.0]*c).reshape(c, 3)
        self.fixations = []
        
        last = 0
        last_id = 0
        for i in xrange(c):
            if res[i,1] != last_id or i == c-1:
                if res[i-1,0] == Filter.FIXATION:
                    # Use manual positions if the first frame has a manual position
                    if manualPositions is not None and manualPositions[last,0] is not None:
                        for j in xrange(last, i):
                            frames[j] = np.hstack([manualPositions[j], j-last])

                        # Store fixation in case a metric wants to retrieve a list
                        # of all fixations.
                        self.fixations.append(self.Fixation(last, i, manualPositions[last]))

                    else:
                        # Calculate mean
                        meanpos = (
                            self.data[last:i,1].mean(),
                            self.data[last:i,2].mean(),
                        ) 
                        
                        # Store fixation in case a metric wants to retrieve a list
                        # of all fixations.
                        self.fixations.append(self.Fixation(last, i, meanpos))

                        for j in xrange(last, i):
                            frames[j] = np.hstack([meanpos, j-last])
                
                last_id = res[i,1]
                last = i
                
        return frames
        
    def getFixations(self):
        return self.fixations
        
    def getFixation(self, loop_count):
        for f in self.fixations:
            if f.start >= loop_count and f.end < loop_count:
                return f
        
    def dataReady(self):
        ''' Returns whether data is ready '''
        return self.dataSize() > 0
    
    def filterReady(self):
        return self.alg != None
        
    def resultsReady(self, filterName=None):
        ''' Returns whether results are ready or not '''
        if not filterName:
            filterName = self.getFilterName()
        return filterName in self.results
  
    def playback(self, milliseconds):
        '''
        This will make everything jump ahead a certain number of
        milliseconds.
        '''
        self.curTime += milliseconds
        while self.getTime() < self.curTime:
            if not self.stepThrough():
                return False
                
        return True
  
    def stepThrough(self):
        '''
        Handles the main loop and outputs preprocessed data.
        '''
        if self.loop_count >= self.dataSize():
            # This makes it render the last frame correctly
            self.loop_count = self.dataSize() - 1
            return False # This signals it should be stopped
        else:
            # TODO This section is becoming obsolete.
            self.pos = (self.data[self.loop_count, 1], self.data[self.loop_count, 2])
            self.loop_count += 1

            # Set state
            if self.resultsReady():
                self.state = self.alg.state
            else:
                self.state = []

            # Output state
            # ...
            
            if self.loop_count == self.dataSize():
                self.loop_count -= 1
                return False
            else:
                return True
        
    def getFilterName(self):
        '''
        Returns the name of the current filter
        '''
        if self.alg:
            return self.alg.__class__.__name__
        else:
            return None
    
    def setLoopCount(self, count):
        ''' Sets the current position of the data '''
        self.loop_count = count
        self.curTime = self.getTime()

    def step(self, offset):
        ''' Changes loop count by offset '''
        new_lp = self.loop_count + offset
        new_lp = max(0, min(self.dataSize()-1, new_lp))
        self.setLoopCount(new_lp)
    
    def dataSize(self):
        ''' Returns the number of data points '''
        if self.data == None:
            return 0
        else:
            return self.data.shape[0]
        
    def getGazePos(self, i=None):
        if i == None: i = self.loop_count
        return self.data[i,1:3]

    def getTime(self, i=None):
        if i == None: i = self.loop_count
        return self.data[i,0]

    def getHeadDistance(self, i=None):
        if i == None: i = self.loop_count
        return self.data[i,3]

    def hasLabelFrames(self):
        return self.labelframes is not None

    def getLabelFrames(self):
        return self.labelframes

    def getFrames(self, filterName=None):
        '''
        Fetches the current frames, or the frames of the given filter name
        '''
        if not filterName:
            filterName = self.getFilterName()
        
        return self.frames[filterName]
        
    
    def getFixationElapsed(self, loop_count, frames):
        ''' 
        Retrieves the (x, y, elapsed time since this fixation started),
        or None if no fixation is happening.
        '''
        if frames[loop_count,2] >= 0.0:
            return frames[loop_count]
        else:
            return None
            
    def getResolution(self):
        '''Returns the screen resolution of the recorded data as a tuple'''
        return self.resolution

    def getScreenSize(self):
        '''Returns the screen size'''
        return self.screenSize
        
    def hasStimulus(self):
        return self.stimulus != None

    def getStimulus(self):
        return self.stimulus     
        
    
    def iterateStateTicks(self, state):
        '''
        Returns a list of tuples with the start and end tick of the
        given state.
        '''    
        returnList = []
        lastOrdinal = None
        lastStartState = None
        res = self.results[self.getFilterName()]
        for i, tup in enumerate(res):
            s, ordinal = tup
            if lastStartState != None and ordinal != lastOrdinal:
                yield lastStartState, i
                lastStartState = None

            if s == state and lastStartState == None:
                lastStartState = i
                lastOrdinal = ordinal
                
            lastOrdinal = ordinal
        
        
    
    def iterateStateTimes(self, state):
        '''
        Returns a list of tuples with the start and end time of the
        given state. Perfect for checking mean fixation time, etc.
        
        TODO Change this so that it uses iterateStateTicks.
        '''
        returnList = []
        lastOrdinal = None
        lastStart = None
        res = self.results[self.getFilterName()]
        for i, tup in enumerate(res):
            s, ordinal = tup
            if lastStart != None and ordinal != lastOrdinal:
                t = self.getTime(i)
                yield lastStart, t
                lastStart = None
                #returnList.append((lastStart, t))

            if s == state and lastStart == None:
                t = self.getTime(i)
                lastStart = t
                lastOrdinal = ordinal
                
            lastOrdinal = ordinal
            
        #print returnList
        #return returnList
        
    def iterateStateDurations(self, state, only_active=True):
        for start, end in self.iterateStateTimes(state):
            if only_active and self.hasStimulus():
                # Check if this state is in an inactive region, it will check only the middle
                # which should be enough for saccades at least.
                t = (start+end)/2.0
                
                e = self.stimulus.getEvent(t)
                if e and not e.active:
                    continue
                
                
            yield end-start
    
    def loadStimulusFromFile(self, filename):
        self.stimulus = Stimulus()
        self.stimulus.loadFromTextFile(filename)
        if self.dataReady():
            self.stimulus.initPositions(self.data, self.resolution, self.screenSize)
        
    # def getStateDurationBins(self, state, bins=0.1):
    #     '''
    #     Returns histogram information about 
    #     '''
    #     bins = dict()
    #     
    #     for start, end in self.iterateStateTimes(state):
    #         dur = end - start
    #         binNumber = int(dur/bins)
    #         try:
    #             bins[binNumbers*bins] += 1
    #         except: 
    #             bins[binNumbers*bins] = 1
