
STATE_NAMES = ['Fixation', 'Saccade', 'Smooth Pursuit', 'Not Found']

# Parameter
class Parameter(object):
    def __init__(self, name, caption, vartype, default=None, varinfo=None):
        self.name = name
        self.caption = caption
        self.vartype = vartype
        self.varinfo = varinfo
        self.default = default
        # Range?

class Filter(object):
    FIXATION = 0
    SACCADE = 1
    SMOOTH_PURSUIT = 2
    NOT_FOUND = 3

    def __init__(self):
        # This is where the results is stored
        self.state = 0 # FIXATION/SACCADE/SMOOTH_PURSUIT
        self.params = {} # Parameters are stored here
        
    def process(self, data, setFixation):
        '''
        Processes the numpy array data at the position index
        Use setFixation(index, state, ordinal) to tag the data.
        '''
        pass
        
    def getParameter(self, name):
        return self.params[name]
        
    def setParameter(self, name, value):
        self.params[name] = value
        
    def getParameterSpec(self, name):
        '''
        returns the Parameter object of the parameter
        '''
        for p in self.parameters():
            if p.name == name:
                return p
        return None
            
    def parameters(self):
        ''' This should be overloaded to return an array of Parameter objects '''
        return []