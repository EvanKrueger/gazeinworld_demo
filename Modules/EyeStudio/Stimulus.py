
# Framework
import re
import numpy as np
from copy import copy

class Stimulus(object):
    """Represents a simple stimulus with a white circle"""
    class Event(object):
        """Event for the white circle"""
        def __init__(self, duration_ms, pos=None, active=False):
            super(Stimulus.Event, self).__init__()
            self.duration_ms = float(duration_ms)
            self.pos = pos
            self.active = active
            assert(pos == None or (isinstance(pos, tuple) and len(pos) == 2))
            
        def __str__(self):
            return "Event(duration_ms=%s, pos=%s)" % (self.duration_ms, self.pos)
        
    def __init__(self):
        super(Stimulus, self).__init__()
        self.events = list()
        self.positions = None
        self.active = None
        
        self.data = None
        self.screen_size = None
        
        # deltaTime can be set to 
        self.deltaTime = 0.0
        
    def getEventsAsArray(self):
        return map(lambda x: [x.duration_ms, x.pos, x.active], self.events)
        
    def setEventsFromArray(self, array):
        self.events = []
        for a in array:
            self.events.append(self.Event(*a))
    
    def addEvent(self, *args):
        '''Adds any number of events'''
        self.events += args

    def reInitPositions(self):
        '''
        Same as initPosition, only it uses stored values
        '''
        return self.initPositions(self.data, self.screen_size)

    def initPositions(self, data, screen_size):
        '''
        This goes through all stimulus and data and determines what their actual
        positions were. You see, when the stimulus change, it uses the last known head
        distance to determine the position of the stimulus (to get it in degrees). This
        is why this must be gone through once before. Otherwise we couldn't do free
        seeking in the recording.

        data: the same format as Engine.data
        screen_size: two-sized tuple
        '''

        # Save copies
        self.data = data
        self.screen_size = screen_size

        c = len(data)
        self.positions = np.array([0, 0]*c).reshape(c, 2)
        self.active = np.array([0]*c).reshape(c, 1)
        last_pos = None

        head_distance = 700.0 # Default to 700 mm
        
        # 

        for tick, d in enumerate(data):
            t = d[0]
            event = self.getEvent(t)
            pos_angles = None
            if event:
                pos_angles = event.pos
                
            if len(d) >= 4 and d[3] > 0 and (last_pos is None or pos_angles != last_pos):
                head_distance = d[3]
            last_pos = copy(pos_angles)

            if pos_angles is None:
                self.positions[tick] = [-1000, -1000]
            else:
                pos_mm = map(lambda theta: head_distance * np.tan(theta * np.pi/180.0), pos_angles)
                pos_pixels = [	
                    screen_size[i]//2 + pos_mm[i]
                for i in [0,1]]

                self.positions[tick] = pos_pixels
            
            self.active[tick] = 0
            if event:
                self.active[tick] = int(event.active)

    def setDeltaTime(self, delta_time):
        self.deltaTime = delta_time

    def getPosition(self, loop_count):
        assert self.positions is not None
        return self.positions[loop_count]

    def getEvent(self, time_ms):
        '''
        Get the position at a given time in milliseconds
        Returns None means it should not be drawn
        '''
        if not len(self.events):
            return None
            
        cur_time = -self.deltaTime
        for event in self.events:
            cur_time += event.duration_ms
            if time_ms <= cur_time:
                return event
        
        return None
        
    def getTotalActiveEventTime(self):
        '''
        Returns the total time of all active events in milliseconds
        '''
        total = 0.0
        for event in self.events:
            if event.active:
                total += event.duration_ms
        return total

    def loadFromTextFile(self, filename):
        '''
        Load the stimulus for a simple file where you specify different events as:
        
         Static: dur=1000.0; pos=(0,0)
         Hide: dur=3000.0

         Smooth: dur=1000.0; from=(0,0); to=(-10,0) <- not implemented yet
        '''
        f = open(filename)
        
        self.events = []
        sub = r'\s*[\d\w]+\s*=\s*[^;]*'
        parser = re.compile(r'(?P<command>\w+):(?P<vars>{0}(?:;{0})*)'.format(sub))
        for line in f:
            line = line.split('#')[0]
            command = None
            d = {}

            res = parser.search(line)
            if res:
                command = res.group('command')
                variables = res.group('vars').split(';')
                
                for v in variables:
                    key, value = v.split('=')
                    d[key.strip()] = eval(value.strip())
            
                event = None
                if command == 'hide':
                    event = Stimulus.Event(d['dur'])
                elif command == 'static':
                    event = Stimulus.Event(d['dur'], d['pos'])
                else:
                    raise Exception('Command not found: ' + command)
                
                if 'active' in d:    
                    event.active = d['active']
                    
                self.events.append(event)

    def numActiveEvents(self):
        def isActive(event): return int(event.active)
        return len(filter(isActive, self.events))
            

    # def __reduce__(self):
    #     s = ""
    #     for e in self.events:
            