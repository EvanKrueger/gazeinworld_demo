"""
Contains TimeMgr class for controlling and updating global time.
"""


class TimeMgr:
    """
    Class for controlling global time in GazeInWorld_Demo visualization and data analysis
    Note: only one instance of TimeMgr should be created in the module scope.
    """

    def __init__(self, data):
        """
        Initializes the current time and addressable duration of time attributes.
        """
        self.now = 0
        self.history = len(data)

    def setTime(self, time):
        """
        Updates the current time
        """
        self.now = time
