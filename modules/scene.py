"""
Contains Python classes for controlling and updating global attributes.
"""
class World:
    """
    Class for gobal world attributes.
    """
    def __init__(self, data):
        self.data = data  # keep data in world for now
        self.scene = {}

    def create(self, dims):
        """
        Create plotting environment for plotly.
        dims : [x, y, z]
        """
        self.scene["xaxis"] = {"autorange" : False, "range" : [-dims[0], dims[0]]}
        self.scene["yaxis"] = {"autorange" : False, "range" : [-dims[1], dims[1]]}
        self.scene["zaxis"] = {"autorange" : False, "range" : [-dims[2], dims[2]]}


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

    def set(self, time):
        """
        Updates the current time
        """
        self.now = time
