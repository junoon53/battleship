import numpy as np
import random


class ModelRandom():
    def __init__(self, name, dim):
        """TODO: Docstring for __init__.

        :arg1: TODO
        :returns: TODO

        """
        self.name = name
        self.dim = dim

    def remember(self, state, action, reward, next_state):
        """TODO: Docstring for remember.
        :returns: TODO

        """

    def replay(self, batch_size):
        """TODO: Docstring for replay.
        :returns: TODO

        """
        pass


    def move(self, state):

        d = self.dim

        x = random.randint(0,d-1)
        y = random.randint(0,d-1)

        return x,y

    
    def __str__(self):

        return "%s (Random)"%(self.name)

