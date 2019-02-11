import numpy as np
import random


class ModelRandom():
    def __init__(self, name):
        """TODO: Docstring for __init__.

        :arg1: TODO
        :returns: TODO

        """
        self.name = name


    def move(self, env):

        d = env.dim

        x = random.randint(0,d-1)
        y = random.randint(0,d-1)

        while env.guesses[x,y] == 1:
        
            x = random.randint(0,d-1)
            y = random.randint(0,d-1)

        return x,y

    
    def __str__(self):

        return "%s (Random)"%(self.name)

