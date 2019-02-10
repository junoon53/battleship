import numpy as np
import random


class ModelRandom():
    def __init__(self, arg1):
        """TODO: Docstring for __init__.

        :arg1: TODO
        :returns: TODO

        """
        pass


    def move(self, env):

        d = env.dim

        x = random.randint(0,d-1)
        y = random.randint(0,d-1)

        while env.guesses[x,y] == 1:
        
            x = random.randint(0,d-1)
            y = random.randint(0,d-1)

        return x,y

