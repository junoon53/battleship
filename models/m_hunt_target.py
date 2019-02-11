import numpy as np
import random


class ModelHuntTarget():
    def __init__(self, name):
        """TODO: Docstring for __init__.

        :arg1: TODO
        :returns: TODO

        """
        self.name = name
        self.mode = 0 # 0=hunt, 1=target
        self.first_hit = None
        self.last_hit = None
        self.orientation = 0 # 0=horizontal, 1=vertical

    def target(self, R, env):

        d = env.dim

        x = random.randint(max(0,self.last_hit[0]-R), min(d-1,self.last_hit[0]+R))
        y = random.randint(max(0,self.last_hit[1]-R), min(d-1,self.last_hit[1]+R))

        M = R**2 
        i = 0
        while i < M and env.guesses[x,y] == 1:

            x = random.randint(max(0,self.last_hit[0]-R), min(d-1,self.last_hit[0]+R))
            y = random.randint(max(0,self.last_hit[1]-R), min(d-1,self.last_hit[1]+R))
            i += 1

        return x,y

    def move(self, env):

        d = env.dim

        if self.mode == 0:

            x = random.randint(0,d-1)
            y = random.randint(0,d-1)

            while env.guesses[x,y] == 1:
            
                x = random.randint(0,d-1)
                y = random.randint(0,d-1)
            
            reward, hit, sunk, done = env.step((x,y))

            if hit == 1 and sunk == 0:
                self.mode = 1
                self.first_hit = (x,y)
                self.last_hit = (x,y)

        elif self.mode == 1:
            
            for R in range(0,10):
                x,y = self.target(R, env)
                if env.guesses[x,y] == 0:
                    break

            reward, hit, sunk, done = env.step((x,y))

            if hit == 1 and sunk == 0:
                self.mode = 1
                self.last_hit = (x,y)
            elif hit == 1 and sunk == 1:
                self.mode = 0
                self.first_hit = None
                self.last_hit = None

    def __str__(self):

        return "%s (Hunt-Target)"%(self.name)

