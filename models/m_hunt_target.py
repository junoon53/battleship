import numpy as np
import random


class ModelHuntTarget():
    def __init__(self, name, dim):
        """Initialize the ModelHuntTarget class.
        """
        self.name = name
        self.dim = dim
        self.mode = 0 # 0=hunt, 1=target
        self.first_hit = None
        self.last_hit = None
        self.last_action = None
        self.radius = 1
        self.target_idx = 0

    def target(self):

        d = self.dim
        R = self.radius

        x = random.randint(max(0,self.last_hit[0]-R), min(d-1,self.last_hit[0]+R))
        y = random.randint(max(0,self.last_hit[1]-R), min(d-1,self.last_hit[1]+R))
        
        self.target_idx += 1
        if self.target_idx >= self.radius**2:
            self.radius += 1
            self.target_idx = 0

        return x,y

    def move(self, state):

        state, hit, sunk, done = state

        if self.mode == 0:

            if hit == 1 and sunk == 0:
                self.mode = 1
                self.first_hit = self.last_action
                self.last_hit = self.last_action
                self.radius = 1
                self.target_idx = 0

        elif self.mode == 1:

            if hit == 1 and sunk == 0:
                self.mode = 1
                self.last_hit = self.last_action
            elif hit == 1 and sunk == 1:
                self.mode = 0
                self.first_hit = None
                self.last_hit = None

        d = self.dim

        if self.mode == 0:

            x = random.randint(0,d-1)
            y = random.randint(0,d-1)
            self.last_action = (x,y)
            return x,y

        if self.mode == 1:
            
            x,y = self.target()
            self.last_action = (x,y)
            return x,y

        
    def __str__(self):

        return "%s (Hunt-Target)"%(self.name)

