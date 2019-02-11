import numpy as np
import sys


class Environment():
    def __init__(self, dim, ships, name):
        """TODO: Docstring for __init__.
        :returns: TODO

        """
        self.name = name
        self.shots = 0
        self.dim = dim
        self.ships = ships
        self.num_sunk = 0
        self.ship_coords = {}
        self.placement = np.zeros([dim, dim], dtype=int)
        self.guesses = np.zeros([dim, dim], dtype=int)
        self.hits = np.zeros([dim, dim], dtype=int)
        self.done = 0

        self._place()

    def _place(self):
        for n,size in enumerate(self.ships):

            placed = False
            while not placed:
                row = np.random.randint(self.dim)
                col = np.random.randint(self.dim)
                if np.random.random() > 0.5:
                    # try to place it horizontally
                    if col + size > self.dim: 
                        continue # off the board

                    elif np.sum(self.placement[row, col:col+size]) > 0:
                        continue # collision

                    self.placement[row, col:col+size] = n+1
                    self.ship_coords[n+1] = ((row,col), (1,size))
                    placed = True
                else:
                    # try to place it vertically
                    if row + size > self.dim:
                        continue # off the board

                    elif np.sum(self.placement[row:row+size, col]):
                        continue # collision

                    self.placement[row:row+size, col] = n+1
                    self.ship_coords[n+1] = ((row,col), (size,1))
                    placed = True

    def step(self, guess):
        """TODO: Docstring for step.

        :guess: TODO
        :returns: TODO

        """

        reward = 0
        x,y = guess
        hit, sunk, done = 0,0,0

        # update guesses
        if self.guesses[x,y] == 0:
            self.shots += 1
            self.guesses[x,y] = 1

            # update hits
            if self.placement[x,y] > 0:
                self.hits[x,y] = 1
                reward = 1
                hit = 1
                # update sunk
                ship_no = self.placement[x,y]
                sunk = 1
                pos = self.ship_coords[ship_no]
                # print(pos)
                for row in range(pos[0][0], pos[0][0] + pos[1][0]):
                    for col in range(pos[0][1], pos[0][1] + pos[1][1]):
                        # print(row,col)
                        if self.hits[row,col] == 0:
                            sunk = 0
                if sunk == 1:
                    reward = 10 
                    self.num_sunk += 1
                
                    # update game_state
                    if self.num_sunk == len(self.ship_coords):
                        self.done = 1
                        reward = 100
            else:
                reward = -1

        return reward, hit, sunk, done


    def __str__(self):
        result = '\n%s\n'%(self.name)
        result += "Shots %d Sunk %d Left %d\n"%(self.shots, self.num_sunk, len(self.ship_coords) - self.num_sunk)
        result += "-------------------\n"
        for row in range(self.dim):
            for col in range(self.dim):
                if self.placement[row,col] and self.hits[row,col]:
                    result +='☀ '
                elif self.placement[row,col]:
                    result += '%d '%(self.placement[row,col])
                elif self.guesses[row,col]:
                    result += 'X '    
                else:
                    result += 'o '
            result += '\n'
        result += "-------------------\n"
        return result




        
