import numpy as np
import random


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
        self.state = np.zeros([len(ships)+1, dim, dim], dtype=int)
        self.done = False

        self._place()

        # print(self.state.shape)
        # print(self.state[1,:,:])
        # print(self.state[2,:,:])
        # print(self.state[3,:,:])

    def _place(self):
        for n,size in enumerate(self.ships):
            success = False
            while not success:
                row = random.randint(0, self.dim-1)
                col = random.randint(0, self.dim-1)
                if random.randint(0,1) == 0: 
                    if col + size > self.dim: 
                        continue 
                    if np.sum(self.state[:,row, col:col+size]) > 0:
                        continue
                    self.state[n+1,row, col:col+size] = 1
                    self.ship_coords[n+1] = ((row,col), (1,size))
                    success = True
                else:
                    if row + size > self.dim:
                        continue 
                    elif np.sum(self.state[:,row:row+size, col]) > 0:
                        continue
                    self.state[n+1,row:row+size, col] = 1
                    self.ship_coords[n+1] = ((row,col), (size,1))
                    success = True

    def reset(self):
        """TODO: Docstring for reset.

        :returns: TODO

        """
        dim = self.dim
        self.shots = 0
        self.num_sunk = 0
        self.ship_coords = {}
        self.state = np.zeros([len(ships)+1, dim, dim], dtype=int)
        self.done = False

        self._place()
    
    def get_state(self):
        """TODO: Docstring for get_state.
        :returns: TODO

        """
        return self.state.copy(), 0, 0, self.done

    def step(self, guess):
        """TODO: Docstring for step.

        :guess: TODO
        :returns: TODO

        """

        reward = -10
        x,y = guess
        hit, sunk = False, False

        # update guesses
        if self.state[0,x,y] == 0:
            self.shots += 1
            self.state[0,x,y] = -1

            # update hits
            for i in range(1, len(self.ships)+1):
                if self.state[i,x,y] == 1:
                    self.state[0,x,y] = 1
                    reward = 10
                    hit = True
                    # update sunk
                    sunk = True
                    pos = self.ship_coords[i]
                    # print(pos)
                    for row in range(pos[0][0], pos[0][0] + pos[1][0]):
                        for col in range(pos[0][1], pos[0][1] + pos[1][1]):
                            # print(row,col)
                            if self.state[i,row,col] == 0:
                                sunk = False
                    if sunk == True:
                        reward = 100 
                        self.num_sunk += 1
                    
                        # update game_state
                        self.done =  (self.num_sunk == len(self.ship_coords))
                        if self.done:
                            reward = 1000
            
        # hits = np.bitwise_and(self.guesses, (self.placement > 0).astype(int))
        # print('placement')
        # print(self.placement)
        # print('guesses')
        # print(self.guesses)
        # print('hits')
        # print(hits)
        return reward, (self.state.copy(), hit, sunk, self.done)


    def __str__(self):
        result = '\n%s\n'%(self.name)
        result += "Shots %d Sunk %d Left %d\n"%(self.shots, self.num_sunk, len(self.ship_coords) - self.num_sunk)
        result += "-------------------\n"
        for row in range(self.dim):
            for col in range(self.dim):
                if self.state[0,row,col] == 1:
                    result += '★ '
                elif np.sum(self.state[:,row,col]) > 0:
                   result += 'S '
                elif self.state[0,row,col] == -1:
                    result += 'X '    
                else:
                    result += 'o '
            result += '\n'
        result += "-------------------\n"
        return result




        
