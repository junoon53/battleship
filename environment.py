import numpy as np
import random


class Environment():
    def __init__(self, dim, ships, name):
        """Docstring for __init__.

        """
        self.name = name
        self.shots = 0
        self.dim = dim
        self.ships = ships
        self.num_sunk = 0
        self.ship_coords = {}
        self.state = np.zeros([len(ships)+1, dim, dim], dtype='long')
        self.placement = np.zeros([len(ships), dim, dim], dtype=int)
        self.open_locations = np.full([dim, dim], 1, 'float32')
        self.done = False
        self.total_ships_lengths = sum(ships)

        self._place()

        # print(self.placement)

    def _place(self):
        ''' Places ships on the grid
        '''
        for n,size in enumerate(self.ships):
            success = False
            while not success:
                row = random.randint(0, self.dim-1)
                col = random.randint(0, self.dim-1)
                if random.randint(0,1) == 0: 
                    if col + size > self.dim: 
                        continue 
                    if np.sum(self.placement[:,row, col:col+size]) > 0:
                        continue
                    self.placement[n, row, col:col+size] = 1
                    self.ship_coords[n] = ((row,col), (1,size))
                    success = True
                else:
                    if row + size > self.dim:
                        continue 
                    elif np.sum(self.placement[:,row:row+size, col]) > 0:
                        continue
                    self.placement[n,row:row+size, col] = 1
                    self.ship_coords[n] = ((row,col), (size,1))
                    success = True

    def reset(self):
        """Resets the environment
        """
        dim = self.dim
        self.shots = 0
        self.num_sunk = 0
        self.ship_coords = {}
        self.state = np.zeros([len(self.ships)+1, dim, dim], dtype='float32')
        self.placement = np.zeros([len(self.ships), dim, dim], dtype='int')
        self.open_locations = np.full([dim, dim], 1, 'float32')
        self.done = False

        self._place()
    
    def get_state(self):
        """Get the latest state from the environment
        :returns: A tuple containing the input state, hit (bool), sunk (bool) and done (bool)

        """
        return self.state.copy(), self.open_locations.copy(), 0, 0, self.done

    def step(self, guess):
        """ Perform an action on the environment
        :guess: The action returned by the state
        :returns: A tuple containing the reward, input state, hit (bool), sunk (bool) and done (bool)

        """
        reward = -1
        x,y = guess
        hit, sunk = 0, 0

        assert(self.state[0,x,y] == 0)

        # update guesses
        if self.state[0,x,y] == 0:
            self.shots += 1
            self.state[0,x,y] = -1
            self.open_locations[x,y] = 0

            # update hits
            for i in range(0, len(self.ships)):
                if self.placement[i,x,y] == 1:
                    self.state[0,x,y] = 1
                    reward = 1
                    hit = 1
                    # update sunk
                    sunk = 1
                    pos = self.ship_coords[i]
                    # print(pos)
                    for row in range(pos[0][0], pos[0][0] + pos[1][0]):
                        for col in range(pos[0][1], pos[0][1] + pos[1][1]):
                            # print(row,col, self.state[0,row,col])
                            if self.state[0,row,col] == 0:
                                sunk = 0
                    if sunk == 1:
                        reward = 1 
                        self.num_sunk += 1
                        self.state[i+1,:,:] = 1
                        # print('sunk', self.num_sunk)
                        # print(self.state)
                    
                        # update game_state
                        self.done =  (self.num_sunk == len(self.ship_coords))
                        if self.done:
                            reward = 1

                    break

        return reward, (self.state.copy(), self.open_locations.copy(), hit, sunk, self.done)


    def __str__(self):
        result = '\n%s\n'%(self.name)
        result += "Shots %d Sunk %d Left %d\n"%(self.shots, self.num_sunk, len(self.ship_coords) - self.num_sunk)
        result += "-------------------\n"
        for row in range(self.dim):
            for col in range(self.dim):
                if self.state[0,row,col] == 1:
                    result += '★ '
                elif np.sum(self.placement[:,row,col]) > 0:
                   result += 'S '
                elif self.state[0,row,col] == -1:
                    result += 'X '    
                else:
                    result += 'o '
            result += '\n'
        result += "-------------------\n"
        return result




        
