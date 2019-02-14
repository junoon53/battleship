import random

class Game(object):

    """Constructs a two player game"""

    def __init__(self, player_a, player_b, board_a, board_b, verbose=True):
        """Init function for the Game Class """

        self.players = [player_a, player_b]
        self.boards = [board_a, board_b]
        self.verbose = True
        self.batch_size = 32

    def play(self):
        """Play a game
        :returns: Nothing

        """
        if random.random() > 0.5:
            self.players.reverse()
            self.boards.reverse()

        rounds = 1
        states = [ self.boards[0].get_state(), self.boards[1].get_state() ] 
        winner = None
        while True:

            action = self.players[1].move(states[0])
            reward, state = self.boards[0].step(action)
            x, hit, sunk, done = state
            states[0] = state

            if done == True:
                winner = self.players[1]
                break

            action = self.players[0].move(states[1])
            reward, state = self.boards[1].step(action)
            x, hit, sunk, done = state
            states[1] = state

            if done == True:
                winner = self.players[0]
                break

            rounds += 1

            if self.verbose:
                print("Round %d"%(rounds))
                print(self.boards[0])
                print(self.boards[1])

        if self.verbose:
            print("Round %d"%(rounds))
            print(self.boards[0])
            print(self.boards[1])
        

        print("%s wins!!"%(winner))
