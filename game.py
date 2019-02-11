import random


class Game(object):

    """Docstring for Game. """

    def __init__(self, player_a, player_b, board_a, board_b):
        """Init function for the Game Class """

  
        self.players = [player_a, player_b]
        self.boards = [board_a, board_b]


    def play(self):
        """Play a game
        :returns: TODO

        """
        if random.random() > 0.5:
            self.players.reverse()
            self.boards.reverse()


        while True:

            self.boards[0].step(self.players[1].move(self.boards[0]))
            if self.boards[0].done == True:
                break
            self.boards[1].step(self.players[0].move(self.boards[1]))
            if self.boards[1].done == True:
                break

            print(self.boards[0])
            print(self.boards[1])

        print(self.boards[0])
        print(self.boards[1])

        if self.boards[1].done:
            print("%s wins!!"%(self.players[0]))
        else:
            print("%s wins!!"%(self.players[1]))
