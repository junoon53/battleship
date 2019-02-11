from game import Game
from models.m_random import ModelRandom
from models.m_hunt_target import ModelHuntTarget
from environment import Environment


def main():
    """TODO: Docstring for main.

    :returns: TODO

    """

    # test()

    DIM = 10
    SHIPS = [2,3,3,4,5]

    g = Game(ModelHuntTarget("Vikram"), ModelRandom("Betal"), Environment(DIM, SHIPS, "Vikram"), Environment(DIM, SHIPS, "Betal"))

    g.play()

def test():

    model_random = m_random.ModelRandom("Vikram")
    env = Environment(10, [2,3,3,4,5], "Vikram")

    for i in range(90):
        model_random.move(env)


if __name__ == "__main__":
    main()
