from game import Game
from models.m_random import ModelRandom
from environment import Environment


def main():
    """TODO: Docstring for main.

    :returns: TODO

    """

    # test()

    g = Game(ModelRandom("Vikram"), ModelRandom("Betal"), Environment(10, [2,3,3,4,5], "Vikram"), Environment(10, [2,3,3,4,5], "Betal"))
    g.play()

def test():

    model_random = m_random.ModelRandom("Vikram")
    env = Environment(10, [2,3,3,4,5], "Vikram")

    reward = 0
    for i in range(90):
        reward += env.step(model_random.move(env))

    print(env)
    print(reward)

if __name__ == "__main__":
    main()
