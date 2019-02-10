from models import m_random
from environment import Environment


def main():
    """TODO: Docstring for main.

    :returns: TODO

    """
    model_random = m_random.ModelRandom(None)
    env = Environment(10, [2,3,3,4,5])

    for i in range(10):
        env.step(model_random.move(env))

    print(env)

if __name__ == "__main__":
    main()
