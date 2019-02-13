import torch
from game import Game
from models.m_random import ModelRandom
from models.m_hunt_target import ModelHuntTarget
# from models.m_regression import ModelRegression
from models.m_qlearning import ModelQLearning
from environment import Environment


def main():
    """TODO: Docstring for main.

    :returns: TODO

    """

    # test()

    DIM = 6
    SHIPS = [3,3,3]

    # g = Game(ModelHuntTarget("Vikram", DIM), ModelRandom("Betal", DIM), Environment(DIM, SHIPS, "Vikram"), Environment(DIM, SHIPS, "Betal"))
    # g.play()

    # tournament([ModelHuntTarget("Vikram"),ModelHuntTarget("Sacchita"),ModelRandom("Betal")])

    train(DIM, SHIPS)


def tournament(players):
    '''Conduct a tournament between a set of players'''
    DIM = 5
    SHIPS = [3,3,3]

    winners = players

    while len(winners) > 1:

        player1 = winners.pop()
        player2 = winners.pop()

        g = Game(player1, player2, Environment(DIM, SHIPS, "player1"), Environment(DIM, SHIPS, "players"))

        winners.append(g.play())


    print("%s wins the tournament"%(winners[0]))


def train(DIM, SHIPS):
    """TODO: Docstring for train.
    :returns: TODO

    """
    agent = ModelQLearning("Vikram", DIM)
    env = Environment(DIM, SHIPS, "Vikram")
    batch_size = 32
    num_episodes = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print("using gpu")

    agent.to(device) 


    for e in range(num_episodes):
        env.reset()
        state = env.get_state()
        for time in range(500):
            action = agent.move(state)
            reward, next_state = env.step(action)
            _,_, _, _, done = next_state
            agent.remember(state, action, reward, next_state)
            state = next_state
            # print(time, action, reward, agent.epsilon, end="\r")

            if done == True:
                print("episode: {}/{}, score: {}, e: {:.2}" .format(e, num_episodes, time, agent.epsilon))
                break

            if len(agent.experiences) > batch_size:
                agent.replay(batch_size)

        # print("episode: {}/{}, score: {}, e: {:.2}" .format(e, num_episodes, time, agent.epsilon))

def test():

    model_random = m_random.ModelRandom("Vikram")
    env = Environment(10, [2,3,3,4,5], "Vikram")

    for i in range(90):
        model_random.move(env)


if __name__ == "__main__":
    main()
