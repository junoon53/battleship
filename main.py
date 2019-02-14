import torch
from collections import deque
import copy
from game import Game
from models.m_random import ModelRandom
from models.m_hunt_target import ModelHuntTarget
# from models.m_regression import ModelRegression
from models.m_qlearning import ModelQLearning
from environment import Environment


def main():
    """Entry point of the application.

    :returns: None

    """

    # test()

    DIM = 5
    SHIPS = [2,2]

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
    """Train a Q-Learning model.
    :returns: None

    """
    agent = ModelQLearning("Vikram", DIM, len(SHIPS))
    env = Environment(DIM, SHIPS, "Vikram")
    batch_size = 64
    num_episodes = 90000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print("using gpu")

    agent.to(device) 
    total_moves = 0

    for e in range(num_episodes):
        env.reset()
        state = env.get_state()
        inputs = []
        actions = []
        hits = []
        for time in range(500):
            action = agent.move(state)
            reward, next_state = env.step(action)
            next_input, hit, sunk, done = next_state
            inputs.append(next_input)
            actions.append(action)
            hits.append(hit)
            state = next_state
            if done == True:
                # print("episode: {}/{}, score: {}, e: {:.2}" .format(e, num_episodes, time, agent.epsilon))
                total_moves += len(hits)
                if e % 50 == 0 and e != 0:
                    print(e,float(total_moves)/float(50))
                    total_moves = 0
                break

        # if len(agent.experiences) > batch_size:
        # print('replaying...')
        agent.replay(inputs, actions, hits, env.total_ships_lengths)

def test():

    model_random = m_random.ModelRandom("Vikram")
    env = Environment(10, [2,3,3,4,5], "Vikram")

    for i in range(90):
        model_random.move(env)


if __name__ == "__main__":
    main()
