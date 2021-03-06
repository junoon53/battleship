import torch
import numpy as np
from collections import deque
import copy
from game import Game
from models.m_random import ModelRandom
from models.m_hunt_target import ModelHuntTarget
from models.m_convnet import ModelConvnet
from models.m_qlearning import ModelQLearning
from environment import Environment


def main():
    """Entry point of the application.

    :returns: None

    """

    # test()

    DIM = 10
    SHIPS = [2,3,3,4,5]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = train_Convnet(DIM, SHIPS, device)
    torch.save(model.state_dict(),'./saved_models/convenet.torch' )

    conv_model = ModelConvnet("Vikram", DIM, len(SHIPS), device).to(device)
    conv_model.load_state_dict(torch.load('./saved_models/convenet.torch'))

    g = Game(conv_model, ModelRandom("Betal", DIM, len(SHIPS), device), Environment(DIM, SHIPS, "Vikram"), Environment(DIM, SHIPS, "Betal"))
    g.play()


def tournament(players):
    '''Conduct a tournament between a set of players'''
    DIM = 5
    SHIPS = [2,2]

    winners = players

    while len(winners) > 1:

        player1 = winners.pop()
        player2 = winners.pop()

        g = Game(player1, player2, Environment(DIM, SHIPS, "player1"), Environment(DIM, SHIPS, "players"))

        winners.append(g.play())


    print("%s wins the tournament"%(winners[0]))

def train_Convnet(DIM, SHIPS, device):
    """Train a convnet model.
    :returns: None

    """
    agent = ModelConvnet("Vikram", DIM, len(SHIPS), device)
    agent.to(device)
    env = Environment(DIM, SHIPS, "Vikram")
    batch_size = 1024
    num_episodes = 1000
    max_running_avg = 64

    batch = 0
    total_moves = 0

    inputs = np.empty([batch_size, 1, DIM, DIM])
    labels = np.empty([batch_size, DIM, DIM])

    for e in range(num_episodes):
        env.reset()
        state = env.get_state()
        done = False
        episode_moves = 0

        for time in range(DIM*DIM):
            action = agent.move(state)
            episode_moves += 1
            reward, next_state = env.step(action)
            next_input, open_locations, hit, sunk, done = next_state
            inputs[batch, :, :] = next_input[0, :, :]
            labels[batch, :, :] = env.get_ground_truth()

            if done == True:
                total_moves += episode_moves
                episode_moves = 0
                if e % max_running_avg == 0 and e != 0:
                    print("Episodes: {}, Avg Moves: {}".format(e,float(total_moves)/float(max_running_avg)))
                    total_moves = 0

                break
            
           
            batch += 1

            if batch == batch_size:
                agent.replay(inputs, labels)
                batch = 0
 
            state = next_state

        if done == False:
            print(env.placement)
            print(inputs,actions, hits)
            # break

    return agent

def test():

    model_random = m_random.ModelRandom("Vikram")
    env = Environment(10, [2,3,3,4,5], "Vikram")

    for i in range(90):
        model_random.move(env)

def train_RL(DIM, SHIPS):
    """Train a Q-Learning model.
    :returns: None

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = ModelQLearning("Vikram", DIM, len(SHIPS), device)
    env = Environment(DIM, SHIPS, "Vikram")
    batch_size = 64
    num_episodes = 100


    total_moves = 0

    for e in range(num_episodes):
        env.reset()
        state = env.get_state()
        inputs = []
        actions = []
        hits = []
        done = False
        for time in range(DIM*DIM):
            action = agent.move(state)
            reward, next_state = env.step(action)
            next_input, open_locations, hit, sunk, done = next_state
            if done == True:
                total_moves += len(hits)
                if e % batch_size == 0 and e != 0:
                    print("Episodes: {}, Avg Moves: {}".format(e,float(total_moves)/float(batch_size)))
                    total_moves = 0

                agent.replay(inputs, actions, hits, env.total_ships_lengths)
                break

            inputs.append(next_input)
            actions.append(action)
            hits.append(hit)
            state = next_state

        if done == False:
            print(env.placement)
            print(inputs,actions, hits)
            # break


def test():

    model_random = m_random.ModelRandom("Vikram")
    env = Environment(10, [2,3,3,4,5], "Vikram")

    for i in range(90):
        model_random.move(env)


if __name__ == "__main__":
    main()
