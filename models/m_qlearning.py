from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random


class ModelQLearning(nn.Module):
    def __init__(self, name, dim, num_ships, device):
        super(ModelQLearning, self).__init__() 
        """Initialize the QLearning Model
        """
        self.name = name
        self.dim = dim
        self.device = device

## DQN Parameters

        self.alpha = 0.001
        self.gamma = 0.99
        self.epsilon = 0.0
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

## Q-Function Model

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, num_ships+1, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_ships+1)
        self.fc = nn.Linear(dim*dim*(num_ships+1), dim*dim) 

        self.softmax = nn.LogSoftmax()

        self.criterion = nn.NLLLoss()
        
    def forward(self, x):
        """Calculate the forward pass
        :returns: action scores

        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        logits = self.fc(x.view(x.size(0), -1)) 

        return logits, self.softmax(logits)

    def move(self, state):
        """ Obtain the next action 
        :returns: Tuple of x,y coordinates
        """

        d = self.dim
        inputs,open_locations,_,_,_ = state
        open_locations = open_locations.flatten()

        if np.random.rand() <= self.epsilon:

            idx = random.choice([ i for i in range(d*d) if open_locations[i] == 1 ]) 
            x,y = divmod(idx,d)

        else:

            # self.eval()
            inputs = inputs[[0], :, :]
            inputs = torch.Tensor(inputs).unsqueeze(0).to(self.device)
            logits, logprobs = self.forward(inputs)
            logprobs = logprobs[0].detach().cpu().numpy() # + np.random.random(d*d)*1e-8
            max_idx = np.argmax(logprobs + open_locations*1e30)
            x,y = divmod(max_idx.item(),d)

        return x,y

    def calc_rewards(self, hits, total_ships_lengths):
        ''' Calculate the discounted sume of rewards over an episode '''

        gamma = self.gamma
        board_size = self.dim**2

        weighted_hits = []
        for i,hit in enumerate(hits):
            weighted_hit = (hit - float(total_ships_lengths - sum(hits[:i])) / float(board_size - i)) * (gamma ** i)
            weighted_hits.append(weighted_hit)
        weighted_rewards = []
        for i in range(len(hits)):
            discounted_reward = ((gamma) ** (-i)) * sum(weighted_hits[i:])
            weighted_rewards.append(discounted_reward)

        return weighted_rewards


    def replay(self, inputs, actions, hits, total_ships_lengths):
        ''' Replay an episode and train the model '''

        discounted_rewards = self.calc_rewards(hits, total_ships_lengths)

        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=0.0000) 
        self.train()
        for inputs, action, reward in zip(inputs, actions, discounted_rewards):

            action_idx = action[0]*self.dim + action[1]
            action_idx = torch.LongTensor([action_idx]).to(self.device)

            inputs = inputs[[0], :, :]
            inputs = torch.Tensor(inputs).unsqueeze(0).to(self.device)
            # print(inputs.size())
            optimizer.zero_grad()
            logits, logprobs = self.forward(inputs)
            lr = reward*self.alpha
            # print(lr)

            loss = lr*self.criterion(logprobs, action_idx)

            # print(loss)

            loss.backward()

            optimizer.step()

## decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *=self.epsilon_decay

    def __str__(self):

        return "%s (QLearning)"%(self.name)

