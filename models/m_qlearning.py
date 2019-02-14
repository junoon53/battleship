from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random


class ModelQLearning(nn.Module):
    def __init__(self, name, dim):
        super(ModelQLearning, self).__init__() 
        """TODO: Docstring for __init__.

        :arg1: TODO
        :returns: TODO

        """
        self.name = name
        self.dim = dim

## DQN Parameters

        self.experiences = deque(maxlen=2000)
        self.gamma = 0.5 
        self.epsilon = 0.0
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005

## Q-Function Model

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(3)

        def calc_conv_size(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convd = calc_conv_size(calc_conv_size(calc_conv_size(dim, 3), 3), 1)

        linear_input_size = convd * convd * 3

        print(convd, linear_input_size)

        self.fc = nn.Linear(linear_input_size, dim*dim) 

        self.criterion = nn.MSELoss()
        self.optimizer = None 
        
    def forward(self, x):
        """TODO: Docstring for forward.
        :returns: TODO

        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.fc(x.view(x.size(0), -1))

    def remember(self, state, action, reward, next_state):
        """TODO: Docstring for remember.
        :returns: TODO

        """
        self.experiences.append((state, action, reward, next_state))


    def move(self, state):

        d = self.dim

        if np.random.rand() <= self.epsilon:

            guesses, hits,_,_,_ = state
            guesses = guesses.flatten()
            # print(guesses)
            idx = random.choice([ i for i in range(d*d) if guesses[i] == 0 ]) 
            x,y = divmod(idx,d)
            # print(x,y)

        else:

            guesses, hits,_,_,_ = state
            inputs = np.zeros((1, 2, self.dim, self.dim))
            inputs[0, 0, :] = guesses
            inputs[0, 1, :] = hits
            inputs = torch.Tensor(inputs).to('cuda')
            preds = self.forward(inputs)[0].detach().cpu().numpy()
            closed_positions = guesses.flatten() * (-1001)
            # print(preds, closed_positions, preds + closed_positions)
            max_idx = np.argmax(preds + closed_positions)
            x,y = divmod(max_idx.item(),d)
            # print (x,y)

        return x,y


    def replay(self, batch_size):

        if self.optimizer == None: 
             self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate) 
             self.criterion.to('cuda')

        minibatch = random.sample(self.experiences, batch_size)
        for state, action, reward, next_state in minibatch:

            action_idx = action[0]*self.dim + action[1]

            guesses, hits,_,_,_ = state
            # closed_positions = guesses.flatten() * (-1001)
            # print ("replay", action, reward)
            # print('guesses')
            # print(guesses)
            # print('hits')
            # print(hits)
            inputs = np.zeros((1, 2, self.dim, self.dim))
            inputs[0, 0, :] = guesses
            inputs[0, 1, :] = hits
            inputs = torch.Tensor(inputs)
            inputs = inputs.to('cuda')
            
            next_guesses, next_hits, hit, sunk, done = next_state
            next_closed_positions = next_guesses.flatten() * (-1001)
            next_inputs = np.zeros((1, 2, self.dim, self.dim))
            # print('next guesses')
            # print(next_guesses)
            # print('next hits')
            # print(next_hits)
            next_inputs[0, 0, :] = next_guesses
            next_inputs[0, 1, :] = next_hits
            next_inputs = torch.Tensor(next_inputs)
            next_inputs = next_inputs.to('cuda')
            
## calculate estimate of discounted reward
            label = reward
            self.eval()
            if not done:
                probs = self.forward(next_inputs)
                # print('probs')
                # print(probs)
                probs_max = np.max(next_closed_positions + probs[0].detach().cpu().numpy(), 0)
                # probs_max,_ = torch.max(probs[0], 0)
                label = reward + self.gamma * probs_max
                # print('label', label)

## calculate the labels
            labels = self.forward(inputs)
            labels[0, action_idx] = label
            # print('labels')
            # print(labels)

## update Q(s,a) using gradient descent
            self.train()
            self.optimizer.zero_grad()

            # loss = self.criterion(self.forward(inputs), labels)
            loss = F.smooth_l1_loss(self.forward(inputs), labels)
            # print(loss)
            loss.backward()
            self.optimizer.step()
            # print(loss)

## decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *=self.epsilon_decay

    def __str__(self):

        return "%s (QLearning)"%(self.name)

