from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random


class ModelConvnet(nn.Module):
    def __init__(self, name, dim, num_ships):
        super(ModelConvnet, self).__init__() 
        """Initialize the Convnet Model
        """
        self.name = name
        self.dim = dim

## DQN Parameters

        self.alpha = 0.001
        self.gamma = 0.99
        self.epsilon = 0.0
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

## Q-Function Model

        self.conv1 = nn.Conv2d(num_ships+1, 32, kernel_size=3, stride=1,padding=1)
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
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)

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

            self.eval()
            inputs = torch.Tensor(inputs).unsqueeze(0)
            logits, logprobs = self.forward(inputs)
            logprobs = logprobs[0].detach().numpy() # + np.random.random(d*d)*1e-8
            max_idx = np.argmax(logprobs + open_locations*1e25)
            x,y = divmod(max_idx.item(),d)

        return x,y

    def replay(self, inputs, labels):
        ''' Replay an episode and train the model '''

        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=0.0001) 
        batch_size = len(inputs)
        minibatch_size = 64
        samples = 0

        self.train()

        while samples < batch_size:

                samples += minibatch_size
                idxs = np.random.randint(0, 1024, [minibatch_size])
                
                input_batch = inputs[idxs, :, :, :]
                label_batch = labels[idxs, :, :]
                input = torch.Tensor(input_batch)
                labels = torch.Tensor(label_batch)
                optimizer.zero_grad()
                logits, logprobs = self.forward(input)
                loss =  torch.mean(torch.sum(- labels * logprobs, 2))

                loss.backward()

                optimizer.step()

    def __str__(self):

        return "%s (Convnet)"%(self.name)

