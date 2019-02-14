from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random


class ModelConvnet(nn.Module):
    def __init__(self, name, dim, num_ships, device):
        super(ModelConvnet, self).__init__() 
        """Initialize the Convnet Model
        """

        self.device = device

        self.name = name
        self.dim = dim

        self.epsilon = 0.0
        self.learning_rate = 0.01

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

        self.eval()
        inputs = inputs[[0], :, :]
        inputs = torch.Tensor(inputs).unsqueeze(0).to(self.device)
        logits, logprobs = self.forward(inputs)
        logprobs = logprobs[0].detach().cpu().numpy() # + np.random.random(d*d)*1e-8
        max_idx = np.argmax(logprobs + open_locations*1e5)
        x,y = divmod(max_idx.item(),d)

        return x,y

    def replay(self, inputs, labels):
        ''' Replay an episode and train the model '''

        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=0.000) 
        batch_size = len(inputs)
        minibatch_size = 128
        samples = 0

        self.train()

        while samples < 10*batch_size:

                samples += minibatch_size
                idxs = np.random.randint(0, 1024, [minibatch_size])
                
                input_mbatch = inputs[idxs, :, :]
                label_mbatch = labels[idxs, :, :].reshape([minibatch_size, -1])

                input_mbatch = torch.Tensor(input_mbatch).to(self.device)
                label_mbatch = torch.Tensor(label_mbatch).to(self.device)

                optimizer.zero_grad()
                logits, logprobs = self.forward(input_mbatch)
                loss =  torch.mean(torch.sum(- label_mbatch * logprobs, 1))

                loss.backward()
                optimizer.step()

    def __str__(self):

        return "%s (Convnet)"%(self.name)

