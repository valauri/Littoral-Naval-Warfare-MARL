import numpy as np
#import matplotlib.pyplot as plt
#from collections import defaultdict
#from matplotlib.patches import Rectangle
#from matplotlib.patches import Circle
#from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import json
import os.path
import random
from IPython.display import clear_output
from time import sleep
#from collections import namedtuple, deque
import math
#from astar.search import AStar
#from skimage.draw import line
from torch.distributions import Normal
from torchviz import make_dot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
class MA_MLP(n_agents, n_inputs, n_outputs, centralised=True, depth=2):
    def __init__(self, n_agents, n_inputs, n_outputs, centralised=centralised, depth=depth):
    
        self.mlp = MultiAgentMLP(n_agents, n_inputs, n_outputs, centralised=centralised, depth=depth)

    def get_mlp(self):
        return self.mlp
"""

class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1, padding=1)
        self.norm1 = nn.BatchNorm2d(5)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(5, 8, 3, 1, padding=1)
        self.norm2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2,2)

        self.convhead = nn.Linear(8, 12, bias=True)

        self.layernorm = nn.LayerNorm(n_inputs)

        self.fc1 = nn.Linear(n_inputs, 64, bias=True) #128
        self.fc2 = nn.Linear(64, 64, bias=True)
        self.fc3 = nn.Linear(64, 32, bias=True)

        self.normal_head = nn.Linear(32, n_outputs, bias=False)
        self.log_std_head = nn.Linear(32, n_outputs, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.normal_head.weight)
        nn.init.xavier_uniform_(self.log_std_head.weight)

        self.logstd = torch.nn.Parameter(torch.log(torch.tensor(1))).to(device)

    def forward(self, x, noise=None):
        if len(x.shape) == 1:
            z = x.unsqueeze(0)
            batch_size = 1
        else:
            z = x
            batch_size = x.shape[0]

        z = z[:, :49].unsqueeze(1).view(batch_size, 1, 7, 7)

        if len(x.shape) == 1:
            x = x[49:]
        else:
            x = x[:, 49:]

        z = self.pool(F.relu(self.norm1(self.conv1(z))))
        z = self.pool2(F.relu(self.norm2(self.conv2(z))))
        z = torch.flatten(z, 1)

        z = self.convhead(z)

        x = torch.concat((torch.flatten(z), x), 0)

        x = self.layernorm(x)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        normal_mean = torch.tanh(self.normal_head(x))
        normal_std = torch.exp(self.log_std_head(x))

        nancheck_std = torch.isnan(normal_std)
        nancheck_mean = torch.isnan(normal_mean)

        if torch.any(nancheck_std) or torch.any(nancheck_mean):
            return None, None
        
        dist = Normal(normal_mean, normal_std)
        actions = dist.sample()

        if noise is not None:
            actions +=  torch.tensor(np.random.normal(0, noise.cpu(), size=actions.shape), dtype=torch.float32, device = device)

        actions = torch.clamp(actions, 0, 1)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return actions, log_probs

    def get_dist(self, x, actions):
        if len(x.shape) == 1:
            z = x.unsqueeze(0)
            batch_size = 1
        else:
            z = x
            batch_size = x.shape[0]

        z = z[:, :49].unsqueeze(1).view(batch_size, 1, 7, 7)

        if len(x.shape) == 1:
            x = x[49:]
        else:
            x = x[:, 49:]

        z = self.pool(F.relu(self.norm1(self.conv1(z))))
        z = self.pool2(F.relu(self.norm2(self.conv2(z))))
        z = torch.flatten(z, 1)

        z = self.convhead(z)

        x = torch.concat((torch.flatten(z), x), 0)

        x = self.layernorm(x)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        normal_mean = torch.tanh(self.normal_head(x))
        normal_std = torch.exp(self.log_std_head(x))

        dist = Normal(normal_mean, normal_std)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        return log_probs, entropies

class Value(nn.Module):
    def __init__(self, n_inputs):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 32, bias=True)
        self.fc2 = nn.Linear(32, 64, bias=True)
        self.fc3 = nn.Linear(64, 64, bias=True)
        self.fc4 = nn.Linear(64, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        value = self.fc4(x)

        return value

class DDQN(nn.Module):

    def __init__(self, n_actions):
        super(DDQN, self).__init__()
        # 
        
        self.conv1 = nn.Conv2d(5,10,2,1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,20,2,1)
        self.bn = nn.BatchNorm2d(10)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 40, 2, 1)
        self.bn2 = nn.BatchNorm2d(40)
        
        self.fc0 = nn.Linear(4840, 2000)
        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. 
    def forward(self, x):
        
        x = x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn2(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.softmax(self.fc4(x), dim=0)
        x = self.fc4(x)
    
        return x
    
class DMLP_(nn.Module):

    def __init__(self, n_inputs):
        super(DMLP, self).__init__()
        # 
        
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 128)

        self.movement = nn.Linear(128, 7*7+1)
        self.attack = nn.Linear(128, 5)
        self.radar = nn.Linear(128, 2)

    # Called with either one element to determine next action, or a batch
    # during optimization. 
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        movement = self.movement(x)
        attack = self.attack(x)
        radar = self.radar(x)
    
        return radar, attack, movement
    
class DMLP(nn.Module):

    def __init__(self, n_inputs):
        super(DMLP, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, 3, 1, padding=1)
        self.norm1 = nn.BatchNorm2d(5)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(5, 8, 3, 1, padding=1)
        self.norm2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2,2)

        self.convhead = nn.Linear(8, 12, bias=True)

        self.movement = nn.Linear(n_inputs-(7*7)+12, 7*7+1, bias=True) #128
        self.attack = nn.Linear(n_inputs-(7*7)+12, 5, bias=True)
        self.radar = nn.Linear(n_inputs-(7*7)+12, 2, bias=True)

        nn.init.xavier_uniform_(self.convhead.weight)
        nn.init.xavier_uniform_(self.movement.weight)
        nn.init.xavier_uniform_(self.attack.weight)
        nn.init.xavier_uniform_(self.radar.weight)

    # Called with either one element to determine next action, or a batch
    # during optimization. 
    def forward(self, x):
        if len(x.shape) == 1:
            z = x.unsqueeze(0)
            batch_size = 1
        else:
            z = x
            batch_size = x.shape[0]

        z = z[:, :49].unsqueeze(1).view(batch_size, 1, 7, 7)

        if len(x.shape) == 1:
            x = x[49:]
        else:
            x = x[:, 49:]

        z = self.pool(F.relu(self.norm1(self.conv1(z))))
        z = self.pool2(F.relu(self.norm2(self.conv2(z))))
        z = torch.flatten(z, 1)

        z = self.convhead(z)

        if len(x.shape) == 1:
            z = z.squeeze()

            x = torch.concat((z, x), 0)

        else:
            x = torch.cat((z, x), 1)

        movement = F.relu(self.movement(x))
        attack = F.relu(self.attack(x))
        radar = F.relu(self.radar(x))
    
        return radar, attack, movement

