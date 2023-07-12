import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os.path
import random
from IPython.display import clear_output
from time import sleep
from game import Game
from collections import namedtuple, deque
import scipy.stats as stats
import math

skip_training = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)



PATH = os.path.join(os.getcwd(), 'models')


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def empty(self):
        self.memory = deque([], maxlen=self.papacity)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))
                        

BATCH_SIZE = 32
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.99
EPSILON_END = 0.01
DECAY = 300
TGT_UPD = 10

env = Game()

def get_epsilon(t):
    threshold = EPSILON_END + (EPSILON - EPSILON_END)*math.exp(-1. * t / DECAY)
    return threshold
    
def optimize(policy_net, target_net, memory, optimizer, criterion):
    
    policy_net.to(device)
    target_net.to(device)
    
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    
    state_batch = torch.cat(batch.state).to(device).to(torch.float)
    action_batch = torch.cat(batch.action).unsqueeze(1).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    done = torch.cat(batch.done).to(device)
    next_state = torch.cat(batch.next_state).to(device).to(torch.float).unsqueeze(1)
    
    policy_net.eval()
    with torch.no_grad():
        #actions_new = torch.argmax(policy_net(next_state.to(torch.float)).detach(), 1).unsqueeze(0)
        actions_new = torch.argmax(policy_net(next_state.to(torch.float)).detach(), 1).unsqueeze(0)
        next_state_values = target_net(next_state.to(torch.float)).squeeze().gather(1, actions_new).squeeze()
    
    policy_net.train()
    
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze()
    
    expected_state_action = reward_batch + (GAMMA * next_state_values*done)
    loss = criterion(state_action_values, expected_state_action)
    
    optimizer.zero_grad()
    loss.backward()
    
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    
    optimizer.step()
    
    return env
    
env.initialize_game()

blue_memory = []
red_memory = []

for i in range(len(env.blue_ships)):
    blue_memory.append(ReplayMemory(10000))

for i in range(len(env.red_ships)):
    red_memory.append(ReplayMemory(10000))
    
if not skip_training:
    """Training"""

    episodes = 200

    seed = 21 # random.randint(0,100)
    random.seed(seed)
    # avg_reward_50 = deque([], 50)
    
    all_blue_rewards = []
    all_red_rewards = []

    total_steps = 0
    

    for i in range(1, episodes):
        print('\n EPISODE', i)
        env.initialize_game()
        
        epochs, penalties, reward = 0, 0, 0

        episode_steps = 0
        
        for unit in env.blue_ships:
            unit.policy.zero_grad()
            unit.target.zero_grad()
            unit.policy.eval()
            #target_net.eval() # Set NN model to evaluation state

        for unit in env.red_ships:
            unit.policy.zero_grad()
            unit.target.zero_grad()
            unit.policy.eval()
            #target_net.eval() # Set NN model to evaluation state

        #criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()

        done = 1
        
        while done == 1:
            
            env.red_ew.clear()
            env.blue_ew.clear()

            blue_memory_inserts = []
            red_memory_inserts = []

            
            if episode_steps % 100 == 0:
                print('\n')
                print('Episode steps: {}'.format(episode_steps))

            for m, unit in enumerate(env.blue_ships):
                
                unit.policy.to(device)

                state = unit.get_obs()
                blue_state = np.expand_dims(state, 0)
                
                e = get_epsilon(unit.steps_done)

                if random.uniform(0, 1) < e:
                    blue_action = random.randint(0, unit.n_actions-1) # Pick random action
                else:
                    
                    with torch.no_grad():
                        a = torch.tensor(blue_state, dtype=torch.float32).unsqueeze(0).to(device)
                        a = unit.policy(a)
                        a = a.detach()
                        blue_action = torch.argmax(a).item()
                
                oldpos = unit.position

                blue_new_state, blue_reward, blue_done, info = env.step(unit, blue_action) # check next state w.r.t. the action
                
                blue_reward -= 0.1*unit.steps_done

                all_blue_rewards.append(blue_reward)

                memory = blue_memory[m]

                blue_memory_inserts.append((blue_state, blue_action, blue_new_state, blue_reward, blue_done))
                
                total_steps += 1
                unit.steps_done += 1
            
                if unit.steps_done % BATCH_SIZE == 0 and memory.__len__() > BATCH_SIZE:
                
                    optimize(unit.policy, unit.target, memory, unit.optimizer, criterion)
                
                if unit.steps_done % TGT_UPD == 0:
                    unit.target.load_state_dict(unit.policy.state_dict())

                if unit.environment.num_red != len(unit.environment.red_ships):
                    print('Red ship destroyed.')
                
                unit.environment.num_red = len(unit.environment.red_ships)

            for m, unit in enumerate(env.red_ships):

                unit.policy.to(device)
                
                state = unit.get_obs()
                state = np.expand_dims(state, 0)

                e = get_epsilon(unit.steps_done)
                    
                if random.uniform(0, 1) < e:
                    action = random.randint(0, unit.n_actions-1) # Pick random action
                else:
                    with torch.no_grad():
                        a = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        #print(a.shape)
                        a = unit.policy(a)
                        a = a.detach()
                        action = torch.argmax(a).item()

                oldpos = unit.position
                new_state, red_reward, done, info = env.step(unit, action) # check next state w.r.t. the action
                
                red_reward -= 0.1*unit.steps_done

                all_red_rewards.append(red_reward)

                red_memory_inserts.append((state, action, new_state, red_reward, done))

                memory = red_memory[m]

                total_steps += 1

                unit.steps_done += 1
            
                if unit.steps_done % BATCH_SIZE == 0 and memory.__len__() > BATCH_SIZE:
                
                    optimize(unit.policy, unit.target, memory, unit.optimizer, criterion)

                if unit.steps_done % TGT_UPD == 0:
                    unit.target.load_state_dict(unit.policy.state_dict())

                if unit.environment.num_blue != len(unit.environment.blue_ships):
                    print('Blue ship destroyed.')

                unit.environment.num_blue = len(unit.environment.blue_ships)


            blue_total_reward  = (env.num_red-len(env.red_ships))*10 - (env.num_blue-len(env.blue_ships))*5

            if env.num_red == 0:
                blue_total_reward += 100

            red_total_reward = (env.num_blue-len(env.blue_ships))*10 - (env.num_red-len(env.red_ships))*5

            if env.num_blue == 0:
                red_total_reward += 100

            for m, n in enumerate(blue_memory_inserts):
                state, action, new_state, breward, done = n
                blue_memory[m].push(
                    torch.tensor(state).unsqueeze(0),
                    torch.tensor([action]),
                    torch.tensor(new_state).unsqueeze(0),
                    torch.tensor([breward + blue_total_reward]),
                    torch.tensor([done])
                    )
                
            for m, n in enumerate(red_memory_inserts):
                state, action, new_state, rreward, done = n
                red_memory[m].push(
                    torch.tensor(state).unsqueeze(0),
                    torch.tensor([action]),
                    torch.tensor(new_state).unsqueeze(0),
                    torch.tensor([rreward + red_total_reward]),
                    torch.tensor([done])
                    )
                
            episode_steps += 1

            if episode_steps > 400:
                done = 0
        
            if i % 50 == 0:
                env.visualize_grid()

            
            """
            if i % 100 == 0:
                clear_output(wait=True)
                print(f"Model saved, Episode: {i}", PATH)
                torch.save(unit.target_net.state_dict, PATH)
            """
    print("Training finished.\n")

    plt.plot(all_blue_rewards)
    plt.plot(all_red_rewards)
    plt.show()
        
    for num, unit in enumerate(env.blue_ships):
        torch.save(unit.target.state_dict(), PATH + num + 'blue')

    for num, unit in enumerate(env.red_ships):
        torch.save(unit.target.state_dict(), PATH + num + 'red')
    
    print("Training finished.\n")
    
"""
else:
    target_net = torch.load(PATH)
    target_net.eval()
    
    print("Loaded existing q-table", PATH)
"""