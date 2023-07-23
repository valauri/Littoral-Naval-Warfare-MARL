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

load_models = False

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
DECAY = 500
TGT_UPD = 10

env = Game()

def average_weights(models):
    averaged_model = models[0].__class__(28)

    averaged_params = averaged_model.parameters()

    for model in models:
        model_params = model.parameters()
        for avg_param, model_param in zip(averaged_params, model_params):
            avg_param.data.add_(model_param.data.to('cpu'))

    for avg_param in averaged_params:
        avg_param.data.div_(len(models))

    return averaged_model

def get_epsilon(t):
    threshold = EPSILON_END + (EPSILON - EPSILON_END)*math.exp(-1. * t / DECAY)
    return threshold
    
def optimize(policy_net, target_net, memory, optimizer, criterion):
    
    policy_net.to(device)
    target_net.to(device)
    
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    
    state_batch = torch.cat(batch.state).to(device).to(torch.float)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    done = torch.cat(batch.done).to(device)
    next_state = torch.cat(batch.next_state).to(device).to(torch.float)
    
    policy_net.eval()
    with torch.no_grad():
        #actions_new = torch.argmax(policy_net(next_state.to(torch.float)).detach(), 1).unsqueeze(0)
        actions_new = torch.argmax(policy_net(next_state.to(torch.float)).detach(), 1).unsqueeze(0)
        next_state_values = target_net(next_state.to(torch.float)).squeeze().gather(1, actions_new).squeeze()
    
    policy_net.train()

    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
    
    expected_state_action = reward_batch + (GAMMA * next_state_values*done)
    loss = criterion(state_action_values, expected_state_action)
    
    optimizer.zero_grad()
    loss.backward()
    
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    
    optimizer.step()
    
    return env
    
env.create_combatants()
env.initialize_game()

blue_memory = []
red_memory = []

for i in range(len(env.blue_ships)):
    blue_memory.append(ReplayMemory(10000))

for i in range(len(env.red_ships)):
    red_memory.append(ReplayMemory(10000))
    
if not skip_training:
    """Training"""

    episodes = 250

    seed = 21 # random.randint(0,100)
    random.seed(seed)
    # avg_reward_50 = deque([], 50)
    
    all_blue_rewards = []
    all_red_rewards = []

    total_steps = 0
    
    if load_models:
        for num, unit in enumerate(env.blue_ships):
            file = os.path.join(PATH, 'blue' + str(num))
            unit.policy.load_state_dict(torch.load(file))
            print(f'Loaded model: {file}')

        for num, unit in enumerate(env.red_ships):
            file = os.path.join(PATH, 'red' + str(num))
            unit.policy.load_state_dict(torch.load(file))
            print(f'Loaded model: {file}')

    env.imagen = 0

    for i in range(1, episodes):
        print('\n EPISODE \n', i)
        env.initialize_game()

        """

        """

        # env.imagen = 0
        # env.visualize_grid()
        
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
            env.engagements.clear()

            blue_memory_inserts = []
            red_memory_inserts = []

            
            if episode_steps % 20 == 0:
                print('Episode steps: {}'.format(episode_steps))

            for m, unit in enumerate(env.blue_ships):
                
                unit.policy.to(device)

                blue_state = unit.get_obs()
                #blue_state = np.expand_dims(state, 0)

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
                
                blue_reward -= 0.01*unit.steps_done

                memory = blue_memory[m]

                blue_memory_inserts.append((blue_state, blue_action, blue_new_state, blue_reward, blue_done))
                
                total_steps += 1
                unit.steps_done += 1
            
                if unit.steps_done % BATCH_SIZE == 0 and memory.__len__() > BATCH_SIZE:
                
                    optimize(unit.policy, unit.target, memory, unit.optimizer, criterion)
                    #file = os.path.join(PATH, 'blue' + str(m))
                    #torch.save(unit.target.state_dict(), file)
                
                if unit.steps_done % TGT_UPD == 0:
                    unit.target.load_state_dict(unit.policy.state_dict())

                if unit.environment.num_red != len(unit.environment.red_ships):
                    print('Red ship destroyed.')
                
                unit.environment.num_red = len(unit.environment.red_ships)

            for m, unit in enumerate(env.red_ships):

                unit.policy.to(device)
                
                state = unit.get_obs()
                #state = np.expand_dims(state, 0)

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
                
                red_reward -= 0.01*unit.steps_done

                all_red_rewards.append(red_reward)

                red_memory_inserts.append((state, action, new_state, red_reward, done))

                memory = red_memory[m]

                total_steps += 1

                unit.steps_done += 1
            
                if unit.steps_done % BATCH_SIZE == 0 and memory.__len__() > BATCH_SIZE:
                
                    optimize(unit.policy, unit.target, memory, unit.optimizer, criterion)
                    #file = os.path.join(PATH, 'red' + str(m))
                    #torch.save(unit.target.state_dict(), file)

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

            min_dist = 200

            for unit in env.red_ships:
                if math.sqrt((unit.position[0]-50)**2 + (unit.position[1]-50)**2) < min_dist:
                    min_dist = math.sqrt((unit.position[0]-50)**2 + (unit.position[1]-50)**2)

            if min_dist != 0:
                red_total_reward += (1/min_dist)
            else:
                red_total_reward += 0

            b_reward = 0

            for m, n in enumerate(blue_memory_inserts):
                state, action, new_state, breward, done = n
                blue_memory[m].push(
                    torch.tensor(state).unsqueeze(0),
                    torch.tensor([action]),
                    torch.tensor(new_state).unsqueeze(0),
                    torch.tensor([breward + blue_total_reward]),
                    torch.tensor([done])
                    )
                b_reward += breward
                
            all_blue_rewards.append(b_reward)

            r_reward = 0
                
            for m, n in enumerate(red_memory_inserts):
                state, action, new_state, rreward, done = n
                red_memory[m].push(
                    torch.tensor(state).unsqueeze(0),
                    torch.tensor([action]),
                    torch.tensor(new_state).unsqueeze(0),
                    torch.tensor([rreward + red_total_reward]),
                    torch.tensor([done])
                    )
                r_reward += rreward

            all_red_rewards.append(r_reward)
            
            episode_steps += 1

            if episode_steps > 220:
                done = 0

            #if i % 50 == 0:
            #    env.visualize_grid()

            if env.engagements:
                env.visualize_grid()

        if i % 20 == 0:
            blue_models = []
            for unit in env.blue_ships:
                blue_models.append(unit.policy) 

            red_models = []
            for unit in env.red_ships:
                red_models.append(unit.policy)

            if len(blue_models) > 1:
                blue_averaged_model = average_weights(blue_models)
                for unit in env.blue_ships:
                    unit.policy.load_state_dict(blue_averaged_model.state_dict())
                print("Blue models aggregated.")

            if len(red_models) > 1:
                red_averaged_model = average_weights(red_models)       
                for unit in env.red_ships: 
                    unit.policy.load_state_dict(red_averaged_model.state_dict())
                print("Red models aggregated.")

            
            """
            if i % 100 == 0:
                clear_output(wait=True)
                print(f"Model saved, Episode: {i}", PATH)
                torch.save(unit.target_net.state_dict, PATH)
            """
    print("Training finished.\n")
    
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(list(range(0, len(all_blue_rewards))), all_blue_rewards, 'b', '-')
    fig.suptitle('Blue rewards')

    axs[1].plot(list(range(0, len(all_red_rewards))), all_red_rewards, 'r', '-')
    fig.suptitle('Red rewards')

    plt.show()
        
    for num, unit in enumerate(env.blue_ships):
        file = os.path.join(PATH, 'blue' + str(num))
        torch.save(unit.target.state_dict(), file)
        print(f'Saved model: {file}')

    for num, unit in enumerate(env.red_ships):
        file = os.path.join(PATH, 'red' + str(num))
        torch.save(unit.target.state_dict(), file)
        print(f'Saved model: {file}')
    
    print("Training finished.\n")
    
"""
else:
    target_net = torch.load(PATH)
    target_net.eval()
    
    print("Loaded existing q-table", PATH)
"""
