import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from game import Game
#import multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
import sys, os
import os.path
import wandb
import math
import random
from collections import deque
import json
from collections import namedtuple, deque

from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from network import MLP, Value, DMLP

PATH = os.path.join(os.getcwd(), 'discrete_models')

with open('config.json') as config_file:
    config = json.load(config_file)

overall = config.get("overall", {})
envirnoment_setup = config.get("environment_setup", {})
model_selection = config.get("model_selection", {})
hyperparameters = config.get("hyperparameters", {})

RANDOM_SEED = overall['seed']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = os.path.join(os.getcwd(), 'gif')
BATCH_SIZE = hyperparameters['batch_size']
STD_START = hyperparameters['std_start']
STD_END = hyperparameters['std_end']
LR = hyperparameters['learning_rate']
K_EPOCHS = hyperparameters['epochs']
EPS_CLIP = hyperparameters['eps_clip']
GAMMA = hyperparameters['gamma']
EPSILON_END = hyperparameters['epsilon_end']
EPSILON = hyperparameters['epsilon']
DECAY = hyperparameters['decay']

ROLLOUT_STEPS = hyperparameters['episode_steps']
SIDE = envirnoment_setup['side']
N_BLUE = envirnoment_setup['n_blue']
N_RED = envirnoment_setup['n_red']
TRAINED_RED = envirnoment_setup['trained_red']
TRAINED_BLUE = envirnoment_setup['trained_blue']
RED_AGGRESSION = envirnoment_setup['red_aggression']

WANDA = overall['wandb']

TGT_UPD = BATCH_SIZE*3

model_path = os.path.join(os.getcwd(), 'discrete_models')


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size, weights=None):
        if weights is None:
            return random.sample(self.memory, batch_size)
        else:
            indices = random.choices(range(len(self.memory)), weights=weights, k=batch_size)
            return [self.memory[idx] for idx in indices]

    def __len__(self):
        return len(self.memory)

    def empty(self):
        self.memory = deque([], maxlen=self.capacity)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class DDQN:
    def __init__(self, env, device, lr=LR, gamma=GAMMA):
        self.env = env
        self.device = device
        self.lr = lr
        self.gamma = gamma

        self.policy_net = DMLP(env.observation_space).to(device)
        self.target_net = DMLP(env.observation_space).to(device)

        self.red_policy_net = DMLP(env.red_observation_space).to(device)
        self.red_target_net = DMLP(env.red_observation_space).to(device)

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.red_optimizer = Adam(self.red_policy_net.parameters(), lr=lr)

        self.max_len = 10000

        self.memory = ReplayMemory(self.max_len)

        self.steps_done = 0

        self.blue_victory = 0
        self.red_victory = 0

    def transfer_weights(self, policy, target):

        self.policy_net.fc2.weight.data = policy.fc2.weight.data
        self.policy_net.conv1.weight.data = policy.conv1.weight.data
        self.policy_net.conv2.weight.data = policy.conv2.weight.data
        self.policy_net.bn1.weight.data = policy.bn1.weight.data
        self.policy_net.bn2.bias.data = policy.bn1.bias.data

        self.policy_net.movement.weight.data = policy.movement.weight.data
        self.policy_net.attack.weight.data = policy.attack.weight.data
        self.policy_net.radar.weight.data = policy.radar.weight.data

        self.target_net.fc2.weight.data = target.fc2.weight.data
        self.target_net.conv1.weight.data = target.conv1.weight.data
        self.target_net.conv2.weight.data = target.conv2.weight.data
        self.target_net.bn1.weight.data = target.bn1.weight.data
        self.target_net.bn2.bias.data = target.bn1.bias.data

        self.target_net.movement.weight.data = target.movement.weight.data
        self.target_net.attack.weight.data = target.attack.weight.data
        self.target_net.radar.weight.data = target.radar.weight.data

    def get_epsilon(self, t):
        threshold = EPSILON_END + (EPSILON - EPSILON_END)*math.exp(-1. * t / DECAY)
        return threshold
    
    def optimize(self, weights=None):
        #print("Optimizer update")
        
        if weights is not None:
            weights = weights + 1e-5 / sum(weights)

        transitions = self.memory.sample(BATCH_SIZE, weights)

        criterion = nn.MSELoss()

        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state).to(device).to(torch.float)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done = torch.cat(batch.done).to(device)
        next_state = torch.cat(batch.next_state).to(device)
        
        if SIDE == "blue":

            self.policy_net.eval()
            with torch.no_grad():
                rad, msl, mov = self.target_net(next_state.to(torch.float))

                rad_target = torch.max(rad, dim=1).values.unsqueeze(1).detach()
                msl_target = torch.max(msl, dim=1).values.unsqueeze(1).detach()
                mov_target = torch.max(mov, dim=1).values.unsqueeze(1).detach()

                next_state_values = torch.cat((rad_target, msl_target, mov_target), dim=1)

            self.policy_net.train()

            rad_p, msl_p, mov_p = self.policy_net(state_batch.to(torch.float))

            rad_actions = rad_p.gather(1, action_batch[:, 0].unsqueeze(1))
            msl_actions = msl_p.gather(1, action_batch[:, 1].unsqueeze(1))
            mov_actions = mov_p.gather(1, action_batch[:, 2].unsqueeze(1))

            state_action_values = torch.cat((rad_actions, msl_actions, mov_actions), dim=1)
        
            expected_state_action = GAMMA * next_state_values*done.unsqueeze(1) + reward_batch.unsqueeze(1)
            expected_state_action = expected_state_action.type(torch.float32)
            state_action_values = state_action_values.type(torch.float32)

            loss = criterion(state_action_values, expected_state_action)
            
            self.optimizer.zero_grad()
            loss.backward()

            if WANDA:
                wandb.log({"Blue loss": loss.item()})
            
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
        
            self.optimizer.step()

        if SIDE == "red":

            self.red_policy_net.eval()
            with torch.no_grad():
                rad, msl, mov = self.red_target_net(next_state.to(torch.float))

                rad_target = torch.max(rad, dim=1).values.unsqueeze(1).detach()
                msl_target = torch.max(msl, dim=1).values.unsqueeze(1).detach()
                mov_target = torch.max(mov, dim=1).values.unsqueeze(1).detach()

                next_state_values = torch.cat((rad_target, msl_target, mov_target), dim=1)

            self.policy_net.train()

            rad_p, msl_p, mov_p = self.red_policy_net(state_batch.to(torch.float))

            rad_actions = rad_p.gather(1, action_batch[:, 0].unsqueeze(1))
            msl_actions = msl_p.gather(1, action_batch[:, 1].unsqueeze(1))
            mov_actions = mov_p.gather(1, action_batch[:, 2].unsqueeze(1))

            state_action_values = torch.cat((rad_actions, msl_actions, mov_actions), dim=1)
        
            expected_state_action = GAMMA * next_state_values*done.unsqueeze(1) + reward_batch.unsqueeze(1)
            expected_state_action = expected_state_action.type(torch.float32)
            state_action_values = state_action_values.type(torch.float32)

            loss = criterion(state_action_values, expected_state_action)

            self.red_optimizer.zero_grad()
            loss.backward()

            if WANDA:
                wandb.log({"Red loss": loss.item()})

            for param in self.red_policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            
            self.red_optimizer.step()


    def learn(self):
        """Training"""

        episodes = 2000
        total_steps = 1

        random.seed(RANDOM_SEED)

        avg_reward_50 = deque([], 50)
        
        all_rewards = []

        reward_weights = deque([], 10000)

        if WANDA:
            wandb.init(project="DDQN")
        
        for i in range(1, episodes):
            print('episode', i)
            self.env.reset(N_BLUE, N_RED)
            steps_done = 0
            
            epochs, penalties, reward = 0, 0, 0
            done = 1
            
            self.policy_net.zero_grad()
            self.target_net.zero_grad()
            self.red_policy_net.zero_grad()
            self.red_target_net.zero_grad()

            actions_for_execution = []
            
            while done == 1:
                #target_net.eval() # Set NN model to evaluation state
                self.policy_net.eval()
                
                e = self.get_epsilon(steps_done)

                if SIDE == "blue":
                    current_states = np.zeros((1, len(self.env.blue_ships), self.env.observation_space))
                else:
                    current_states = np.zeros((1, len(self.env.red_ships), self.env.red_observation_space))
                
                if SIDE == "blue":
                    for idx, ship in enumerate(self.env.blue_ships):
                        if ship is not None:
                            state = ship.get_obs()
                            current_states[0, idx, :] = state

                            if random.uniform(0, 1) < e:
                                action = [random.randint(0,1), random.randint(0,4), random.randint(0,49)] # Pick random action
                            else:
                                with torch.no_grad():
                                    a = self.target_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                                    rad, msl, mov = a
                                    rad_action = torch.argmax(rad).item()
                                    msl_action = torch.argmax(msl).item()
                                    mov_action = torch.argmax(mov).item()
                                    action = [rad_action, msl_action, mov_action]

                            actions_for_execution.append(action)
                        else:
                            actions_for_execution.append([0, 0, 0])

                    for idx, ship in enumerate(self.env.red_ships):
                        if ship is not None:
                            if not TRAINED_RED:
                                state = ship.get_obs()
                                current_states[0, idx, :] = state
                                if steps_done < 20:
                                    action = [np.round(random.randint(0,1)), np.round(random.randint(0,1)), np.round(random.randint(2,4))]
                                    actions_for_execution.append(action)
                                else:
                                    if random.random() < RED_AGGRESSION and ship.target_list:
                                        msl_action = random.randint(1,4)
                                    else:
                                        msl_action = 0
                                    action = [random.randint(0,1), msl_action, random.randint(0,49)]
                                    actions_for_execution.append(action)
                            else:
                                with torch.no_grad():
                                    a = self.red_target_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                                    rad, msl, mov = a
                                    rad_action = torch.argmax(rad).item()
                                    msl_action = torch.argmax(msl).item()
                                    mov_action = torch.argmax(mov).item()
                                    action = [rad_action, msl_action, mov_action]

                                actions_for_execution.append([rad_action, msl_action, mov_action])
                        else:
                            actions_for_execution.append([0, 0, 0])

                else:
                    for idx, ship in enumerate(self.env.blue_ships):
                        if ship is not None:
                            state = ship.get_obs()
                            current_states[0, idx, :] = state

                            if random.uniform(0,1) < e:
                                action = [random.randint(0,1), random.randint(0,4), random.randint(0,49)]
                            else:
                                with torch.no_grad():
                                    a = self.target_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                                    rad, msl, mov = a
                                    rad_action = torch.argmax(rad).item()
                                    msl_action = torch.argmax(msl).item()
                                    mov_action = torch.argmax(mov).item()
                                    action = [rad_action, msl_action, mov_action]

                            actions_for_execution.append(action)
                        else:
                            actions_for_execution.append([0, 0, 0])

                    for idx, ship in enumerate(self.env.red_ships):
                        if ship is not None:
                            state = ship.get_obs()
                            current_states[0, idx, :] = state

                            if random.uniform(0, 1) < e:
                                action = [random.randint(0,1), random.randint(0,4), random.randint(0,49)]
                            else:
                                with torch.no_grad():
                                    a = self.red_target_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                                    rad, msl, mov = a
                                    rad_action = torch.argmax(rad).item()
                                    msl_action = torch.argmax(msl).item()
                                    mov_action = torch.argmax(mov).item()
                                    if steps_done < 20:
                                        mov_action = np.random.randint(2,4)

                                    action = [rad_action, msl_action, mov_action]

                            actions_for_execution.append(action)
                        else:
                            actions_for_execution.append([0, 0, 0])

                
                new_states, reward, done, info = self.env.step(actions_for_execution) # check next state w.r.t. the action

                if WANDA:
                    if SIDE == "blue":
                        wandb.log({"Blue 1 reward": reward[0], "Blue 2 reward": reward[1]})
                    else:
                        wandb.log({"Red 1 reward": reward[0], "Red 2 reward": reward[1]})

                self.blue_victory += self.env.blue_victory
                self.red_victory += self.env.red_victory

                if WANDA:
                    wandb.log({"Red victory": self.red_victory, "Blue victory": self.blue_victory})
                    
                for r in reward:
                    reward_weights.append(r)
                
                if SIDE == "blue":
                    for k in range(len(self.env.blue_ships)):
                        state = current_states[0, k, :]
                        action = actions_for_execution[k]
                        new_state = new_states[0, k, :]
                        r = np.asarray(reward[k])
                        

                        self.memory.push(
                            torch.tensor(state).unsqueeze(0),
                            torch.tensor(action).unsqueeze(0),
                            torch.tensor(new_state).unsqueeze(0),
                            torch.tensor(r).unsqueeze(0),
                            torch.tensor(np.asarray(done)).unsqueeze(0)
                            )
                else:
                    for k in range(len(self.env.red_ships)):
                        state = current_states[0, k, :]
                        action = actions_for_execution[len(self.env.blue_ships)+k]
                        new_state = new_states[0, k, :]
                        r = np.asarray(reward[k])
                        
                        self.memory.push(
                            torch.tensor(state).unsqueeze(0),
                            torch.tensor(action).unsqueeze(0),
                            torch.tensor(new_state).unsqueeze(0),
                            torch.tensor(r).unsqueeze(0),
                            torch.tensor(np.asarray(done)).unsqueeze(0)
                            )

                total_steps += 1
                steps_done += 1
                
                if total_steps % BATCH_SIZE == 0:
                    self.optimize()
                    
                if steps_done % TGT_UPD == 0:
                    if SIDE == "blue":
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    else:
                        self.red_target_net.load_state_dict(self.red_policy_net.state_dict())
                
                state = new_state

                if steps_done > 50:
                    done = 0
                
            all_rewards.append(reward)
                
            avg_reward_50.append(reward)
            
        if SIDE == "blue":
            self.target_net.load_state_dict(self.policy_net.state_dict())
            torch.save(self.target_net.state_dict(), os.path.join(model_path, 'target.pth'))
            torch.save(self.policy_net.state_dict(), os.path.join(model_path, 'blue_policy.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'blue_optimizer.pth'))
        else:
            self.red_target_net.load_state_dict(self.red_policy_net.state_dict())
            torch.save(self.red_target_net.state_dict(), os.path.join(model_path, 'red_target.pth'))
            torch.save(self.red_policy_net.state_dict(), os.path.join(model_path, 'red_policy.pth'))
            torch.save(self.red_optimizer.state_dict(), os.path.join(model_path, 'red_optimizer.pth'))
        
        #with open('reward_log.json', 'w') as f:
        #    f.write(json.dumps(all_rewards))
        
        if WANDA:
            wandb.finish()

        print("Training finished.\n")
    
    
