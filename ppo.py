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

from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from network import MLP, Value

with open('config.json') as config_file:
    config = json.load(config_file)

PATH = os.path.join(os.getcwd(), 'gif')

overall = config.get("overall", {})
environment_setup = config.get("environment_setup", {})
model_selection = config.get("model_selection", {})
hyperparameters = config.get("hyperparameters", {})

random_seed = overall['seed']
WANDA = overall['wandb']
COA_PATH = overall['coa_path']
LANDING_OPS = overall['landing_ops']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINED_RED = environment_setup['trained_red']
SIDE = environment_setup['side']

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
NETWORK_NOISE = hyperparameters['network_noise']
NETWORK_NOISE_CLIP = hyperparameters['network_noise_clip']



model_path = os.path.join(os.getcwd(), 'models')

random.seed(random_seed)

class PrioritizedDataset(Dataset):
    def __init__(self, data, priorities):
        self.data = data
        self.priorities = priorities

    def __len__(self):
        return len(self.data)  # Assuming all components have the same length

    def __getitem__(self, index):
        return tuple(component[index] for component in self.data), self.priorities[index]

class PPO:

    def __init__(self, env, device, lr=LR, gamma=GAMMA, K_epochs=K_EPOCHS, eps_clip=EPS_CLIP, training=SIDE):

        self.actor = MLP(env.observation_space-7*7+12, env.action_space).to(device)
        self.critic = Value(env.observation_space*len(env.blue_ships)).to(device)
        self.red_actor = MLP(env.red_observation_space-7*7+12, env.action_space).to(device)
        self.red_critic = Value(env.red_observation_space*len(env.red_ships)).to(device)

        self.red_landing_ops = MLP(env.red_observation_space-7*7+12, env.action_space).to(device)
        self.red_landing_critic = Value(env.red_observation_space*len(env.red_ships)).to(device)

        self.env = env
        self.device = device
        self.gamma = gamma
        self.learning_rate = lr
        self.eps_clip = torch.tensor(eps_clip).to(device)
        self.K_epochs = K_epochs
        self.n_rollouts = 10
        self.n_rollout_steps = ROLLOUT_STEPS

        self.epsilon = 0.2

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.red_actor_optimizer = Adam(self.red_actor.parameters(), lr=lr)
        self.red_critic_optimizer = Adam(self.red_critic.parameters(), lr=lr)

        self.red_landing_ops_optimizer = Adam(self.red_landing_ops.parameters(), lr=lr)
        self.red_landing_critic_optimizer = Adam(self.red_landing_critic.parameters(), lr=lr)

        self.blue_networks = []
        self.blue_optimizers = []

        self.red_networks = []
        self.red_optimizers = []

        self.actor_loss_log = []

        self.reward_log = []

        self.blue_victory = 0
        self.red_victory = 0

        self.param_noise_clip_factor = NETWORK_NOISE_CLIP

        self.blue_reward_stack = deque(maxlen=20)
        self.red_reward_stack = deque(maxlen=20)

        self.noise_ratio = torch.tensor(STD_START).to(device)
        self.add_param_noise = True

        self.blue_rollout_rewards = []
        self.red_rollout_rewards = []

        self.blue_rollout_states = {}
        self.red_rollout_states = {}

    def init_networks(self, n_blue, n_red):
        for n in range(n_blue):
            self.blue_networks.append(MLP(self.env.observation_space, self.env.action_space).to(device))
            self.blue_optimizers.append(Adam(self.blue_networks[n].parameters(), lr=self.learning_rate))

        for n in range(n_red):
            self.red_networks.append(MLP(self.env.observation_space, self.env.action_space).to(device))
            self.red_optimizers.append(Adam(self.red_networks[n].parameters(), lr=self.learning_rate))
    
    def transfer_weights(self, actor, critic, side):
        if side == "blue":
            self.actor.conv1 = actor.conv1
            self.actor.norm1 = actor.norm1
            self.actor.conv2 = actor.conv2
            self.actor.norm2 = actor.norm2
            self.convhead = actor.convhead.weight.data

            self.actor.fc2.weight.data = actor.fc2.weight.data
            self.actor.fc3.weight.data = actor.fc3.weight.data
            self.actor.normal_head.weight.data = actor.normal_head.weight.data
            self.actor.log_std_head.weight.data = actor.log_std_head.weight.data

            self.critic.fc2.weight.data = critic.fc2.weight.data
            self.critic.fc3.weight.data = critic.fc3.weight.data
            self.critic.fc4.weight.data = critic.fc4.weight.data
        else:
            self.actor.conv1 = actor.conv1
            self.actor.norm1 = actor.norm1
            self.actor.conv2 = actor.conv2
            self.actor.norm2 = actor.norm2
            self.convhead = actor.convhead.weight.data
            self.red_actor.fc2.weight.data = actor.fc2.weight.data
            self.red_actor.fc3.weight.data = actor.fc3.weight.data
            self.red_actor.normal_head.weight.data = actor.normal_head.weight.data
            self.red_actor.log_std_head.weight.data = actor.log_std_head.weight.data

            self.red_critic.fc2.weight.data = critic.fc2.weight.data
            self.red_critic.fc3.weight.data = critic.fc3.weight.data
            self.red_critic.fc4.weight.data = critic.fc4.weight.data

    def calculate_entropy(self, probs):
        entropy = -torch.sum(probs * torch.log(probs))

        return entropy
    
    def reduce_std(self, initial_std, final_std, timesteps, total_timesteps):
        ratio = min(timesteps / total_timesteps, 1.0)
        return initial_std - (initial_std - final_std) * ratio
    
    def trim_noise_clip(self, initial_clip, final_clip, timesteps, total_timesteps):
        ratio = min(timesteps / total_timesteps, 1.0)
        return initial_clip - (initial_clip - final_clip) * ratio

    def get_epsilon(self, t):
        threshold = EPSILON_END + (EPSILON - EPSILON_END)*math.exp(-1. * t / DECAY)
        return threshold

    def learn(self, total_timesteps, save_models=False):
        if WANDA:
            wandb.init(project="MAPPO")

        print("Learning")

        if SIDE == "blue":
            self.actor.train()
            self.critic.train()
        else:
            self.red_actor.train()
            self.red_critic.train()

        initial_std = STD_START # For Gaussian noise to enhance exploration
        final_std = STD_END # For Gaussian noise to enhance exploration

        t = 0
        i = 0

        blue_victory = 0
        red_victory = 0

        while t < total_timesteps:
            
            print("Timesteps: ", t)

            if NETWORK_NOISE:
                if SIDE == "blue":
                    delta_victory = abs(blue_victory-self.blue_victory)
                    blue_victory = self.blue_victory
                else:
                    delta_victory = abs(red_victory-self.red_victory)
                    red_victory = self.red_victory

                #self.param_noise_clip_factor = self.trim_noise_clip(NETWORK_NOISE_CLIP, NETWORK_NOISE_CLIP/100, t, total_timesteps)
            
                noise_std = self.noise_ratio

                if delta_victory > 0:
                    if delta_victory == 1:
                        self.noise_ratio = self.noise_ratio/1.5
                        noise_std = self.noise_ratio
                        self.param_noise_clip_factor = self.param_noise_clip_factor/1.5
                    if delta_victory == 2:
                        t = t + 2500
                        self.param_noise_clip_factor = self.param_noise_clip_factor/2
                        if SIDE == "blue":
                            for g in self.actor_optimizer.param_groups:
                                g['lr'] = g['lr']/2
                            for g in self.critic_optimizer.param_groups:
                                g['lr'] = g['lr']/2
                        else:
                            for g in self.red_actor_optimizer.param_groups:
                                g['lr'] = g['lr']/2
                            for g in self.red_critic_optimizer.param_groups:
                                g['lr'] = g['lr']/2
                    if delta_victory >= 3:
                        t = t + 1000 *delta_victory
                        self.param_noise_clip_factor = self.param_noise_clip_factor/3
                        if SIDE == "blue":
                            for g in self.actor_optimizer.param_groups:
                                g['lr'] = g['lr']/delta_victory
                            for g in self.critic_optimizer.param_groups:
                                g['lr'] = g['lr']/delta_victory
                        else:
                            for g in self.red_actor_optimizer.param_groups:
                                g['lr'] = g['lr']/delta_victory
                            for g in self.red_critic_optimizer.param_groups:
                                g['lr'] = g['lr']/delta_victory
                else:
                    noise_std = self.noise_ratio 
                    if SIDE == "blue":
                        for g in self.actor_optimizer.param_groups:
                            if g['lr'] < 0.0001:
                                g['lr'] = g['lr']*2
                        for g in self.critic_optimizer.param_groups:
                            if g['lr'] < 0.0001:
                                g['lr'] = g['lr']*2
                    else:
                        for g in self.red_actor_optimizer.param_groups:
                            if g['lr'] < 0.0001:
                                g['lr'] = g['lr']*2
                        for g in self.red_critic_optimizer.param_groups:
                            if g['lr'] < 0.0001:
                                g['lr'] = g['lr']*2

                    if self.param_noise_clip_factor < NETWORK_NOISE_CLIP:
                        self.param_noise_clip_factor = self.param_noise_clip_factor*1.1
                    if self.noise_ratio < STD_START:
                        self.noise_ratio = self.noise_ratio*1.1
                        noise_std = self.noise_ratio
                        
            else:
                noise_std = self.reduce_std(initial_std, final_std, t, total_timesteps)

            #if t >= total_timesteps:
            #    break

            if NETWORK_NOISE:
                noise_std = self.noise_ratio

            obs, acts, old_log_probs, rtgs, lens, global_states, old_values = self.rollout(self.n_rollouts, self.n_rollout_steps, noise_std)

            t += lens
            i += 1

            if SIDE == "blue":
                states = obs.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.blue_ships), self.env.observation_space)
                actions = acts.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.blue_ships), self.env.action_space)
                old_log_probs = old_log_probs.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.blue_ships), self.env.action_space)
                rewards_to_go = rtgs.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.blue_ships), 1)
                global_states = global_states.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.blue_ships), self.env.observation_space*len(self.env.blue_ships))
                old_values = old_values.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.blue_ships), 1)
            else:
                states = obs.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.red_ships), self.env.observation_space)
                actions = acts.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.red_ships), self.env.action_space)
                old_log_probs = old_log_probs.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.red_ships), self.env.action_space)
                rewards_to_go = rtgs.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.red_ships), 1)
                global_states = global_states.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.red_ships), self.env.observation_space*len(self.env.blue_ships))
                old_values = old_values.reshape(self.n_rollouts*self.n_rollout_steps*len(self.env.red_ships), 1)

            priorities = torch.abs(torch.flatten(rewards_to_go)) + 1e-5
            priorities = priorities/torch.sum(priorities)

            data_list = [states, actions, old_log_probs, rewards_to_go, global_states, old_values]

            prioritized_dataset = PrioritizedDataset(data_list, priorities)

            weighted_sampler = WeightedRandomSampler(priorities, num_samples=BATCH_SIZE, replacement=False)  # Initially drawn BATCH_SIZE samples from the dataset WITH replacement to highlight the importance of the samples with higher priority
            dataloader = DataLoader(prioritized_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)

            for epoch in range(self.K_epochs+delta_victory*2):
                for mini_batch, _ in dataloader:
                    (states, actions, old_log_probs, rewards, global_states, old_values) = mini_batch

                    states = states.to(device)
                    old_log_probs = old_log_probs.to(device)
                    actions = actions.to(device)
                    global_states = global_states.to(device)
                    rtgs = rewards.to(device)
                    old_values = old_values.to(device)

                    # Calculate the ratio (pi_theta / pi_theta__old):

                    values, new_log_probs, entropies = self.evaluate(states, global_states, actions)

                    advantage = self.gae(rtgs, values.detach()).to(device)

                    advantage = self.popart_normalize(advantage, rtgs)

                    advantage = torch.reshape(advantage, (-1,1))

                    ratio = torch.exp(new_log_probs.to(device) - old_log_probs.detach()).to(device)

                    #ratio = torch.reshape(ratio, (self.n_rollouts*self.n_rollout_steps*len(self.env.blue_ships), self.env.action_space))

                    surr1 =  advantage * ratio

                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                    #entropies = torch.reshape(entropies, (self.n_rollouts*self.n_rollout_steps*len(self.env.blue_ships), self.env.action_space))

                    #actor_loss = (-torch.min(surr1, surr2) + (-entropies.to(device) * self.epsilon)).mean()

                    actor_loss = -((torch.min(surr1, surr2)).mean() + self.epsilon*(entropies.to(device)).mean())

                    if math.isinf(actor_loss):
                        print("The loss is infinite.")

                    val = torch.flatten(values)
                    rew = torch.flatten(rtgs)

                    critic_loss = torch.sqrt(torch.max((val-rew)**2, ((torch.clamp(val, torch.flatten(old_values.detach()).to(device)-self.eps_clip, torch.flatten(old_values.detach()).to(device)+self.eps_clip))-rew)**2).mean())

                    #critic_loss = criterion(torch.flatten(V), torch.flatten(rtgs))
                    
                    if WANDA:
                        wandb.log({"Actor Loss": -actor_loss.item(), "Critic Loss:": critic_loss.item()})
                    
                    self.actor_loss_log.append(actor_loss.item())

                    if SIDE == "blue":
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward() # retain graph=True
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                        self.actor_optimizer.step()

                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                        self.critic_optimizer.step()
                    else:
                        self.red_actor_optimizer.zero_grad()
                        actor_loss.backward() # retain graph=True
                        torch.nn.utils.clip_grad_norm_(self.red_actor.parameters(), 1)
                        self.red_actor_optimizer.step()

                        self.red_critic_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.red_critic.parameters(), 1)
                        self.red_critic_optimizer.step()
                
                if sum(self.blue_reward_stack)/20 > len(self.env.blue_ships)*self.n_rollout_steps*2:
                    t = total_timesteps

                if t > 200000:
                    with open('config.json', 'r') as file:
                        config = json.load(file)

                        config['overall']['tactics'] = 'defensive'

                        # Write the modified content back to the file
                        with open('config.json', 'w') as file:
                            json.dump(config, file, indent=4)  # indent for pretty formatting (optional)

        if WANDA:
            wandb.finish()

        if save_models:
            if SIDE == "blue":
                torch.save(self.actor.state_dict(), os.path.join(model_path, 'blue_actor.pth'))
                torch.save(self.critic.state_dict(), os.path.join(model_path, 'blue_critic.pth'))
                torch.save(self.actor_optimizer.state_dict(), os.path.join(model_path, 'blue_actor_optimizer.pth'))
                torch.save(self.critic_optimizer.state_dict(), os.path.join(model_path, 'blue_critic_optimizer.pth'))
            else:
                torch.save(self.red_actor.state_dict(), os.path.join(model_path, 'red_actor.pth'))
                torch.save(self.red_critic.state_dict(), os.path.join(model_path, 'red_critic.pth'))
                torch.save(self.red_actor_optimizer.state_dict(), os.path.join(model_path, 'red_actor_optimizer.pth'))
                torch.save(self.red_critic_optimizer.state_dict(), os.path.join(model_path, 'red_critic_optimizer.pth'))


    def rollout(self, n_rollouts, n_rollout_steps, noise_std):

        rollout_env = Game()
        rollout_env.reset(self.env.num_blue, self.env.num_red)

        if torch.is_tensor(noise_std):
            self.noise_ratio = noise_std.to(device)
        else:
            self.noise_ratio = torch.tensor(noise_std).to(device) #torch.tensor(noise_std).to(device)

        if SIDE == "blue":
            batch_states = np.zeros((n_rollouts, n_rollout_steps, len(self.env.blue_ships), self.env.observation_space))
            batch_actions = np.zeros((n_rollouts, n_rollout_steps, len(self.env.blue_ships), self.env.action_space))
            batch_log_probs = torch.zeros((n_rollouts, n_rollout_steps, len(self.env.blue_ships), self.env.action_space))
            batch_global_states = np.zeros((n_rollouts, n_rollout_steps, len(self.env.blue_ships), self.env.observation_space*len(self.env.blue_ships)))
            batch_rewards = np.zeros((n_rollouts, n_rollout_steps, len(self.env.blue_ships), 1))
            batch_rewards_to_go = np.zeros((n_rollouts, n_rollout_steps, len(self.env.blue_ships), 1))
            batch_values = torch.zeros((n_rollouts, n_rollout_steps, len(self.env.blue_ships), 1))
        else:
            batch_states = np.zeros((n_rollouts, n_rollout_steps, len(self.env.red_ships), self.env.observation_space))
            batch_actions = np.zeros((n_rollouts, n_rollout_steps, len(self.env.red_ships), self.env.action_space))
            batch_log_probs = torch.zeros((n_rollouts, n_rollout_steps, len(self.env.red_ships), self.env.action_space))
            batch_global_states = np.zeros((n_rollouts, n_rollout_steps, len(self.env.red_ships), self.env.observation_space*len(self.env.blue_ships)))
            batch_rewards = np.zeros((n_rollouts, n_rollout_steps, len(self.env.red_ships), 1))
            batch_rewards_to_go = np.zeros((n_rollouts, n_rollout_steps, len(self.env.red_ships), 1))
            batch_values = torch.zeros((n_rollouts, n_rollout_steps, len(self.env.red_ships), 1))

        lens = 0

        preordained_red_actions = [rollout_env.red1_actions, rollout_env.red2_actions, rollout_env.red3_actions]

        noiseless_actor = MLP(self.env.observation_space-7*7+12, self.env.action_space).to(device)
        noiseless_actor.load_state_dict(self.actor.state_dict())

        for batch in range(n_rollouts):
            rollout_env.reset(rollout_env.num_blue, rollout_env.num_red)#, orig_grid, blues, reds)

            ep_rewards = []

            done = 1

            actions_for_step = []

            #if len(ep_rewards) > 2 and ep_rewards[-1] < ep_rewards[-2]:

            self.actor.load_state_dict(noiseless_actor.state_dict())

            if self.add_param_noise and NETWORK_NOISE:
                with torch.no_grad():
                    if SIDE == "blue":
                        for name, param in self.actor.named_parameters():
                            if 'norm1' in name.lower() or 'norm2' in name.lower() or 'layernorm' in name.lower():
                                continue
                            noise = torch.clamp(torch.normal(torch.zeros(param.size()).to(device), self.noise_ratio.to(device)), -self.param_noise_clip_factor, self.param_noise_clip_factor) # Blue stats # -0.005, 0.005
                            param.to(device).add_(noise)
                    else:
                        for name, param in self.red_actor.named_parameters():
                            if 'norm1' in name.lower() or 'norm2' in name.lower() or 'layernorm' in name.lower():
                                continue
                            noise = torch.clamp(torch.normal(torch.zeros(param.size()).to(device), self.noise_ratio.to(device)), -self.param_noise_clip_factor, self.param_noise_clip_factor) # Blue stats -0.005, 0.005
                            param.to(device).add_(noise)

            for s in range(n_rollout_steps):
                actions_for_step.clear()

                rollout_env.engagements.clear()
                rollout_env.blue_ew.clear()
                rollout_env.red_ew.clear()

                if SIDE == "blue":
                    compiled_picture = np.zeros((len(self.env.blue_ships), self.env.observation_space))
                    combined_actions = torch.zeros((len(self.env.blue_ships), self.env.action_space))
                else:
                    compiled_picture = np.zeros((len(self.env.red_ships), self.env.observation_space))
                    combined_actions = torch.zeros((len(self.env.red_ships), self.env.action_space))
                
                for i, ship in enumerate(self.env.blue_ships):
                    if SIDE == "blue":
                        if ship is not None:
                            state = ship.get_obs()
                            compiled_picture[i, :] = state
                            batch_states[batch, s, i, :] = state
                            if NETWORK_NOISE:
                                action, log_prob = self.actor(torch.tensor(state, dtype=torch.float32, device = device))
                                if action == None and log_prob == None: # A solution to the nan problem, i.e. if the network outputs nan, load the noiseless network
                                    self.actor.load_state_dict(noiseless_actor.state_dict())
                                    action, log_prob = self.actor(torch.tensor(state, dtype=torch.float32, device = device))
                            else:
                                action, log_prob = self.actor(torch.tensor(state, dtype=torch.float32, device = device), noise_std)
                            combined_actions[i, :] = action
                            batch_actions[batch, s, i, :] = action.cpu().detach().numpy()
                            actions_for_step.append(action.cpu().detach().numpy())
                            batch_log_probs[batch, s, i, :] = log_prob.to(device) #.cpu().detach().numpy()
                        else:
                            actions_for_step.append(np.zeros((4)))
                            combined_actions[i, :] = action
                            batch_actions[batch, s, i, :] = action.cpu().detach().numpy()
                    else:
                        if ship is not None:
                            state = ship.get_obs()
                            with torch.no_grad():
                                self.actor.eval()
                                action, log_prob = self.actor(torch.tensor(state, dtype=torch.float32, device = device))
                                actions_for_step.append(action.cpu().detach().numpy())
                        else:
                            actions_for_step.append(np.zeros((4)))

                for i, ship in enumerate(self.env.red_ships):
                    if SIDE == "red":
                        if s > 14:
                            if ship is not None:
                                state = ship.get_obs()
                                compiled_picture[i, :] = state
                                batch_states[batch, s, i, :] = state
                                if NETWORK_NOISE:
                                    action, log_prob = self.red_actor(torch.tensor(state, dtype=torch.float32, device = device))
                                    if action == None and log_prob == None: # A solution to the nan problem, i.e. if the network outputs nan, load the noiseless network
                                        self.red_actor.load_state_dict(noiseless_actor.state_dict())
                                        action, log_prob = self.red_actor(torch.tensor(state, dtype=torch.float32, device = device))
                                else:
                                    action, log_prob = self.red_actor(torch.tensor(state, dtype=torch.float32, device = device), noise_std)
                                combined_actions[i, :] = action
                                batch_actions[batch, s, i, :] = action.cpu().detach().numpy()
                                actions_for_step.append(action.cpu().detach().numpy())
                                batch_log_probs[batch, s, i, :] = log_prob.to(device) #.cpu().detach().numpy()
                            else:
                                state = ship.get_obs()
                                actions_for_step.append(np.zeros((4)))
                                combined_actions[i, :] = action
                                batch_actions[batch, s, i, :] = action.cpu().detach().numpy()
                        else:
                            if ship is not None:
                                state = ship.get_obs()
                                preordained_actions = preordained_red_actions[i]
                                actions_for_step.append(preordained_actions[s])
                                log_prob, entropies = self.actor.get_dist(torch.tensor(state, dtype=torch.float32, device = device), torch.tensor(preordained_actions[s], dtype=torch.float32, device = device))
                                batch_log_probs[batch, s, i, :] = log_prob.to(device)
                            else:
                                actions_for_step.append(np.zeros((4)))

                    else:
                        if ship is not None:
                            if not TRAINED_RED:
                                state = ship.get_obs()
                                preordained_actions = preordained_red_actions[i]
                                actions_for_step.append(preordained_actions[s])
                            else:
                                state = ship.get_obs()
                                with torch.no_grad():
                                    self.red_actor.eval()
                                    action, log_prob = self.red_actor(torch.tensor(state, dtype=torch.float32, device = device))
                                    actions_for_step.append(action.cpu().detach().numpy())
                        else:
                            actions_for_step.append(np.zeros((4)))


                _, reward, done, cog_dist = rollout_env.step(np.asarray(actions_for_step))

                ep_rewards.append(sum(reward))

                if SIDE == "blue":
                    self.blue_reward_stack.append(sum(reward))
                else:
                    self.red_reward_stack.append(sum(reward))

                if NETWORK_NOISE:
                    if sum(reward) > n_rollout_steps*self.env.num_blue if SIDE == "blue" else n_rollout_steps*self.env.num_red:
                        self.noise_ratio = (self.noise_ratio*0.9).to(device)
                    elif sum(reward) < n_rollout_steps*self.env.num_blue if SIDE == "blue" else n_rollout_steps*self.env.num_red:
                        if self.noise_ratio < STD_START:
                            self.noise_ratio = (self.noise_ratio*1.1).to(device)
                    else:
                        if torch.is_tensor(noise_std):
                            self.noise_ratio = noise_std.to(device)
                        else:
                            self.noise_ratio = torch.tensor(noise_std).to(device)

                if SIDE == "blue":
                    cop = np.reshape(compiled_picture, (1, len(self.env.blue_ships)*self.env.observation_space))
                    value_estimate = self.critic(torch.tensor(cop, dtype=torch.float32, device=device).to(device))
                else:
                    cop = np.reshape(compiled_picture, (1, len(self.env.red_ships)*self.env.observation_space))
                    value_estimate = self.red_critic(torch.tensor(cop, dtype=torch.float32, device=device).to(device))

                batch_values[batch, s, :, :] = value_estimate
                
                if SIDE == "blue":
                    for i in range(len(self.env.blue_ships)):
                        batch_global_states[batch, s, i, :] = cop

                    for u in range(len(self.env.blue_ships)):
                        batch_rewards[batch, s, u, :] = reward[u]
                else:
                    for i in range(len(self.env.red_ships)):
                        batch_global_states[batch, s, i, :] = cop

                    for u in range(len(self.env.red_ships)):
                        batch_rewards[batch, s, u, :] = reward[u]

                lens += 1

                self.blue_victory += rollout_env.blue_victory
                self.red_victory += rollout_env.red_victory

                if WANDA:
                    if SIDE == "blue":
                        if len(rollout_env.blue_ships) > 2:
                            wandb.log({"Blue victory": self.blue_victory, "Red victory": self.red_victory, "Blue 1 reward": reward[0], "Blue 2 reward": reward[1], "Blue 3 reward": reward[2], \
                                       "Blue engagements": rollout_env.blue_engagements, "Red engagements": rollout_env.red_engagements, "Ducting factor": rollout_env.ducting_factor, "COG distance": cog_dist})
                        else:
                            wandb.log({"Blue victory": self.blue_victory, "Red victory": self.red_victory, "Blue 1 reward": reward[0], "Blue 2 reward": reward[1], "Blue engagements": rollout_env.blue_engagements, \
                                    "Red engagements": rollout_env.red_engagements, "Ducting factor": rollout_env.ducting_factor})
                    else:
                        if len(rollout_env.red_ships) > 2:
                            wandb.log({"Blue victory": self.blue_victory, "Red victory": self.red_victory, "Red 1 reward": reward[0], "Red 2 reward": reward[1], "Red 3 reward": reward[2], \
                                    "Blue engagements": rollout_env.blue_engagements, "Red engagements": rollout_env.red_engagements})
                        else:
                            wandb.log({"Blue victory": self.blue_victory, "Red victory": self.red_victory, "Red 1 reward": reward[0], "Red 2 reward": reward[1], "Blue engagements": rollout_env.blue_engagements, "Red engagements": rollout_env.red_engagements})

                if done == 0:
                    break
                    
            #calculate Reward-To-Go

            discounted_rewards = []

            for batch in range(n_rollouts):
                reward_batch = batch_rewards[batch, :, :, :]
                reversed_reward_batch = np.flip(reward_batch, axis=0)
                discounted_reward = 0
                discounted_rewards.clear()

                for n in range(reversed_reward_batch.shape[0]):
                    for s in range(len(rollout_env.blue_ships)):
                        discounted_reward += self.gamma * reversed_reward_batch[n, s]
                        discounted_rewards.append(discounted_reward)

                dic_r = np.reshape(np.asarray(discounted_rewards), (n_rollout_steps, len(rollout_env.blue_ships), 1))
                batch_rewards_to_go[batch, :, :, :] = dic_r


        batch_obs = torch.tensor(batch_states, dtype=torch.float32, device=self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float32, device=self.device)
        #batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=self.device)
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float32, device=self.device)
        batch_global_states = torch.tensor(batch_global_states, dtype=torch.float32, device=self.device)
        #batch_values = torch.tensor(batch_values, dtype=torch.float32, device=self.device)

        self.actor.load_state_dict(noiseless_actor.state_dict())

        return batch_obs, batch_actions, batch_log_probs, batch_rewards_to_go, lens, batch_global_states, batch_values
    
    def evaluate(self, obs, global_states, acts):
        
        log_probs = torch.zeros(BATCH_SIZE, self.env.action_space)
        entropies = torch.zeros(BATCH_SIZE, self.env.action_space)

        for i in range(BATCH_SIZE):
            if SIDE == "blue":
                log_prob, entropy = self.actor.get_dist(obs[i], acts[i])
                log_probs[i] = log_prob.to(device) 
                entropies[i] = entropy.to(device)
            else:
                log_prob, entropy = self.red_actor.get_dist(obs[i], acts[i])
                log_probs[i] = log_prob.to(device) 
                entropies[i] = entropy.to(device)

        if SIDE == "blue":
            V = self.critic(global_states)
        else:
            V = self.red_critic(global_states)

        return V.to(device), log_probs, entropies
    
    def gae(self, rewards, values, gamma=GAMMA, lambda_=0.95):

        gae = 0
        returns = torch.zeros_like(rewards)
        
        rewards = rewards
        values = values

        # Iterate in reverse to calculate GAE
        for i in reversed(range(len(rewards))):
            if i < len(rewards)-1:
                delta = rewards[i] + gamma * values[i + 1] - values[i]
                gae = delta + gamma * lambda_ * gae
                returns[i] = gae + values[i]
            else:
                delta = rewards[i] - values[i]
                gae = delta
                returns[i] = gae + values[i]
        
        return returns.to(device)
    
    def popart_normalize(self, values, targets):
        # Compute the mean and standard deviation of the values
        values_mean = values.mean()
        values_std = values.std()

        # Normalize the values to match the targets' mean and standard deviation
        values_normalized = (values - values_mean) / values_std
        targets_mean = targets.mean()
        targets_std = targets.std()

        # Scale and shift the normalized values to match the target distribution
        scaled_values = values_normalized * targets_std + targets_mean

        return scaled_values


        



    

