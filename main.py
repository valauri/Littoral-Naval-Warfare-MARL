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
import math
import json
import game
import ppo
from ppo import PPO
from ddqn import DDQN
from network import MLP, Value, DMLP
import wandb
import sys

with open('config.json') as config_file:
    config = json.load(config_file)

overall = config.get("overall", {})
environment_setup = config.get("environment_setup", {})
model_selection = config.get("model_selection", {})
hyperparameters = config.get("hyperparameters", {})

WANDA = overall['wandb']

BATCH_SIZE = hyperparameters['batch_size']
TOTAL_TIMESTEPS = hyperparameters['total_timesteps']
TEST_EPISODES = hyperparameters['test_episodes']

TRAINED_RED = environment_setup['trained_red']
SIDE = environment_setup['side']
TRAINED_BLUE = environment_setup['trained_blue']

n_blue = environment_setup['n_blue']
n_red = environment_setup['n_red']
n_red_landingship = environment_setup['n_red_landingship']
transfer_weights = model_selection['transfer_weights']

SAVE_MODELS = overall['save_models']
LANDING_OPS = overall['landing_ops']

D_PATH = os.path.join(os.getcwd(), 'discrete_models')
PATH = os.path.join(os.getcwd(), 'models')

gif_path = os.path.join(os.getcwd(), 'gif')

arg1 = sys.argv[0]
if len(sys.argv) > 1:
    arg2 = sys.argv[1]
if len(sys.argv) > 2:
    arg3 = sys.argv[2]
if len(sys.argv) > 3:
    arg4 = sys.argv[3]
if len(sys.argv) > 4:
    arg5 = sys.argv[4]

"""
print(f'PyTorch version: {torch.__version__}')
print('*'*10)
print(f'_CUDA version: ')
print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')
"""

skip_training = arg2
load_models = arg3
visualize_first_test = arg4

env = Game()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = overall['seed']
random.seed(seed)

ALGO = model_selection['algo']

if not (skip_training == 'true'):

    env.reset(n_blue, n_red)

    if ALGO == "ppo":
        """Initialize PPO"""
        ppo_train = PPO(env, device)
        #ppo_train.init_networks(2, 0)

        """Training"""
        if load_models == 'true':
            if transfer_weights:
                if SIDE == "blue":
                    actor = MLP(4*2+3+12, 4).to(device)
                    actor.load_state_dict(torch.load(os.path.join(PATH, 'blue_actor.pth')))
                    critic = Value((4*2+7**2+3)*2).to(device)
                    critic.load_state_dict(torch.load(os.path.join(PATH, 'blue_critic.pth')))
                    ppo_train.transfer_weights(actor, critic, SIDE)
                    print("Loaded existing Blue actor model and transferred weights", PATH)
                    print("Loaded existing Blue critic model and transferred weights", PATH)
                else:
                    actor = MLP(4*2+3+12, 4).to(device)
                    actor.load_state_dict(torch.load(os.path.join(PATH, 'red_actor.pth')))
                    critic = Value((4*2+7**2+3)*2).to(device)
                    critic.load_state_dict(torch.load(os.path.join(PATH, 'red_critic.pth')))
                    ppo_train.transfer_weights(actor, critic, SIDE)
                    print("Loaded existing Red actor model and transferred weights", PATH)
                    print("Loaded existing Red critic model and transferred weights", PATH)
                    ppo_train.actor.load_state_dict(torch.load(os.path.join(PATH, 'blue_actor.pth')))
                    print("Loaded existing Blue actor model", os.path.join(PATH, 'blue_actor.pth'))
                    ppo_train.critic.load_state_dict(torch.load(os.path.join(PATH, 'blue_critic.pth')))
                    print("Loaded existing Blue critic model", os.path.join(PATH, 'blue_critic.pth'))
            else:
                if TRAINED_BLUE:
                    ppo_train.actor.load_state_dict(torch.load(os.path.join(PATH, 'blue_actor.pth')))
                    print("Loaded existing Blue actor model", os.path.join(PATH, 'blue_actor.pth'))
                    ppo_train.critic.load_state_dict(torch.load(os.path.join(PATH, 'blue_critic.pth')))
                    print("Loaded existing Blue critic model", os.path.join(PATH, 'blue_critic.pth'))
                    ppo_train.actor_optimizer.load_state_dict(torch.load(os.path.join(PATH, 'blue_actor_optimizer.pth')))
                    print("Loaded existing Blue actor optimizer", os.path.join(PATH, 'blue_actor_optimizer.pth'))
                    ppo_train.critic_optimizer.load_state_dict(torch.load(os.path.join(PATH, 'blue_critic_optimizer.pth')))
                    print("Loaded existing Blue critic optimizer", os.path.join(PATH, 'blue_critic_optimizer.pth'))
            if TRAINED_RED and not transfer_weights:
                ppo_train.red_actor.load_state_dict(torch.load(os.path.join(PATH, 'red_actor.pth')))
                print("Loaded existing Red actor model", os.path.join(PATH, 'red_actor.pth'))
                ppo_train.red_critic.load_state_dict(torch.load(os.path.join(PATH, 'red_critic.pth')))
                print("Loaded existing Red critic model", os.path.join(PATH, 'red_critic.pth'))
                ppo_train.red_actor_optimizer.load_state_dict(torch.load(os.path.join(PATH, 'red_actor_optimizer.pth')))
                print("Loaded existing Red actor optimizer", os.path.join(PATH, 'red_actor_optimizer.pth'))
                ppo_train.red_critic_optimizer.load_state_dict(torch.load(os.path.join(PATH, 'red_critic_optimizer.pth')))
                print("Loaded existing Red critic optimizer", os.path.join(PATH, 'red_critic_optimizer.pth'))
            elif TRAINED_RED and transfer_weights:
                actor = MLP(4*2+3+12, 4).to(device)
                actor.load_state_dict(torch.load(os.path.join(PATH, 'red_actor.pth')))
                critic = Value((4*2+7**2+3)*2).to(device)
                critic.load_state_dict(torch.load(os.path.join(PATH, 'red_critic.pth')))
                ppo_train.transfer_weights(actor, critic, SIDE)
                print("Loaded existing Red actor model and transferred weights", PATH)
                print("Loaded existing Red critic model and transferred weights", PATH)

        env.reset(n_blue, n_red)

        all_blue_rewards = []
        all_red_rewards = []

        total_steps = 1

        env.imagen = 0

        red_victory = 0
        blue_victory = 0

        print('\n Training\n')
        
        #env.visualize_grid(show=True)

        ppo_train.learn(TOTAL_TIMESTEPS, SAVE_MODELS)

        print("Training finished.\n")
        
        #fig, axs = plt.subplots(2, 1, constrained_layout=True)
        #axs[0].plot(list(range(0, len(ppo_train.reward_log))), ppo_train.reward_log, 'b', '-')
        #fig.suptitle('Blue rewards')

        #axs[1].plot(list(range(0, len(all_red_rewards))), all_red_rewards, 'r', '-')
        #fig.suptitle('Red rewards')

        #plt.show()

    if ALGO == "ddqn":
        
        ddqn = DDQN(env, device)

        """Training"""
        if load_models == 'true':
            if transfer_weights:
                policy = DMLP(env.observation_space).to(device)
                policy.load_state_dict(torch.load(os.path.join(D_PATH, 'target.pth')))
                target = DMLP(env.observation_space).to(device)
                target.load_state_dict(torch.load(os.path.join(D_PATH, 'target.pth')))
                ddqn.transfer_weights(policy, target)
                print("Loaded existing Blue policy model and transferred weights", D_PATH)
                print("Loaded existing Blue target model and transferred weights", D_PATH)
            else:
                if TRAINED_BLUE:
                    ddqn.policy_net.load_state_dict(torch.load(os.path.join(D_PATH, 'target.pth')))
                    print("Loaded existing Blue policy model", os.path.join(D_PATH, 'target.pth'))
                    ddqn.target_net.load_state_dict(torch.load(os.path.join(D_PATH, 'target.pth')))
                    print("Loaded existing Blue target model", os.path.join(D_PATH, 'target.pth'))
            if TRAINED_RED:
                ddqn.red_policy_net.load_state_dict(torch.load(os.path.join(D_PATH, 'red_target.pth')))
                print("Loaded existing Red policy model", os.path.join(D_PATH, 'red_target.pth'))
                ddqn.red_target_net.load_state_dict(torch.load(os.path.join(D_PATH, 'red_target.pth')))
                print("Loaded existing Red target model", os.path.join(D_PATH, 'red_target.pth'))

        env.reset(n_blue, n_red)
        env.imagen = 0

        ddqn.learn()

        print("Training finished.\n")


if skip_training == 'true':
    #Load actor model

    project_name = ALGO+"_test"
    if WANDA:
        wandb.init(project=project_name)

    env.reset(n_blue, n_red)

    if ALGO == "ppo":
        actor = MLP(env.observation_space-7*7+12, env.action_space).to(device)
        actor.load_state_dict(torch.load(os.path.join(PATH, 'blue_actor.pth')))
        actor.eval()
    if ALGO == "ddqn":
        policy = DMLP(env.observation_space).to(device)
        policy.load_state_dict(torch.load(os.path.join(D_PATH, 'target.pth')))
        policy.eval()
    if ALGO == "ddpg":
        actor = MLP(env.observation_space, env.action_space).to(device)
        actor.load_state_dict(torch.load(os.path.join(PATH, 'ddpg_actor.pth')))

    print(f"Loaded existing BLUE {ALGO} model {PATH}")

    if TRAINED_RED:
        if ALGO == "ppo":
            red_actor = MLP(env.red_observation_space-7*7+12, env.action_space).to(device)
            red_actor.load_state_dict(torch.load(os.path.join(PATH, 'red_actor.pth')))
            red_actor.eval()
        if ALGO == "ddqn":
            red_policy = DMLP(env.red_observation_space).to(device)
            red_policy.load_state_dict(torch.load(os.path.join(D_PATH, 'red_target.pth')))
            red_policy.eval()

        print(f"Loaded existing RED {ALGO} model {PATH}")

    red_episode_win = 0
    blue_episode_win = 0

    heatmap = np.zeros((env.grid_size, env.grid_size))
    coldmap = np.zeros((env.grid_size, env.grid_size))
    
    with torch.no_grad():
        preordained_actions = [env.red1_actions, env.red2_actions, env.red3_actions]

        episodes_with_encounter = 0

        blue_engagements = 0
        red_engagements = 0

        for episode in range(TEST_EPISODES):
            if not LANDING_OPS:
                env.reset(n_blue, n_red)
            else:
                env.reset(n_blue, n_red+n_red_landingship)
            print('\n TEST EPISODE \n', episode)
            for j in range(40):
                
                epochs, penalties, reward = 0, 0, 0

                episode_steps = 0

                actions_for_step = []

                env.engagements.clear()
                env.blue_ew.clear()
                env.red_ew.clear()
                
                for i, ship in enumerate(env.blue_ships):
                    if ship is not None:
                        state = ship.get_obs()
                        if ALGO == "ppo":
                            with torch.no_grad():
                                action, log_prob = actor(torch.tensor(state, dtype=torch.float32, device = device))
                                actions_for_step.append(action.cpu().numpy())
                        if ALGO == "ddqn":
                            rad, msl, mov = policy(torch.tensor(state, dtype=torch.float32, device = device))
                            
                            rad = rad.argmax(dim=0).item()
                            msl = msl.argmax(dim=0).item()
                            mov = mov.argmax(dim=0).item()
                            actions_for_step.append(np.asarray([rad, msl, mov]))
                    else:
                        actions_for_step.append(np.zeros((4)))

                for i, ship in enumerate(env.red_ships):
                    if ALGO == "ppo":
                        if TRAINED_RED and j > 13:
                            if ship is not None:
                                state = ship.get_obs()
                                with torch.no_grad():
                                    action, log_prob = red_actor(torch.tensor(state, dtype=torch.float32, device = device))
                                    actions_for_step.append(action.cpu().numpy())
                            else:
                                actions_for_step.append(np.zeros((4)))
                        else:
                            preordained_red_actions = preordained_actions[i]
                            actions_for_step.append(preordained_red_actions[j])
                    if ALGO == "ddqn":
                        if TRAINED_RED:
                            for i, ship in enumerate(env.red_ships):
                                if ship is not None:
                                    state = ship.get_obs()
                                    rad, msl, mov = red_policy(torch.tensor(state, dtype=torch.float32, device = device))
                                    rad = rad.argmax(dim=0).item()
                                    msl = msl.argmax(dim=0).item()
                                    mov = mov.argmax(dim=0).item()
                                    if j < 20:
                                        mov = np.random.randint(1,5)
                                    actions_for_step.append(np.asarray([rad, msl, mov]))
                                else:
                                    actions_for_step.append(np.zeros((4)))
                        else:
                            actions_for_step.append([np.random.randint(0,1), np.random.randint(0,4), np.random.randint(0,49)])


                observations, rewards, done, _ = env.step(actions_for_step)
                
                if visualize_first_test == 'true' and episode == 0:
                    print("Visualizing test episode step " + str(j))
                    env.visualize_grid(path=gif_path, animation=True)

                if done == 0:
                    break
            
            if env.blue_engagements > blue_engagements or env.red_engagements > red_engagements:
                blue_engagements = env.blue_engagements
                red_engagements = env.red_engagements
                episodes_with_encounter += 1

            blue_episode_win += env.blue_victory
            red_episode_win += env.red_victory
            heatmap += env.heatmap
            coldmap += env.coldmap
    
    if WANDA:
        wandb.finish()

    env.visualize_heatmap(heatmap, coldmap)
                

    print(f'Blue victories: {blue_episode_win}\nBlue sinkings: {env.blue_engagements}\nRed victories: {red_episode_win}\nRed sinkings: {env.red_engagements}\nEpisodes with encounter: {episodes_with_encounter}\n')
