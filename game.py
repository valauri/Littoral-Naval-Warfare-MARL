import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import sys, os
import os.path
import random
from IPython.display import clear_output
from time import sleep
from collections import namedtuple, deque
import math
#from astar.search import AStar
from skimage.draw import line
import combatant
from combatant import Combatant
from landingship import LandingShip
import matplotlib.patches as patches
import csv
import wandb
from sklearn.cluster import KMeans
import itertools
from itertools import permutations


PATH = os.path.join(os.getcwd(), 'gif')

with open('config.json') as json_file:
    config = json.load(json_file)

overall = config.get("overall", {})
environment_setup = config.get("environment_setup", {})
model_selection = config.get("model_selection", {})
hyperparameters = config.get("hyperparameters", {})

RED_AGGRESSION = environment_setup['red_aggression']
N_BLUE = environment_setup['n_blue']
N_RED = environment_setup['n_red']
N_RED_LANDINGSHIP = environment_setup['n_red_landingship']
SIDE = environment_setup['side']
TRAINED_RED = environment_setup['trained_red']

DISCRETE = overall['discrete']
LANDING_OPS = overall['landing_ops']
COA_PATH = overall['coa_path']
TACTICS = overall['tactics']

EPISODE_STEPS = hyperparameters['episode_steps']

sys.path.append(os.getcwd())

#######################################
            
"""
_____________________________________________________________________________________________________________________________________________________________
"""
            
class AirRecon:
    def __init__(self, position):
        self.position = position
        self.speed = 20
        self.line_of_sight = 3
        self.radar_coverage = 30
        self.play_time = 10
        
    def move(self, new_position):
        x, y = new_position
        
        if 0<= x <= 99 and 0 <= y <= 99:
            self.position = new_position
        else:
            if x < 0:
                x = 0
                self.position = (x, y)
            if x > 99:
                x = 99
                self.position = (x, y)
            if y < 0:
                y = 0
                self.position = (x, y)
            if y > 99:
                y = 99
                self.position = (x, y)
        
    def take_action(self, action):
        x, y = self.position
        if action == 0:
            self.move(x, y+20)
        if action == 1:
            self.move(x, y-20)
        if action == 2:
            self.move(x-20, y)
        if action == 3:
            self.move(x+20, y)
    
        

#_____________________________________________________________________________________________________________________________________________________________


class Game:
    def __init__(self):
        self.grid_size = 100
        self.grid = np.zeros((self.grid_size, self.grid_size))  # Create a 100x100 grid
        self.blue_ships = []
        self.red_ships = []
        self.blue_replenishment_points = [(6, 76), (13, 86)]
        self.red_replenishment_points = [(98,40)] #(80,28)
        self.num_blue = N_BLUE
        self.num_red = N_RED
        self.beta = np.random.beta(1,3)
        self.ducting_factor = 1 + self.beta
        self.imagen = 0
        self.red_ew = []
        self.blue_ew = []
        self.engagements = []

        self.blue = []
        self.red = []

        self.steps_done = 0

        self.action_space = 4
        if not DISCRETE:
            self.observation_space = 60 
        else:
            self.observation_space = 50
        self.blue_movement = 3
        self.red_movement = 3

        self.red_observation_space = 60 

        self.red1_actions = [] # 1st red player actions for pre-determined movement and radar usage profiles
        self.red2_actions = [] # 2nd red player actions for pre-determined movement and radar usage profiles
        self.red3_actions = [] # 3rd red player actions for pre-determined movement and radar usage profiles

        self.red_victory = 0
        self.blue_victory = 0

        self.blue_engagements = 0
        self.red_engagements = 0

        self.red_landing_ships = 0 if LANDING_OPS == False else N_RED_LANDINGSHIP

        self.heatmap = np.zeros((100, 100))
        self.coldmap = np.zeros((100, 100))

        self.coa_path = {'blue': [], 'red': [], 'ls': []}
        self.launch_sites = {"blue": [], "red":[]}

        self.neutralized_units = {"blue": [], "red":[]}
        self.n_blue_left = 0
        self.r_red_left = 0

    def get_grid(self):
        return self.grid

    def get_grid_pos(self, x, y):
        return self.grid[x, y]

    def is_littoral(self, x, y):
        return self.grid[x, y] == 1
    
    def set_littoral_area(self, coordinates):
        for coord in coordinates:
            self.grid[coord[0], coord[1]] = 1  # Set the grid square as littoral area

    # Defines the pre-determined movement and radar usage profiles for the red players according to a separate csv file
    def define_red_actions(self, list, file):
        file_path = os.path.join(os.getcwd(), file)

        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                float_row = [float(cell) for cell in row]
                list.append(float_row)

    def create_ship(self, side, ship_type, position, replenishment, landing_spot=None):
        
        if ship_type == "small":
            ship = Combatant(side, ship_type, position, replenishment, self)
            if side == "blue":
                self.blue_ships.append(ship)
            elif side == "red":
                self.red_ships.append(ship)
                
        if ship_type == "medium":
            ship = Combatant(side, ship_type, position, replenishment, self)
            if side == "blue":
                self.blue_ships.append(ship)
            elif side == "red":
                self.red_ships.append(ship)
                
        if ship_type == "large":
            ship = Combatant(side, ship_type, position, replenishment, self)
            if side == "blue":
                self.blue_ships.append(ship)
            elif side == "red":
                self.red_ships.append(ship)

        if ship_type == "ls":
            ship = LandingShip(side, ship_type, position, landing_spot, self)
            if side == "blue":
                self.blue_ships.append(ship)
            elif side == "red":
                self.red_ships.append(ship)
        
    def calculate_reward(self, unit, movement, engagement, earlier_hits):
        
        reward = 0

        #reward = - unit.steps_done * 0.05
        unit.steps_done += 1

        if len(unit.target_list) > 0:
            reward += len(unit.target_list) * 3

        movement_success = movement
        engagement_threshold, n_hit = engagement

        
        if movement_success:
            #if unit.side == "blue":
            reward += 1 # reward for successful movement
        else:
            reward = max(reward-0.5, 0) # penalty for unsuccessful movement

        if len(unit.target_list) > 0 and not engagement_threshold:
            reward = reward/2 #reward = max(reward-5, 0) # penalty for not engaging while having a target
        elif len(unit.target_list) > 0 and engagement_threshold and n_hit == 0:
            reward += 0.5 # reward for engaging while having a target
        #elif len(unit.target_list) > 0 and engagement_threshold and n_hit > 0:
            
        reward += n_hit*10 # reward for hitting a target

        #if engagement_threshold and n_hit == 0:
        #    reward = max(reward-0.5, 0) # small penalty for engaging but not hitting i.e. too small salvo

        """
        if ras:
            reward += 1 # reward for succesful replenishment

        for ship in self.blue_ships if unit.side == "blue" else self.red_ships:
            if ship is not None and ship.missiles == 0:
                replehnishment_distance = 200
                for rrr in ship.replenishment_points:
                    if math.sqrt((ship.position[0]-rrr[0])**2 + (ship.position[1]-rrr[1])**2) < replehnishment_distance:
                        replehnishment_distance = math.sqrt((ship.position[0]-rrr[0])**2 + (ship.position[1]-rrr[1])**2)
                reward += (1/(max(replehnishment_distance, 1)))*2
        """

        if unit.side == "red" and unit.ship_type != "ls" and TACTICS != "aggressive":
            if unit.steps_done > 14:
                if unit.position[0] < 19 or unit.position[0] > 55 or unit.position[1] < 40 or unit.position[1] > 70:
                    reward = max(reward-2, 0) # penalty for leaving the operational area
                else:
                    reward += 1

        if unit.side == "red" and TACTICS == "aggressive" and unit.ship_type != "ls":
            focal_point = (15, 60)

            nominator = max(math.sqrt((unit.position[0]-focal_point[0])**2 + (unit.position[1]-focal_point[1])**2), 1)
            denominator = (math.sqrt((4/3)*6370*2)*(math.sqrt((unit.mast_height)/1000) + math.sqrt((15)/1000)))/5

            d = (1/(nominator/denominator))*1

            reward += d

        if unit.ship_type == "ls":
            distance_to_landing = math.sqrt((unit.position[0]-unit.landing_zone[0])**2 + (unit.position[1]-unit.landing_zone[1])**2)
            if distance_to_landing > 0:
                if distance_to_landing < unit.distance_to_landing_zone:
                    reward += 2
                    unit.distance_to_landing_zone = distance_to_landing
                else:
                    reward -= 1
            else:
                reward += 100
            
            """
            if distance_to_landing == 0:
                reward += 100
            else:
                reward += (math.log10(100/distance_to_landing))*5
            """
        #if TACTICS == "aggressive":
        #    reward = reward * ((40 / (unit.steps_done+1)))

        return reward
    

    def step(self, action):
        
        self.neutralized_units["blue"].clear()
        self.neutralized_units["red"].clear()

        observations = np.zeros((1, self.num_blue, self.observation_space))
        red_observations = np.zeros((1, self.num_red, self.red_observation_space))

        blue_rewards = []
        red_rewards = []
        done = 1

        blue_nonecounter = 0
        red_nonecounter = 0

        for ship in self.blue_ships:
            if ship is None:
                blue_nonecounter += 1

        for ship in self.red_ships:
            if ship is None:
                red_nonecounter += 1

        ras = False

        blue_hits = 0
        engaging_blue_units = []

        blue_pos = []

        for a, ship in enumerate(self.blue_ships):
            if ship is not None:
                if SIDE == "blue":
                    blue_pos.append(ship.position)
                    """
                    if ship.missiles <= 1:
                        for rrr in ship.replenishment_points:
                            if ship.position[0] == rrr[0] and ship.position[1] == rrr[1]:
                                if DISCRETE:
                                    action[a][2] = 0
                                else:
                                    action[a][2:] = 0
                                ship.missiles = 4 if ship.ship_type == 'small' else 8
                                ras = True
                    """
                    obs, movement, engagement = ship.take_action(action[a])
                    observations[0, a, :] = obs
                    reward = self.calculate_reward(ship, movement, engagement, blue_hits)
                    if engagement[1] > 0:
                        engaging_blue_units.append(a)
                    blue_hits += engagement[1]
                    blue_rewards.append(reward)
                else:
                    #state = ship.get_obs()
                    obs, movement, engagement = ship.take_action(action[a])
                    observations[0, a, :] = obs
                    reward = self.calculate_reward(ship, movement, engagement, blue_hits)
                    if engagement[0]:
                        engaging_blue_units.append(a)
                    blue_hits += engagement[1]
                    blue_rewards.append(reward)

            else:
                blue_rewards.append(0) # No reward for a destroyed ship

        # For Red, the actions are pre-determined in the first training phase
        # therefore, rewards are not regarded and there is no training based on the change in environment
        # the only thing that matters is the movement pattern, radar usage and randomized missile usage
        ### After the training phase, given that Blue side finds a working model to perform tactics, Red side can be trained to fight against it by switching side in configuration file SIDE = "red"

        red_hits = 0
        engaging_red_units = []

        red_pos = []
        for a, ship in enumerate(self.red_ships):
            if ship is not None:
                red_pos.append(ship.position)
                if not TRAINED_RED:
                    #obs = ship.get_obs()
                    if random.random() < RED_AGGRESSION:
                        if ship.target_list is not None:
                            action[len(self.blue_ships) + a][1] = random.random() # random.randint(0, ship.missiles)
                    obs, movement, engagement = ship.take_action(action[len(self.blue_ships) + a])
                    red_observations[0, a, :] = obs
                    reward = self.calculate_reward(ship, movement, engagement, red_hits)
                    if engagement[0]:
                        engaging_red_units.append(a)
                    red_hits += engagement[1]
                    red_rewards.append(reward)
                else:
                    """
                    if ship.missiles <= 1:
                        for rrr in ship.replenishment_points:
                            if ship.position[0] == rrr[0] and ship.position[1] == rrr[1]:
                                if DISCRETE:
                                    action[a][2] = 0
                                else:
                                    action[a][2:] = 0
                                ship.missiles = 4 if ship.ship_type == 'small' else 8
                                ras = True
                    """
                    obs, movement, engagement = ship.take_action(action[len(self.blue_ships) + a])
                    red_observations[0, a, :] = obs
                    reward = self.calculate_reward(ship, movement, engagement, red_hits)
                    red_hits += engagement[1]
                    if engagement[1] > 1:
                        engaging_red_units.append(a)
                    red_rewards.append(reward)
            else:
                red_rewards.append(0) # No reward for a destroyed ship

        blue_new_losses = 0
        red_new_losses = 0

        blue_new_losses = len(self.neutralized_units["blue"])
        self.n_blue_left -= blue_new_losses

        red_new_losses = len(self.neutralized_units["red"])
        self.n_red_left -= red_new_losses

        no_blue_ships_left = False
        no_red_ships_left = False

        if self.n_blue_left == 0:
            no_blue_ships_left = True

        if self.n_red_left == 0:
            no_red_ships_left = True

        for a, ship in enumerate(self.blue_ships):
            if a not in engaging_blue_units and ship is not None:
                blue_rewards[a] += blue_hits*2

        for a, ship in enumerate(self.red_ships):
            if a not in engaging_red_units and ship is not None:
                red_rewards[a] += red_hits*2
                
        if TACTICS != "aggressive":
            if blue_new_losses > 0:
                blue_rewards = [max(r-blue_new_losses*5, 0) for r in blue_rewards]

            if red_new_losses > 0:
                red_rewards = [max(r-red_new_losses*5, 0) for r in red_rewards]

        if no_blue_ships_left and not no_red_ships_left:
            done = 0
            if TACTICS != "aggressive":
                blue_rewards = [r - r for r in blue_rewards]
            red_rewards = [r + 100 for r in red_rewards]
            self.red_victory += 1
            print("Red victory")

        if no_red_ships_left and not no_blue_ships_left:
            done = 0
            blue_rewards = [r + 100 for r in blue_rewards]
            if TACTICS != "aggressive":
                red_rewards = [r - r for r in red_rewards]
            self.blue_victory += 1
            print("Blue victory")

        if no_blue_ships_left and no_red_ships_left:
            done = 0
            blue_rewards = [r + 10 for r in blue_rewards]
            red_rewards = [r + 10 for r in red_rewards]
            # No victory for either side, another solutions applicable if loss of all units is acceptable for either or both sides
            print("Both sides annihilated")

        if LANDING_OPS:
            remaining_landing_ships = []
            for ship in self.red_ships:
                if ship is not None:
                    if ship.ship_type == "ls":
                        remaining_landing_ships.append(ship)

            if not remaining_landing_ships:
                done = 0
                blue_rewards = [r + 100 for r in blue_rewards]
                red_rewards = [r - r for r in red_rewards]
                self.blue_victory += 1
                print("Blue victory")
            else:
                for ls in remaining_landing_ships:
                    if ls.landing_zone[0] == ls.position[0] and ls.landing_zone[1] == ls.position[1]:
                        done = 0
                        blue_rewards = [r - r for r in blue_rewards]
                        red_rewards = [r + 100 for r in red_rewards]
                        self.blue_victory += 1
                        print("Red victory")

        self.steps_done += 1

        if done == 0 or self.steps_done == EPISODE_STEPS-1:
            if COA_PATH:
                for ship in self.blue_ships:
                    if ship is not None:
                        self.coa_path['blue'].append(ship.position)
                for ship in self.red_ships:
                    if ship is not None and ship.ship_type != "ls":
                        self.coa_path['red'].append(ship.position)
                    if ship is not None and ship.ship_type == "ls":
                        self.coa_path['ls'].append(ship.position)

        if self.neutralized_units["blue"]:
            for tgt in self.neutralized_units["blue"]:
                self.blue_ships[tgt] = None
        if self.neutralized_units["red"]:
            for tgt in self.neutralized_units["red"]:
                self.red_ships[tgt] = None

        if blue_pos:
            blue_cog = np.asarray(blue_pos).mean(axis=0)
        else:
            blue_cog = None

        if red_pos:
            red_cog = np.asarray(red_pos).mean(axis=0)
        else:
            red_cog = None

        if blue_cog is not None and red_cog is not None:
            cog_dist = math.sqrt((blue_cog[0]-red_cog[0])**2 + (blue_cog[1]-red_cog[1])**2)
        else:
            cog_dist = None

        if SIDE == "blue":
            return observations, blue_rewards, done, cog_dist # {}
        else:
            return red_observations, red_rewards, done, cog_dist # {}


    def reset(self, n_blue, n_red, grid=None, blue_ships=None, red_ships=None):
        self.steps_done = 0
        self.imagen = 0
        self.ducting_factor = 1 + np.random.beta(1, 3)

        self.blue_victory = 0
        self.red_victory = 0

        if n_red == 2:
            self.define_red_actions(self.red1_actions, 'red_steps.csv')
            self.define_red_actions(self.red2_actions, 'red_steps2.csv')
        elif n_red == 3:
            self.define_red_actions(self.red1_actions, 'red_steps.csv')
            self.define_red_actions(self.red2_actions, 'red_steps2.csv')
            self.define_red_actions(self.red3_actions, 'red_steps3.csv')

        if grid is None:
            self.define_grid_from_image("balt_mod_400x400_2.png", 100)
        else:
            self.grid = grid
        
        if blue_ships is None:
            
            if n_blue == 2:
                blue_pos = [(6, 61), (10, 81)]
            elif n_blue == 3:
                blue_pos = [(6, 61), (10, 81), (8, 70)]
            elif n_blue == 4:
                blue_pos = [(6, 61), (10, 81), (8, 70), (11, 58)]

            self.blue_ships.clear()

            for pos in blue_pos:
                self.create_ship("blue", "small", pos, self.blue_replenishment_points)

            self.num_blue = len(self.blue_ships)

        else:
            self.blue_ships = blue_ships.copy()
            self.num_blue = len(self.blue_ships)

        if red_ships is None:
            
            #if n_red == 2:
            red_pos = [(98, 48), (98, 52)] #(65, 47)]
            
            if n_red == 3 and N_RED_LANDINGSHIP == 0:
                red_pos.append((98,56)) #= [(98, 48), (96, 52), (98, 56)]

            self.red_ships.clear()
            
            for pos in red_pos:
                self.create_ship("red", "large", pos, self.red_replenishment_points)

            self.num_red = len(self.red_ships)
        else:
            self.red_ships = red_ships.copy()
            self.num_red = len(self.red_ships)

        if N_RED_LANDINGSHIP > 0:
            for i in range(N_RED_LANDINGSHIP):
                xs, ys = random.randint(98, 99), random.randint(48, 56)
                self.create_ship("red", "ls", (xs, ys), self.red_replenishment_points, (14, 82))
                self.red_landing_ships += 1

        self.num_red = len(self.red_ships)

        max_speed = 0
        for ship in self.blue_ships:
            if ship.speed > max_speed:
                max_speed = ship.speed
        
        self.blue_movement = max_speed*2+1

        max_speed = 0
        for ship in self.red_ships:
            if ship.speed > max_speed:
                max_speed = ship.speed
        
        self.red_movement = max_speed*2+1

        self.observation_space = (len(self.blue_ships)*4 + (self.blue_movement)**2 + 3) 
        self.red_observation_space = (len(self.red_ships)*4 + (self.red_movement)**2 + 3) 

        self.n_blue_left = len(self.blue_ships)
        self.n_red_left = len(self.red_ships)
        
        
    def define_grid_from_image(self, image_path, grid_size):
        # Load and resize the image
        image = Image.open(image_path)
        resized_image = image.resize((grid_size, grid_size), Image.ANTIALIAS)

        # Convert the image to grayscale
        grayscale_image = resized_image.convert("L")

        grid = np.asarray(grayscale_image)

        self.grid = grid
        
    def visualize_grid(self, show=False, path=None, animation=False):

        #grid_data = np.rot180(self.grid, 1, axes=(0,1))
        grid_data = self.grid

        # Create a figure and axes
        fig, ax = plt.subplots()
        ax.set_aspect("equal")  # Set aspect ratio to make squares appear as squares

        # Plot the grid
        ax.imshow(grid_data, cmap="gray", origin="upper", extent=[-0.5, 100-0.5, -0.5, 100-0.5])
        ax.set_xticks(np.arange(-0.5, self.grid.size))
        ax.set_yticks(np.arange(-0.5, self.grid.size))
        ax.grid(True, color='black', linestyle='-', linewidth=0.5)

        # Plot ship positions
        for ship in self.blue_ships:
            if ship is not None:
                ship_position = (ship.position[1], 100 - ship.position[0] - 1)
                if ship.ship_type == 'small':
                    ax.plot(*ship_position, "bo", markersize=4, label="Small Combatant")
                elif ship.ship_type == 'medium':
                    ax.plot(*ship_position, "bo", markersize=6, label="Medium Combatant")
                elif ship.ship_type == 'large':
                    ax.plot(*ship_position, "bo", markersize=8, label="Large Combatant")

        for ship in self.red_ships:
            if ship is not None:
                ship_position = (ship.position[1], 100 - ship.position[0] - 1)
                if ship.ship_type == 'small':
                    ax.plot(*ship_position, "ro", markersize=4, label="Small Combatant")
                elif ship.ship_type == 'medium':
                    ax.plot(*ship_position, "ro", markersize=6, label="Medium Combatant")
                elif ship.ship_type == 'large':
                    ax.plot(*ship_position, "ro", markersize=8, label="Large Combatant")
                elif ship.ship_type == 'ls':
                    ax.plot(*ship_position, "rs", markersize=6, label="Landing Ship")
                    landing = (ship.landing_spot[1], 100 - ship.landing_spot[0] - 1)
                    ax.plot(*landing, "r*", markersize=6, label="Landing Spot")

        # Plot field of vision
        for ship in self.blue_ships + self.red_ships:
            if ship is not None:
                if ship.radar_transmission == 1:
                    x, y = ship.position
                    radius = ((math.sqrt((4/3)*6370*2)*(math.sqrt((ship.mast_height)/1000) + math.sqrt((30)/1000)))/5)*self.ducting_factor
                    # x = max(radius, min(x, self.grid_size - 1 - radius))
                    # y = max(radius, min(y, self.grid_size - 1 - radius))

                    #radius = min(radius, min(x, y, self.grid_size - 1 - x, self.grid_size - 1 - y))

                    circle = Circle((y, 100-x-1), radius, alpha=0.2, edgecolor=None)
                    ax.add_patch(circle)
        
        if self.blue_replenishment_points:
            for point in self.blue_replenishment_points:
                p = (point[1], 100-point[0]-1)
                ax.plot(*p, 'bv', markersize=5, label='Replenishment Point')
                
        if self.red_replenishment_points:
            for point in self.red_replenishment_points:
                p = (point[1], 100-point[0]-1)
                ax.plot(*p, 'rv', markersize=5, label='Replenishment Point')

        # plot ew bearings
        if self.blue_ew:
            for p in self.blue_ew:
                x1 = p[0][1]
                x2 = p[1][1]
                y1 = 100 - p[0][0] - 1
                y2 = 100 - p[1][0] - 1
                ax.plot([x1, x2], [y1, y2], 'b-')
        
        if self.red_ew:
            for p in self.red_ew:
                x1 = p[0][1]
                x2 = p[1][1]
                y1 = 100 - p[0][0] - 1
                y2 = 100 - p[1][0] - 1
                ax.plot([x1, x2], [y1, y2], 'r-')

        if self.engagements:
            for e in self.engagements:
                launch, target, msl = e
                x1 = launch[1]
                x2 = target[1]
                y1 = 100 - launch[0] - 1
                y2 = 100 - target[0] - 1
                ax.plot(x2, y2, 'X', color="orange")
                if msl == 0:
                    ax.plot([x1, x2], [y1, y2], '-', color="yellow")
                    ax.text(x2, y2, f'Gun engagement')
                else:
                    ax.plot([x1, x2], [y1, y2], '-', color="orange")
                    ax.text(x2, y2, f'{msl} missiles')

            self.engagements.clear()

        red_oa = patches.Rectangle((40, 45), 30, 36, linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(red_oa)

        # Add legend and grid labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("Game Grid")

        ax.set_xlim(0 - 0.5, 100 + 0.5)
        ax.set_ylim(0 - 0.5, 100 + 0.5)

        # print(np.unique(self.grid))

        # Show the plot
        plt.grid(True)
        plt.tight_layout()
        if show:
            plt.show()
        if path is not None:
            plt.savefig(PATH + f"\imagen{self.imagen}.png")
        self.imagen += 1
        plt.close(fig)

    def visualize_heatmap(self, heatmap, coldmap):

        self.reset(self.num_blue, self.num_red)
        #main_image = self.grid.copy()
        image = Image.open("balt_mod_400x400_2.png")
        resized_image = image.resize((100, 100), Image.ANTIALIAS)
    
        plt.imshow(resized_image, origin="upper", extent=[-0.5, 100 - 0.5, -0.5, 100 - 0.5])
        if np.max(heatmap) > 0:
            plt.imshow(heatmap, cmap="hot", alpha=0.25, origin="upper", extent=[-0.5, 100 - 0.5, -0.5, 100 - 0.5])

        blue_start_pos = [(6, 61), (10, 81), (8, 70), (11, 58)]
        red_start_pos = [(98, 48), (96, 52), (98, 56)]

        keys = ['blue', 'red', 'ls']

        for key in keys:
            if key != 'ls':
                launch_n = self.num_blue if key == "blue" else self.num_red

                if key == "red" and LANDING_OPS:
                    launch_n = launch_n - self.red_landing_ships

                if len(self.launch_sites[key]) >= launch_n:

                    launch_s = KMeans(n_clusters=launch_n, random_state=0, n_init='auto').fit(np.asarray(self.launch_sites[key]))
                    sites = launch_s.cluster_centers_

                    for site in sites:
                        y, x = site
                        y = 100 - y - 1
                        plt.plot(x, y, 'yo', markersize=25, alpha=0.2)

                    
                    relevant_start_pos = []

                    if key == "blue":
                        for i in range(self.num_blue):
                            relevant_start_pos.append(blue_start_pos[i])
                    else:
                        for i in range(self.num_red):
                            relevant_start_pos.append(red_start_pos[i])

                    unique_combinations = []

                    permuts = list(itertools.permutations(relevant_start_pos, len(sites)))

                    for combination in permuts:
                        zipped = zip(combination, sites)
                        unique_combinations.append(list(zipped))

                    total_range = 1000

                    for combination in unique_combinations:
                        d_tot = 0
                        for set in combination:
                            start, end = set
                            y1, x1 = start
                            y2, x2 = end
                            d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                            d_tot += d

                        if d_tot < total_range:
                            total_range = d_tot
                            best_combination = combination
                            
                    for pair in best_combination:
                        start, end = pair
                        y1, x1 = start
                        y2, x2 = end
                        y1 = 100 - y1 - 1
                        y2 = 100 - y2 - 1
                        if key == 'blue':
                            plt.arrow(x1, y1, x2-x1, y2-y1, width=0.1, color='blue', shape='full', head_width=2, head_length=2, length_includes_head=True)
                        else:
                            plt.arrow(x1, y1, x2-x1, y2-y1, width=0.1, color='red', shape='full', head_width=2, head_length=2, length_includes_head=True)

                else:
                    last_pos = self.coa_path[key]

                    pos_centers = KMeans(n_clusters=launch_n, random_state=0, n_init='auto').fit(np.asarray(last_pos)).cluster_centers_
                    
                    for site in pos_centers:
                        y, x = site
                        y = 100 - y - 1
                        plt.plot(x, y, 'yo', markersize=25, alpha=0.2)

                    
                    relevant_start_pos = []

                    if key == "blue":
                        for i in range(self.num_blue):
                            relevant_start_pos.append(blue_start_pos[i])
                    else:
                        for i in range(self.num_red):
                            relevant_start_pos.append(red_start_pos[i])

                    unique_combinations = []

                    permuts = list(itertools.permutations(relevant_start_pos, len(pos_centers)))

                    for combination in permuts:
                        zipped = zip(combination, pos_centers)
                        unique_combinations.append(list(zipped))

                    total_range = 1000

                    for combination in unique_combinations:
                        d_tot = 0
                        for set in combination:
                            start, end = set
                            y1, x1 = start
                            y2, x2 = end
                            d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                            d_tot += d

                        if d_tot < total_range:
                            total_range = d_tot
                            best_combination = combination
                            
                    for pair in best_combination:
                        start, end = pair
                        y1, x1 = start
                        y2, x2 = end
                        y1 = 100 - y1 - 1
                        y2 = 100 - y2 - 1
                        if key == 'blue':
                            plt.arrow(x1, y1, x2-x1, y2-y1, width=0.1, color='aqua', shape='full', head_width=2, head_length=2, length_includes_head=True)
                        else:
                            plt.arrow(x1, y1, x2-x1, y2-y1, width=0.1, color='orangered', shape='full', head_width=2, head_length=2, length_includes_head=True)

            else:
                landing_site = KMeans(n_clusters=1, random_state=0, n_init='auto').fit(np.asarray(self.launch_sites[key])).cluster_centers_

                plt.plot(landing_site[0][1], 100 - landing_site[0][0] - 1, 'rs', markersize=25, alpha=0.2)


        plt.colorbar()

        plt.show()

