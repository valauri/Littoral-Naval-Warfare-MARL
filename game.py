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
import os.path
import random
from IPython.display import clear_output
from time import sleep
from collections import namedtuple, deque
from scipy.stats import binom
import math
from astar.search import AStar


PATH = os.path.join(os.getcwd(), 'gif')

        
"""
_____________________________________________________________________________________________________________________________________________________________
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDQN(nn.Module):

    def __init__(self, n_actions):
        super(DDQN, self).__init__()
        # 
        self.conv1 = nn.Conv2d(1,5,5,2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(5,15,5,1)
        
        self.fc1 = nn.Linear(1500, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. 
    def forward(self, x):
        x = x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x


class Combatant:
    def __init__(self, game, side, ship_type, position, replenishment_points):
        self.ship_type = ship_type
        self.side = side
        self.position = position
        self.speed = 2 if ship_type == "medium" else 3
        self.line_of_sight = 3
        self.radar_coverage = 20
        self.missiles = 4 if ship_type == "small" else 8
        self.missile_range = 50
        self.replenishment_points = replenishment_points  # List of replenishment points for the ship
        self.can_move = True  # Flag indicating if the ship can move in the current turn
        self.target_list = []
        self.radar_transmission = False
        self.n_actions = 21 if ship_type == "medium" else 29
        self.policy = DDQN(self.n_actions)
        self.target = DDQN(self.n_actions)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.steps_done = 0
        self.environment = game
        
    def get_obs(self):
        self.target_list.clear()
        
        
        if self.side == 'blue':
            own_units = self.environment.blue_ships
            
            observed_opposing_units = []
            electronic_bearings = {}
            
            for ship in own_units:
                for opponent in self.environment.red_ships:
                    if math.sqrt((opponent.position[0] - ship.position[0]) ** 2 + (opponent.position[1] - ship.position[1]) ** 2) < ship.radar_coverage:
                        if random.random() < 0.9 and opponent.position not in observed_opposing_units:
                            observed_opposing_units.append((opponent.position[0], opponent.position[1]))
                    if math.sqrt((opponent.position[0] - ship.position[0]) ** 2 + (opponent.position[1] - ship.position[1]) ** 2) < 75 and opponent.radar_transmission == True:
                        not_obstructed = self.check_path((opponent.position[0], opponent.position[1]), ship.position)
                        if opponent.position not in observed_opposing_units and not_obstructed == True:
                            if opponent in electronic_bearings.keys():
                                electronic_bearings[opponent] += [(ship, self.calculate_bearing(ship, opponent))]
                            else:
                                electronic_bearings[opponent] = [(ship, self.calculate_bearing(ship, opponent))]
            
            obs_space = self.environment.grid.copy()
            
            ew_fixes = []
            
            if electronic_bearings:
                for opponent in electronic_bearings.keys():
                    if electronic_bearings[opponent] is not None:
                        if len(electronic_bearings[opponent]) > 1:
                            
                            ew_x = []
                            ew_y = []
                            
                            observations = electronic_bearings[opponent]
                            
                            for i in range(len(observations)):
                                if i+1 < len(observations):
                                    coord = self.calculate_fixed_position(observations[i][0], observations[i][1], observations[i+1][0], observations[i+1][1])
                                    ew_x.append(coord[0])
                                    ew_y.append(coord[1])
            
                            ew_fixes.append((round(np.mean(ew_x)), round(np.mean(ew_y))))
                            self.environment.blue_ew.append((self.position, (round(np.mean(ew_x)), round(np.mean(ew_y)))))
            
            for unit_pos in own_units:
                x, y = unit_pos.position
                obs_space[x, y] = 2
                
            for unit_pos in observed_opposing_units:
                x, y = unit_pos
                obs_space[x, y] = 3
                self.target_list.append((x, y))
                
            for unit_pos in ew_fixes:
                x, y = unit_pos
                if 0 <= x < 100 and 0 <= y < 100:
                    obs_space[x, y] =  3
                    self.target_list.append((x, y))
                
            obs_space[self.position[0], self.position[1]] = 4

            return obs_space

        else:
            if self.side == 'red':
                own_units = self.environment.red_ships
                
                observed_opposing_units = [] #list for opposing unit positions as (x,y) coordinates
                electronic_bearings = {} #dictionary to store [spotter, bearing] tuples for each opposing unit
                
                for ship in own_units:
                    for opponent in self.environment.red_ships:
                        if math.sqrt((opponent.position[0] - ship.position[0]) ** 2 + (opponent.position[1] - ship.position[1]) ** 2) < ship.radar_coverage and self.check_path(ship.position, opponent.position):
                            if random.random() < 0.9 and opponent.position not in observed_opposing_units:
                                observed_opposing_units.append((opponent.position[0], opponent.position[1]))
                        if math.sqrt((opponent.position[0] - ship.position[0]) ** 2 + (opponent.position[1] - ship.position[1]) ** 2) < 75 and opponent.radar_transmission == True:
                            not_obstructed = self.check_path((opponent.position[0], opponent.position[1]), ship.position)
                            if opponent.position not in observed_opposing_units and not_obstructed == True:
                                if opponent in electronic_bearings.keys():
                                    electronic_bearings[opponent] += [(ship, self.calculate_bearing(ship, opponent))]
                                else:
                                    electronic_bearings[opponent] = [(ship, self.calculate_bearing(ship, opponent))]
                
                obs_space = self.environment.grid.copy()
                
                ew_fixes = []
                
                if electronic_bearings:
                    for opponent in electronic_bearings.keys():
                        if electronic_bearings[opponent] is not None:
                            if len(electronic_bearings[opponent]) > 1:
                                
                                ew_x = []
                                ew_y = []
                                
                                observations = electronic_bearings[opponent]
                                
                                for i in range(len(observations)):
                                    if i+1 < len(observations):
                                        coord = self.calculate_fixed_position(observations[i][0], observations[i][1], observations[i+1][0], observations[i+1][1])
                                        ew_x.append(coord[0])
                                        ew_y.append(coord[1])
                                
                                ew_fixes.append((round(np.mean(ew_x)), round(np.mean(ew_y))))
                                self.environment.red_ew.append((self.position, (round(np.mean(ew_x)), round(np.mean(ew_y)))))
                
                for unit in own_units:
                    x, y = unit.position
                    obs_space[x, y] = 2
                    
            obs_space[self.position[0], self.position[1]] = 4
                    
            for unit_pos in observed_opposing_units:
                x, y = unit_pos
                obs_space[x, y] = 3
                self.target_list.append((x, y))
                
            for unit_pos in ew_fixes:
                x, y = unit_pos
                if 0 <= x < 100 and 0 <= y < 100:
                    obs_space[x, y] = 3
                    self.target_list.append((x, y))
                    
            return obs_space
            
    def calculate_bearing(self, measuring_ship, target_ship):
        dx = target_ship.position[0] - measuring_ship.position[0]
        dy = target_ship.position[1] - measuring_ship.position[1]
        bearing = math.degrees(math.atan2(dy, dx))+random.gauss(0,1) #add distortion
        return bearing
        
    @staticmethod
    def calculate_fixed_position(ship1, bearing_ship1, ship2, bearing_ship2):
        x1, y1 = ship1.position
        x2, y2 = ship2.position
        bearing1 = math.radians(bearing_ship1)
        bearing2 = math.radians(bearing_ship2)

        dx = x2 - x1
        dy = y2 - y1

        distance = math.sqrt(dx**2 + dy**2)

        angle = math.atan2(dy, dx) + bearing1 -bearing2

        x3 = x1 + distance * math.cos(angle)
        y3 = y1 + distance * math.sin(angle)

        return x3, y3
        
    def check_path(self, origin, destination):
        reverse_grid = self.environment.grid
        path = AStar(reverse_grid).search(origin, destination)
        # Check if any square along the path is a littoral area
        #print(path)
        if path is None:
            return False

        for position in path:
            x, y = position

            if self.environment.grid[y, x] == 1:  # Check if the square is an obstruction
                return False
                
        #if unobstructed:
        return True
    
    
    def move(self, new_position):
        self.position = new_position
        
    def radar_silence(self, silence):
        if silence == True:
            self.radar_transmission = False
        else:
            self.radar_transmission = True
        
    def can_move_to(self, x, y):
        #x, y = position
        if 0 <= x < self.environment.grid_size and 0 <= y < self.environment.grid_size:
            if self.environment.grid[x, y] == 1:
                #print(self.environment.grid[x, y])
                return False
            else:
                return True
        else:
            return False
        
    def get_reachable_positions(self, grid_size):
        reachable_positions = []

        for x in range(grid_size):
            for y in range(grid_size):
                distance = math.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
                if distance <= self.speed:
                    reachable_positions.append((x, y))

        return reachable_positions
                
    def define_targets(self):
        available_targets = []
        for t in self.target_list:
            if math.sqrt((t[0] - self.position[0]) ** 2 + (t[1] - self.position[1]) ** 2) <= 100:
                available_targets.append(t)
                
        return available_targets
                
                
    def take_action(self, action):
        # Action returns values for current state space after the action and the success of the action taken as [obs, success]
        
        x, y = self.position
        
        if self.side == 'blue':
            opposing_side = 'red'
        else:
            opposing_side = 'blue'
        
        if action == 0:
            return (self.get_obs(), True)
        
        if action == 1:
            targets = self.define_targets()
            if not targets:
                return (self.get_obs(), False)
            else:
                for t in targets:
                    succesful_engagement = self.fire_missile(t, opposing_side)
                return (self.get_obs(), succesful_engagement)
                
        if action == 2:
            if self.radar_transmission == True:
                self.radar_transmission = False
            else:
                self.radar_transmission = True
            return (self.get_obs(), True)
        
     
        if action == 3: #Move one square North
            destination = (x, y+1)
            if self.can_move_to(x, y+1):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x, y+1))
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            else:
                return (self.get_obs(), False)

                
        if action == 4: #Move two squares North
            destination = (x, y+2)
            if self.can_move_to(x, y+2):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x, y+2))
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            else:
                return (self.get_obs(), False)
                
                
        if action == 5: #Move one square East
            destination = (x+1, y)
            if self.can_move_to(x+1, y):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x+1, y))
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            else:
                return (self.get_obs(), False)

            
        if action == 6: #Move two squares East
            destination = (x+2, y)
            if self.can_move_to(x+2, y):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x+2, y))
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            else:
                return (self.get_obs(), False)

            
        if action == 7: #Move one square South
            destination = (x, y-1)
            if self.can_move_to(x, y-1):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x, y-1))
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            else:
                return (self.get_obs(), False)

                
        if action == 8: #Move two squares South
            destination = (x, y-2)
            if self.can_move_to(x, y-2):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x, y-2))
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            else:
                return (self.get_obs(), False)
                
        if action == 9: #Move one square West
            destination = (x-1, y)
            if self.can_move_to(x-1, y):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x-1, y))
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            else:
                return (self.get_obs(), False)
            
        if action == 10: #Move two squares West
            destination = (x-2, y)
            if self.can_move_to(x-2, y):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x-2, y))
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            else:
                return (self.get_obs(), False)          
            
        if action == 11:
            for ras in self.replenishment_points:
                if math.sqrt((x - ras[0]) ** 2 + (y - ras[1]) ** 2) <= 1 and self.can_move == True:
                    self.can_move = False
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            return (self.get_obs(), False)
        
                    
        if action == 12:
            if self.can_move == False:
                self.can_move = True
                return (self.get_obs(), True)
            else:
                self.can_move = False
                return (self.get_obs(), True)
                
                
        if action == 13: #Move one square North-East
            destination = (x+1, y+1)
            if self.can_move_to(x+1, y+1):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x+1, y+1))
                    return (self.get_obs(), True)
                else:
                    return (self.get_obs(), False)
            else:
                return (self.get_obs(), False)

                
        if action == 14: #Move two squares North-East
            destination = (x+2, y+2)
            if self.can_move_to(x+2, y+2):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x+2, y+2))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                
    
        if action == 15: #Move one square South-East
            destination = (x+1, y-1)
            if self.can_move_to(x+1, y-1):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x+1, y-1))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False

            
        if action == 16: #Move two squares South-East
            destination = (x+2, y-2)
            if self.can_move_to(x+2, y-2):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x+2, y-2))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False

            
        if action == 17: #Move one square South-West
            destination = (x-1, y-1)
            if self.can_move_to(x-1, y-1):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x-1, y-1))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False

                
        if action == 18: #Move two squares South-West
            destination = (x-2, y-2)
            if self.can_move_to(x-2, y-2):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x-2, y-2))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False

                
        if action == 19: #Move one square North-West
            destination = (x-1, y+1)
            if self.can_move_to(x-1, y+1):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x-1, y+1))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
            
        if action == 20: #Move two squares North-West
            destination = (x-2, y+2)
            if self.can_move_to(x-2, y+2):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x-2, y+2))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                
                
        if action == 21: #Move three squares North-West
            destination = (x-3, y+3)
            if self.can_move_to(x-3, y+3):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x-3, y+3))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                
        if action == 22: #Move three squares North
            destination = (x, y+3)
            #print('Can move: {}'.format(self.can_move_to(x, y+3)))
            if self.can_move_to(x, y+3):
                #print('Path: {}'.format(self.check_path(self.position, destination)))
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x, y+3))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                
        if action == 23: #Move three squares North-East
            destination = (x+3, y+3)
            if self.can_move_to(x+3, y+3):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x+3, y+3))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                
        if action == 24: #Move three squares East
            destination = (x+3, y)
            if self.can_move_to(x+3, y):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x+3, y))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                
        if action == 25: #Move three squares South-East
            destination = (x+3, y-3)
            if self.can_move_to(x+3, y-3):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x+3, y-3))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                
        if action == 26: #Move three squares South
            destination = (x, y-3)
            if self.can_move_to(x, y-3):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x, y-3))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                
        if action == 27: #Move three squares South-West
            destination = (x-3, y-3)
            if self.can_move_to(x-3, y-3):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x-3, y-3))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                
        if action == 28: #Move three squares West
            destination = (x-3, y)
            if self.can_move_to(x-3, y):
                if self.check_path(self.position, destination) and self.can_move == True:
                    self.move((x-3, y))
                    return self.get_obs(), True
                else:
                    return (self.get_obs(), False)
            else:
                return self.get_obs(), False
                    
        
        
##############################
    def check_target(self, target, side):
        if side == 'blue':
            opponent = self.environment.blue_ships
        else:
            opponent = self.environment.red_ships
            
        for ship in opponent:
            if math.sqrt((ship.position[0] - target[0]) ** 2 + (ship.position[1] - target[1]) ** 2) <= 1.5:
                return ship
            else:
                return None


    def fire_missile(self, target, side):
        hit = False
            
        target = self.check_target(target, side)
            
        if target is not None:
            if self.missiles == 0:
                print("No missiles left")
                return hit
                # missile firing logic here
            else:
                if target.ship_type == "small":
                    if self.missiles >= 2:
                        self.missiles = self.missiles - 2
                        prob = self.calculate_hit_probability(2, 0.9)
                        if prob > random.random():
                            hit = True
                        print(f"Fired 2 missiles from {self.position} to {target.position}. Hit: {hit}")
                    else:
                        msl = self.missiles
                        prob = self.calculate_hit_probability(msl, 0.9)
                        if prob > random.random():
                            hit = True
                        self.missiles = 0
                        print(f"Fired {msl} missiles from {self.position} to {target.position}. Hit: {hit}. No missiles left")
                if target.ship_type == "medium":
                    if self.missiles >= 3:
                        self.missiles = self.missiles - 3
                        prob = self.calculate_hit_probability(3, 0.8)
                        if prob > random.random():
                            hit = True
                        print(f"Fired 2 missiles from {self.position} to {target.position}. Hit: {hit}")
                    else:
                        msl = self.missiles
                        prob = self.calculate_hit_probability(msl, 0.8)
                        if prob > random.random():
                            hit = True
                        self.missiles = 0
                        print(f"Fired {msl} missiles from {self.position} to {target.position}. Hit: {hit}. No missiles left")
                if target.ship_type == "large":
                    if self.missiles >= 4:
                        self.missiles = self.missiles - 4
                        prob = self.calculate_hit_probability(4, 0.7)
                        if prob > random.random():
                            hit = True
                        print(f"Fired 2 missiles from {self.position} to {target.position}. Hit: {hit}")
                    else:
                        msl = self.missiles
                        prob = self.calculate_hit_probability(msl, 0.7)
                        if prob > random.random():
                            hit = True
                        self.missiles = 0
                        print(f"Fired {msl} missiles from {self.position} to {target.position}. Hit: {hit}. No missiles left")
                        
            if hit:
                if side == 'blue':
                    self.environment.blue_ships.remove(target)
                else:
                    self.environment.red_ships.remove(target)
                
            return hit
        
                    
                    
    def calculate_hit_probability(self, num_missiles, probability_of_hit):
        #binomial_dist = stats.binom(num_missiles, probability_of_hit)
        probability = binom.pmf(1, num_missiles, probability_of_hit)
        return probability

    def replenish_missiles(self):
        if not self.can_move:
            self.missiles = 4 if self.ship_type == "small" else 8
            print(f"Missiles replenished for {self.position}.")
        else:
            print(f"{self.position} cannot replenish missiles in this turn.")

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
"""
class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # Grid representing the terrain

    def set_littoral_area(self, coordinates):
        for coord in coordinates:
            self.grid[coord[0], coord[1]] = 1  # Set the grid square as littoral area

    def is_littoral(self, x, y):
        return self.grid[x, y] == 1
    
    def get_grid(self):
        return self.grid
    
    def get_grid_pos(self, x, y):
        return self.grid[x, y]
    
    def modify(self, x, y, input):
        self.grid[x, y] = input
        return self.grid
"""

class Game:
    def __init__(self):
        self.grid = np.zeros((100, 100)) #Grid(100)  # Create a 100x100 grid
        self.grid_size = 100
        self.blue_ships = []
        self.red_ships = []
        self.blue_replenishment_points = [(89, 84), (74, 90)]
        self.red_replenishment_points = [(23,10), (55,4)]
        self.num_blue = 2
        self.num_red = 2
        self.imagen = 0
        self.red_ew = []
        self.blue_ew = []

    def get_grid(self):
        return self.grid

    def get_grid_pos(self, x, y):
        return self.grid[x, y]

    def is_littoral(self, x, y):
        return self.grid[x, y] == 0
    
    def set_littoral_area(self, coordinates):
        for coord in coordinates:
            self.grid[coord[0], coord[1]] = 0  # Set the grid square as littoral area

    def create_ship(self, side, ship_type, position, replenishment):
        
        if ship_type == "small":
            ship = Combatant(self, side, ship_type, position, replenishment)
            if side == "blue":
                self.blue_ships.append(ship)
            elif side == "red":
                self.red_ships.append(ship)
                
        if ship_type == "medium":
            ship = Combatant(self, side, ship_type, position, replenishment)
            if side == "blue":
                self.blue_ships.append(ship)
            elif side == "red":
                self.red_ships.append(ship)
                
        if ship_type == "large":
            ship = Combatant(self, side, ship_type, position, replenishment)
            if side == "blue":
                self.blue_ships.append(ship)
            elif side == "red":
                self.red_ships.append(ship)
            
    def move_ship(self, ship, destination):
        current_position = ship.position
        path = AStar(np.array(self.environment.grid.get_grid())).search(current_position, destination)

        # Check if any square along the path is a littoral area
        for position in path:
            x, y = position
            if self.grid.grid[y, x] == 0:  # Check if the square is a littoral area
                return False

        # Move the ship to the destination
        ship.position = destination
        return True
        
    def calculate_joint_reward(self, obs, side):
        count = np.count_nonzero(obs == 3)
        
        reward = count
                
        reward -= 1
        
        return reward
        
        

    def initialize_game(self):

        self.define_grid_from_image("balt_mod_400x400.png", 100)
        self.imagen = 0
        
        # Create ships for blue side
        self.blue_ships.clear()
        self.create_ship("blue", "small", (90, 80), self.blue_replenishment_points)
        self.create_ship("blue", "small", (80, 80), self.blue_replenishment_points)

        self.num_blue = len(self.blue_ships)
        #self.create_ship("blue", "medium", (70, 80), self.blue_replenishment_points)

        # Create ships for red side
        self.red_ships.clear()
        #self.create_ship("red", "medium", (50, 12), self.red_replenishment_points)
        self.create_ship("red", "large", (45, 15), self.red_replenishment_points)
        self.create_ship("red", "large", (40, 12), self.red_replenishment_points)

        self.num_red = len(self.red_ships)


    def print_grid(self):
        print(self.grid.grid)

    def step(self, unit, action):
        obs, success = unit.take_action(action)
        #print('Action success: {}'.format(success))

        reward = self.calculate_joint_reward(obs, unit.side)
        done = 1
        if unit.side == 'blue':
            if len(self.red_ships) == 0:
                reward += 10
                done = 0
            if len(self.blue_ships) == 0:
                reward -= 10
                done = 0
        if unit.side == 'red':
            if len(self.blue_ships) == 0:
                reward += 10
                done = 0
            if len(self.red_ships) == 0:
                reward -= 10
                done = 0

        if not success:
            reward -= 1

        return obs, reward, done, {}

    def play_game(self):
        # Game logic goes here
        self.initialize_game()
        self.print_grid()

        # Example turn-based gameplay for one round
        for ship in self.blue_ships:
            ship.fire_missile(self.red_ships[0])
            ship.can_move = False
            ship.replenish_missiles()

        for ship in self.red_ships:
            ship.move((85, 85))

        self.print_grid()
        
    def define_grid_from_image(self, image_path, grid_size):
        # Load and resize the image
        image = Image.open(image_path)
        resized_image = image.resize((grid_size, grid_size), Image.ANTIALIAS)

        # Convert the image to grayscale
        grayscale_image = resized_image.convert("L")

        # Threshold the grayscale image
        threshold_value = 128
        thresholded_image = grayscale_image.point(lambda x: 0 if x < threshold_value else 255, "1")

        # Create the game grid
        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

        for x in range(grid_size):
            for y in range(grid_size):
                pixel = thresholded_image.getpixel((x, y))
                if pixel == 255:
                    grid[y, x] = 1  # Open sea
                else:
                    grid[y, x] = 0  # Littoral area

        self.grid = grid
        
    def visualize_grid(self):

        grid_data = self.grid

        # Create a figure and axes
        fig, ax = plt.subplots()
        ax.set_aspect("equal")  # Set aspect ratio to make squares appear as squares

        # Plot the grid
        ax.imshow(grid_data, cmap="binary", origin="lower")
        ax.set_xticks(np.arange(-0.5, self.grid.size))
        ax.set_yticks(np.arange(-0.5, self.grid.size))
        ax.grid(True, color='black', linestyle='-', linewidth=0.5)

        # Plot ship positions
        for ship in self.blue_ships:
            ship_position = (ship.position[0] + 0.0, ship.position[1] + 0.0)
            if ship.ship_type == 'small':
                ax.plot(*ship_position, "bo", markersize=4, label="Small Combatant")
            elif ship.ship_type == 'medium':
                ax.plot(*ship_position, "bo", markersize=6, label="Medium Combatant")
            elif ship.ship_type == 'large':
                ax.plot(*ship_position, "bo", markersize=8, label="Large Combatant")

        for ship in self.red_ships:
            ship_position = (ship.position[0] + 0.0, ship.position[1] + 0.0)
            if ship.ship_type == 'small':
                ax.plot(*ship_position, "ro", markersize=4, label="Small Combatant")
            elif ship.ship_type == 'medium':
                ax.plot(*ship_position, "ro", markersize=6, label="Medium Combatant")
            elif ship.ship_type == 'large':
                ax.plot(*ship_position, "ro", markersize=8, label="Large Combatant")

        # Plot field of vision
        for ship in self.blue_ships + self.red_ships:
            if ship.rad_transmission:
                x, y = ship.position
                radius = ship.radar_coverage
                x = max(0, min(x, self.grid_size - 1))
                y = max(0, min(y, self.grid_size - 1))
                circle = Circle((x + 0.0, y + 0.0), radius, alpha=0.2, edgecolor=None)
                ax.add_patch(circle)
        
        if self.blue_replenishment_points:
            for point in self.blue_replenishment_points:
                p = (point[0]+0.0, point[1]+0.0)
                ax.plot(*p, 'bx', markersize=8, label='Replenishment Point')
                
        if self.red_replenishment_points:
            for point in self.red_replenishment_points:
                p = (point[0]+0.0, point[1]+0.0)
                ax.plot(*p, 'rx', markersize=8, label='Replenishment Point')

        # plot ew bearings
        if self.blue_ew:
            for p in self.blue_ew:
                ax.plot(p[0], p[1], 'b-')
        
        if self.red_ew:
            for p in self.red_ew:
                ax.plot(p[0], p[1], 'r-')
        
        # Add legend and grid labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("Game Grid")

        # Show the plot
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        plt.savefig(f"imagen{self.imagen}.png")
        self.imagen += 1
        plt.close(fig)

# Example usage
#game = Game()
#game.initialize_game()
#game.visualize_grid()
#game.play_game()

