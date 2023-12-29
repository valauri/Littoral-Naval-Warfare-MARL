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
import network
from network import MLP, Value

with open('config.json') as json_file:
    config = json.load(json_file)

overall = config.get("overall", {})
environment_setup = config.get("environment_setup", {})
model_selection = config.get("model_selection", {})
hyperparameters = config.get("hyperparameters", {})

sys.path.append(os.getcwd())
PATH = os.path.join(os.getcwd(), 'gif')
EW_THRESHOLD = environment_setup["ew_threshold"]
MOVEMENT_THRESHOLD = environment_setup["movement_threshold"]
DISCRETE = overall['discrete']
COA_PATH = overall['coa_path']

CUR_SIDE = environment_setup["side"]

"""
_____________________________________________________________________________________________________________________________________________________________
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

class Combatant:
    def __init__(self, side, ship_type, position, replenishment_points, env):
        self.ship_type = ship_type
        self.side = side
        self.position = position
        self.speed = 2 if ship_type == "medium" else 3
        self.line_of_sight = 4
        self.radar_coverage = 20
        self.missiles = 4 if ship_type == "small" else 8
        self.missile_range = 60
        self.replenishment_points = replenishment_points  # List of replenishment points for the ship
        self.target_list = []
        self.radar_transmission = 1
        self.environment = env
        self.n_actions = env.action_space
        self.n_obs = env.observation_space
        self.steps_done = 0
        self.mast_height = 15 if ship_type == "small" else 30
        
        self.replenishment_timer = 0
        self.replenishment = False
        self.rcs = 0.7 if ship_type == "small" else 1 # radar cross section indicator
        self.successful_engagements = 0
        self.movement_grid = 9

        #self.actor = MLP(self.n_obs, self.n_actions).to(device)
        #self.optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)

        self.grid_threshold = MOVEMENT_THRESHOLD
        self.ew_threshold = EW_THRESHOLD

    def get_obs(self):
        
        env_grid_size = self.environment.grid.shape[0]

        self.target_list.clear()
        
        if self.side == 'blue':
            own_units = self.environment.blue_ships
            opposing_units = self.environment.red_ships
        else:
            own_units = self.environment.red_ships
            opposing_units = self.environment.blue_ships

        observed_opposing_units = []
        electronic_bearings = {}

        for ship in own_units:
            if ship is not None:
                for opponent in opposing_units:
                    if opponent is not None:
                        if self.check_line_of_sight(ship.position, opponent.position, 'radar'):
                            if self.radar_transmission == 1:
                                if math.sqrt((opponent.position[0] - ship.position[0]) ** 2 + (opponent.position[1] - ship.position[1]) ** 2) < self.radar_range(ship, opponent):
                                    if opponent.position not in observed_opposing_units:
                                        observed_opposing_units.append(opponent.position)

                            if math.sqrt((opponent.position[0]-ship.position[0])**2 + (opponent.position[1]-ship.position[1])**2) < 4 and opponent.position not in observed_opposing_units:
                                    observed_opposing_units.append(opponent.position)

                            if math.sqrt((opponent.position[0] - ship.position[0]) ** 2 + (opponent.position[1] - ship.position[1]) ** 2) < self.ew_range(ship, opponent) and opponent.radar_transmission == 1 and self.check_line_of_sight(ship.position, opponent.position, 'ew'):
                                if opponent.position not in observed_opposing_units:
                                    if opponent in electronic_bearings.keys():
                                        electronic_bearings[opponent] += [(ship, ship.calculate_bearing(ship, opponent))]
                                    else:
                                        electronic_bearings[opponent] = [(ship, ship.calculate_bearing(ship, opponent))]
                    
        obs_space = self.environment.grid.copy()
        
        ew_fixes = []
        
        if electronic_bearings:
            for opponent in electronic_bearings.keys():
                if opponent is not None and electronic_bearings[opponent] is not None: # if electronic_bearings[opponent] is not None:
                    if len(electronic_bearings[opponent]) > 1:
                        
                        ew_x = []
                        ew_y = []
                        
                        observations = electronic_bearings[opponent]
                        
                        for i in range(len(observations)):
                            if i+1 < len(observations):
                                bx, by = self.calculate_fixed_position(observations[i][0], observations[i][1], observations[i+1][0], observations[i+1][1])
                                ew_x.append(bx)
                                ew_y.append(by)
        
                        ew_fixes.append((round(np.mean(ew_x)), round(np.mean(ew_y))))
                        if self.side == 'blue':
                            self.environment.blue_ew.append((self.position, (round(np.mean(ew_x)), round(np.mean(ew_y)))))
                        else:
                            self.environment.red_ew.append((self.position, (round(np.mean(ew_x)), round(np.mean(ew_y)))))
            
        for unit_pos in observed_opposing_units:
            x, y = unit_pos
            self.target_list.append((x, y))
            
        for unit_pos in ew_fixes:
            x, y = unit_pos
            if 0 <= x < env_grid_size and 0 <= y < env_grid_size:
                for ship in opposing_units:
                    if ship is not None and math.sqrt((ship.position[0] - x)**2 + (ship.position[1] - y)**2) < 2:
                        self.target_list.append((x, y))

        # obs_space[self.position[0], self.position[1]] = 0.5

        observations = np.zeros(len(own_units)*4 + (self.speed*2 + 1)**2 + 3) # 4 for each own unit (position x and y, radar and missiles), ~7x7 flattened movement grid and one for number of detected targets and one for RAS distance

        start_x, start_y = np.round(self.position)

        start_x = start_x - self.speed
        start_y = start_y - self.speed

        idx = 0

        for x in range(start_x, start_x + self.speed * 2 + 1):
            for y in range(start_y, start_y + self.speed * 2 + 1):
                if 0 <= x < 100 and 0 <= y < 100:
                    observations[idx] = (obs_space[x, y])/255
                    idx += 1
                else:
                    observations[idx] = 0
                    idx += 1

        observations[idx] = self.position[0]/env_grid_size
        idx += 1
        observations[idx] = self.position[1]/env_grid_size
        idx += 1
        observations[idx] = self.radar_transmission
        idx += 1
        observations[idx] = self.missiles/(4 if self.ship_type == 'small' else 8)
        idx += 1

        for ship in own_units:
            if ship is not None:
                if ship != self:
                    observations[idx] = (ship.position[0])/env_grid_size
                    idx += 1
                    observations[idx] = (ship.position[1])/env_grid_size
                    idx += 1
                    observations[idx] = ship.radar_transmission
                    idx += 1
                    observations[idx] = (ship.missiles / (4 if ship.ship_type == 'small' else 8))
                    idx += 1
            else:
                idx += 4

        observations[idx] = (len(self.target_list))
        idx += 1

        """
        ras_dist = 200

        for rrr in self.replenishment_points:

            distance = math.sqrt((self.position[0]-rrr[0])**2 + (self.position[1]-rrr[1])**2)
            if distance < ras_dist:
                ras_dist = distance

        if ras_dist < 1:
            ras_dist = 1
        else:
            ras_dist = 1/ras_dist

        observations[idx] = ras_dist
        idx += 1
        """

        observations[idx] = 0 if self.ship_type != 'ls' else 1
        idx += 1
        

        observations[idx] = self.environment.ducting_factor/2

        return observations
    
    def radar_range(self, ship, opponent):
        d = math.sqrt((4/3)*6370*2)*(math.sqrt((ship.mast_height)/1000) + math.sqrt((opponent.mast_height)/1000))
        d = d/5   #/2.7 #convert first to nautical miles and then to grid squares that equal 2.7 nautical miles each

        return math.ceil(d*opponent.rcs*self.environment.ducting_factor)
    
    def ew_range(self, ship, opponent):

        d = math.sqrt((4/3)*6370*2)*(math.sqrt((ship.mast_height)/1000) + math.sqrt((opponent.mast_height)/1000))
        d = ((d/5))*self.environment.ducting_factor   #/2.7 #convert first to nautical miles and then to grid squares that equal 2.7 nautical miles each
        d = 2*d         # double to approximate the EW detection range

        return math.ceil(d)
            
    def calculate_bearing(self, measuring_ship, target_ship):
        dx = target_ship.position[0] - measuring_ship.position[0]
        dy = target_ship.position[1] - measuring_ship.position[1]
        bearing = math.degrees(math.atan2(dy, dx))

        # add distortion due to inaccuracies in EW measurements
        distortion = random.gauss(0, 1)

        if bearing + distortion < 0:
            bearing = bearing + distortion + 360
        else:
            bearing = bearing + distortion

        # print(f'Calculated bearing: \n Side {measuring_ship.side} \n Bearing {bearing}')
        return bearing
        
    @staticmethod
    def calculate_fixed_position(ship1, bearing_ship1, ship2, bearing_ship2):

        x1, y1 = ship1.position
        x2, y2 = ship2.position
        
        m1 = math.tan(math.radians(bearing_ship1))
        m2 = math.tan(math.radians(bearing_ship2))

        x3 = (m1 * x1 - m2 * x2 + y2 - y1) / (m1 - m2)
        y3 = m1 * (x3 - x1) + y1

        return x3, y3
    
    def return_path(self, node):
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent

        return path[::-1]

    # Code by Nicholas Swift, presented in his Medium article "A* Pathfinding in Python" https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
    def astar(self, maze, start, end):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""

        # Create start and end node
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        max_distance = math.sqrt((0-self.speed)**2 + (0-self.speed)**2)

        iterations = 0
        max_iterations = (self.speed * 2 + 1) ** 2

        adjacent_squares = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Loop until you find the end
        while len(open_list) > 0:
            iterations += 1

            if iterations > max_iterations:
                return self.return_path(current_node)
            # Get the current node
            current_node = open_list[0]

            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

                # Pop current off open list, add to closed list
                open_list.pop(current_index)

                closed_list.append(current_node)

                # Found the goal
                if current_node == end_node:
                    return self.return_path(current_node) # Return reversed path

                # Generate children
                children = []
                for new_position in adjacent_squares: # Adjacent squares

                    # Get node position
                    node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                    # Make sure within grid
                    if node_position[0] > (self.environment.grid.shape[0] - 1) or node_position[0] < 0 or node_position[1] > (self.environment.grid.shape[0] - 1) or node_position[1] < 0:
                        continue

                    # Make sure water depth is adequate, if not, skip
                    if maze[node_position[0], node_position[1]] > self.grid_threshold: # != 0:
                        continue

                    # Create new node
                    new_node = Node(current_node, node_position)

                    # Append
                    children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = math.sqrt((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                if math.sqrt(((child.position[0] - start_node.position[0]) ** 2) + ((child.position[1] - start_node.position[1]) ** 2)) <= max_distance:
                    open_list.append(child)

        return None
    
    # Function that checks weather a path is unobstructed i.e. the destination is reachable within the speed of the ship
    def check_path(self, origin, destination):
        dest_x = destination[0]
        dest_y = destination[1]

        steps = self.speed + 2 # give some leeway for the path to be unobstructed

        if dest_x < 0 or dest_x > 99 or dest_y < 0 or dest_y > 99:
            return False
        
        else:
            grid = self.environment.grid

            path = self.astar(grid, origin, destination)     # AStar(grid).search(origin, destination)

            # Check if any square along the path is a littoral area
            #print(path)

            if path is None or len(path) > steps:
                return False

            for position in path:
                x, y = position

                if self.environment.grid[x, y] > self.grid_threshold: # == 1:  # Check if the square is an obstruction
                    return False
        #if unobstructed:
        return True
    
    # Function that returns a list of points along a line between two points
    def bresenham_line(self, x1, y1, x2, y2):
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        err = dx - dy

        while True:
            points.append((x1, y1))

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return points
    
    # Function that checks if a line of sight exists between two points for both radar and EW sensors
    def check_line_of_sight(self, origin, destination, sensor):

        grid = self.environment.grid

        x1, y1 = origin
        x2, y2 = destination

        line_points = self.bresenham_line(x1, y1, x2, y2)

        if line_points is False:
            return False
        
        for x, y in line_points:
            if sensor == 'radar':
                if grid[x, y] > self.grid_threshold: # == 1:
                    return False
            else:
                if grid[x, y] > EW_THRESHOLD:
                    return False

        return True

    # Function that converts the continuous action space to discrete movement
    def continuous_to_discrete(self, action):

        course = 2*math.pi*action[0]
        distance = self.speed*action[1]

        x, y = self.position
        dx = math.cos(math.degrees(course)) * distance
        dy = math.sin(math.degrees(course)) * distance

        float_x = x + dx
        float_y = y + dy
        new_x = round(float_x)
        new_y = round(float_y)

        if self.can_move_to(new_x, new_y) and self.check_path(self.position, (new_x, new_y)):
            return (new_x, new_y)
        else:
            return False
    
    def move(self, new_position):
        self.position = new_position
        
    # Function that checks if a position is within the grid and if it is a littoral area
    def can_move_to(self, x, y):
        if 0 <= x < 100 and 0 <= y < 100:
            if self.environment.grid[x, y] > self.grid_threshold: # == 1:
                return False
            else:
                return True
        else:
            return False
                
    # Function that defines the available targets for the ship i.e. spotted targets within range of surface-to-surface missiles
    def define_targets(self):
        available_targets = []
        for t in self.target_list:
            if math.sqrt((t[0] - self.position[0]) ** 2 + (t[1] - self.position[1]) ** 2) <= self.missile_range:
                available_targets.append(t)
                
        return available_targets
                
    # Action execution function
    def take_action(self, action):

        rad_action = action[0]
        if DISCRETE:
            engagement = round(action[1])
        else:
            engagement = action[1]

        self.successful_engagements = 0

        if not DISCRETE:
            new_position = self.continuous_to_discrete(action[2:])
        else:
            new_position = self.value_to_coordinates(action[2])

        """
        rrr_dist = 200
        for rrr in self.replenishment_points:
            distance = math.sqrt((self.position[0]-rrr[0])**2 + (self.position[1]-rrr[1])**2)
            if distance < rrr_dist:
                rrr_dist = distance

        if self.missiles == 0 and rrr_dist < 1 and self.replenishment == False:
                self.replenishment_timer += 1
                self.replenishment = True
        """
        engage = False
        engagement_threshold = round(engagement*self.missiles)

        if engagement_threshold > 0:
            engage = True

        destroyed_targets = 0
        
        if engage:
            if self.target_list:
                for tgt in self.target_list:
                    target = self.check_target(tgt, self.side)
                    if self.side == "blue":
                        if target not in self.environment.neutralized_units["red"]:
                            destroyed = self.fire_missile(tgt, self.side, salvo=engagement)
                            if destroyed:
                                destroyed_targets += 1
                    else:
                        if target not in self.environment.neutralized_units["blue"]:
                            destroyed = self.fire_missile(tgt, self.side, salvo=engagement)
                            if destroyed:
                                destroyed_targets += 1

        #if self.target_list and self.side == 'blue':
        #    print(f"Unit on side {self.side} has {len(self.target_list)} targets available at {self.steps_done} timesteps, engagement threshold {engagement_threshold}, engaged {destroyed_targets} targets")

        if self.side == 'blue':
            self.environment.blue_engagements += destroyed_targets
        else:
            self.environment.red_engagements += destroyed_targets

        self.radar_transmission = round(rad_action)

        if new_position is not False:
            self.move(new_position)
            return self.get_obs(), True, (engage, destroyed_targets)
        
        else:
            return self.get_obs(), False, (engage, destroyed_targets)
                    
##############################

    # Checks if the targeted position is within the reach of the surface-to-surface missiles search area
    def check_target(self, target, side):
        if side == 'blue':
            opponent = self.environment.red_ships
        else:
            opponent = self.environment.blue_ships
            
        #if side == 'blue':
        #    print("Target list: ", self.target_list)

        for ship in opponent:
            if ship is not None:
                if math.sqrt((ship.position[0] - target[0]) ** 2 + (ship.position[1] - target[1]) ** 2) <= 3.5: # The targeting accuracy reflects the accuracy of the ship's sensors and the speed of the target, i.e. how far the target may have moved during the missile flight time
                    return ship
        
        return None

    # Function that executes the missile firing logic
    def fire_missile(self, target, side, salvo):
        hit = False
        
        target = self.check_target(target, side)

        num_msl = 0
            
        if target is not None:

            if math.sqrt((target.position[0]-self.position[0])**2+(target.position[1]-self.position[1])**2) < 2:
                # Main gun engagement
                hit = True
            else:
                if self.missiles == 0:
                    print(f'{self.side} unit at {self.position} has no missiles left')
                    return hit
                
                    # Missile firing logic here

                else:
                    detected = True

                    if target.radar_transmission == 1:
                        detected_prob = 0.345 - 0.1
                    else:
                        detected_prob = 0.345 + 0.1

                    if random.random() < detected_prob:
                        detected = False

                    if detected:
                        hit_prob = 0.45
                    else:
                        hit_prob = 0.63

                    if not DISCRETE:
                        num_msl = np.round(self.missiles*salvo)
                    else:
                        if self.ship_type == 'small':
                            num_msl = salvo
                        else:
                            num_msl = salvo*2

                    if num_msl > self.missiles:
                        num_msl = self.missiles

                    self.missiles = self.missiles - num_msl

                    prob = self.calculate_hit_probability(num_msl, hit_prob)

                    if random.random() < prob:
                        hit = True
                        print(f"{self.side} fired {num_msl} missiles from {self.position} to {target.position}. Hit: {hit}")    
                        self.successful_engagements += 1            

                        if CUR_SIDE == 'blue' and self.side == 'blue':
                            self.environment.heatmap[self.position[0], self.position[1]] += 1
                            self.environment.coldmap[target.position[0], target.position[1]] += 1
                        if CUR_SIDE == 'red' and self.side == 'red':
                            self.environment.heatmap[self.position[0], self.position[1]] += 1
                            self.environment.coldmap[target.position[0], target.position[1]] += 1

                        if self.side == 'blue':
                            self.environment.launch_sites['blue'].append((self.position))
                        else:
                            self.environment.launch_sites['red'].append((self.position))
                    else:
                        print(f"{self.side} fired {num_msl} missiles from {self.position} to {target.position}. Hit: {hit}") 
                        
            if hit:
                self.environment.engagements.append((self.position, target.position, num_msl))

                if side == 'blue':
                    idx = self.environment.red_ships.index(target)
                    self.environment.neutralized_units["red"].append(idx)
                    #self.environment.red_ships[idx] = None
                else:
                    idx = self.environment.blue_ships.index(target)
                    self.environment.neutralized_units["blue"].append(idx)
                    #self.environment.blue_ships[idx] = None
                
        return hit
        
                    
    # Function that calculates the probability of a hit for a given number of missiles and probability of hit
    def calculate_hit_probability(self, num_missiles, probability_of_hit):
        #binomial_dist = stats.binom(num_missiles, probability_of_hit)
        #probability = binom.pmf(1, num_missiles, probability_of_hit)

        prob_no_success = (1-probability_of_hit)**num_missiles

        probability = 1 - prob_no_success

        return probability

    def replenish_missiles(self):
        if not self.can_move:
            self.missiles = 4 if self.ship_type == "small" else 8
            print(f"Missiles replenished for {self.position}.")
        else:
            print(f"{self.position} cannot replenish missiles in this turn.")

    def value_to_coordinates(self, value):
        x = value // 7
        y = value % 7

        old_x, old_y = self.position

        if 0 <= old_x - 3 + x < self.environment.grid.shape[0] and 0 <= old_y - 3 + y < self.environment.grid.shape[0]:
            x = old_x - 3 + x
            y = old_y - 3 + y

            if self.check_path(self.position, (x, y)):
                return x, y
            else:
                return False
        else:
            return False