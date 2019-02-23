import random
import time
import copy
from random import Random
from copy import deepcopy

#ANN imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense


class Grid():
    grid = []
    start = None
    goal = None
    location = [0,0]
    agent = None
    episodes = 0
    current_episode = 0
    
    def __init__(self, episodes):    
        self.grid = [[-1] * 12,[-1] * 12,[-1] * 12,[-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0]]
        self.start = self.grid[3][0]
        self.goal = self.grid[3][11]
        self.location[0] = 3
        self.location[1] = 0
        self.episodes = episodes
    
    def __str__(self):
        grid_string="____________\n"
        for x in range(len(self.grid)):
            row = ""
            if x < 3:
                for y in range(len(self.grid[x])):
                    if x == self.location[0] and y == self.location[1]:
                        row = str(row) + "|*"
                    else:
                        row = str(row) + "|X"
            
            else:
                for y in range(len(self.grid[x])):
                    if x == self.location[0] and y == self.location[1]:
                        row = row + "|*"
                    elif y == 0:
                        row = str(row) + "|S"
                    elif y == 11:
                        row = str(row) + "|G"
                    else:
                        row = str(row) + "|C"
            grid_string = grid_string + row + "|\n"
        return grid_string
    
    def set_agent(self, agent):
        self.agent = agent
    
    def make_move(self):
        original_location = copy.deepcopy(self.agent.location)
        move = self.agent.make_move()
        self.location[0] = self.location[0] + move[0]
        self.location[1] = self.location[1] + move[1]
        reward = self.grid[self.location[0]][self.location[1]]
        if reward == -100 or (self.location[0] == 3 and self.location[1] == 11):
            if (self.location[0] == 3 and self.location[1] == 11):
                x=1
            self.finish_episode(original_location, move, reward)
        else:
            self.agent.update_score(reward)
            self.agent.set_agent_location(self.location)
            self.agent.update_table(original_location, move, reward)
    
    def finish_episode(self, original_location, move, reward):
        self.agent.update_score(reward)
        self.agent.set_agent_location(self.location)
        self.agent.update_table(original_location, move, reward, True)
        #print(str(grid))
        finish_string = str("Episode: " + str(self.current_episode+1) + ". Score: " + str(self.agent.score))
        if(self.agent.location[0] == 3 and self.agent.location[1] == 11):
            finish_string = str("Episode: " + str(self.current_episode+1) + ". Score: " + str(self.agent.score))
            finish_string = finish_string + str("\t\tCorrect location reached!")
        print(finish_string)
        print("\n\n\n\n\n\n\n\n\n")
        #time.sleep(1.75)
        self.agent.score = 0
        self.current_episode = self.current_episode + 1
        self.agent.finish_episode(self.agent.location, move, reward)
        self.location = [3,0]
        self.agent.set_agent_location(self.location)
        
class td_lambda():
    location = [None,None]
    epsilon = 0
    lookup_table = {}
    eligibilities = {}
    rand = None
    score = 0
    step_size = 0
    discount = 0
    lambda_value = 0
    
    def __init__(self, starting_location, epsilon, step_size, discount, lambda_value):
        self.location = [int(starting_location[0]), int(starting_location[1])]
        self.epsilon = epsilon
        self.rand = random.Random()
        self.step_size = step_size
        self.discount = discount
        self.lambda_value = lambda_value

    
    def get_possible_actions(self):
        possible_actions = []  
        #vertical moves
        if self.location[0] != 0 and self.location[0] != 3:
            action = [1,0]
            possible_actions.append(action)
            action = [-1,0]
            possible_actions.append(action)
        elif self.location[0] == 0:
            action = [1,0]
            possible_actions.append(action)
        elif self.location[0] == 3:
            action = [-1,0]
            possible_actions.append(action)
            
        #horizontal moves
        if self.location[1] != 0 and self.location[1] != 11:
            action = [0,1]
            possible_actions.append(action)
            action = [0,-1]
            possible_actions.append(action)
        elif self.location[1] == 0:
            action = [0,1]
            possible_actions.append(action) 
        elif self.location[1] == 11:
            action = [0,-1]
            possible_actions.append(action)
        return possible_actions
    
    def make_move(self):   
        possible_actions = self.get_possible_actions()
        for x in range(len(possible_actions)):
            key_string = str(self.location[0] + possible_actions[x][0]) + "," + str(self.location[0] + possible_actions[x][1])
            self.query(key_string)
            possible_actions[x].append(self.lookup_table[key_string])
        
        choose_optimal = self.rand.random()
        move = possible_actions[0]
        if choose_optimal > self.epsilon:
            for x in range(len(possible_actions)):
                if possible_actions[x][2] > move[2]:
                    move = possible_actions[x]
        else:
            x = self.rand.randrange(0,len(possible_actions))
            move = possible_actions[x]
        return move
    
    def query(self, state):
        #Checks table for value of state
        if state not in self.lookup_table:
                self.lookup_table[state] = 0
        return self.lookup_table[state]
    
    def update_table(self, previous_state, action, reward):
        current_state_string = str(self.location[0]) + "," + str(self.location[1])
        previous_state_string = str(previous_state[0]) + "," + str(previous_state[1])
        
        current_state_value = self.query(current_state_string)
        previous_state_value = self.query(previous_state_string)
        delta = reward + (self.discount * current_state_value) - previous_state_value
        
        #Calculate eligibility traces
        if previous_state_string not in self.eligibilities:
            self.eligibilities[previous_state_string] = 0
        #if current_state_string not in self.eligibilities:
        #    self.eligibilities[current_state_string] = 0
        
        self.eligibilities[previous_state_string] = (self.discount * self.lambda_value * self.eligibilities[previous_state_string]) + 1
        
        for key in self.lookup_table:
            if key in self.eligibilities:   
                self.lookup_table[key] = self.query(key) + (self.step_size * delta * self.eligibilities[key])
        for key in self.eligibilities:
            if key != previous_state_value:
                self.eligibilities[key] = (self.discount * self.lambda_value * self.eligibilities[key])
        x=1
        
    def update_score(self, score):
        self.score = self.score + score   
        
    def set_agent_location(self, location):
        self.location[0] = location[0]
        self.location[1] = location[1] 
    
    def finish_episode(self,previous_state, action, reward):
        self.update_table(self.location, action, reward)
        self.eligibilities = {}

