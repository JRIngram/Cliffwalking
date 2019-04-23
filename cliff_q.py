import random
import time
import copy
import datetime
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
        """
        Creates a 13 by 3 Grid and sets the number of episodes.
        """
        self.grid = [[-1] * 12,[-1] * 12,[-1] * 12,[-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0]]
        self.start = self.grid[3][0]
        self.goal = self.grid[3][11]
        self.location[0] = 3
        self.location[1] = 0
        self.episodes = episodes
    
    def __str__(self):
        """
        Prints the Gridworld and the agent's placement on the grid.
        """
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
        """
        Set the agent to be used in the Gridworld
        """
        self.agent = agent
    
    def make_move(self):
        """
        Have the agent make a move and apply reward/punishments. Store state-action-reward in memory
        """
        original_location = copy.deepcopy(self.agent.location)
        move = self.agent.make_move()
        self.location[0] = self.location[0] + move[0]
        self.location[1] = self.location[1] + move[1]
        reward = self.grid[self.location[0]][self.location[1]]
        if reward == -100 or (self.location[0] == 3 and self.location[1] == 11):
            self.finish_episode(original_location, move, reward)
        else:
            self.agent.update_score(reward)
            self.agent.set_agent_location(self.location)
            self.agent.update_table(original_location, move, reward)
            if self.agent.score <= self.agent.minimum_score:
                self.finish_episode() 
    
    def finish_episode(self, original_location, move, reward):
        """
        Finish the current episode. Print if goal state reached and print final score.
        """
        self.agent.update_score(reward)
        self.agent.set_agent_location(self.location)
        self.agent.update_table(original_location, move, reward)
        self.agent.scores.append(self.agent.score)
        print(str(self))
        finish_string = str("Episode: " + str(self.current_episode+1) + ". Score: " + str(self.agent.score))
        if(self.agent.location[0] == 3 and self.agent.location[1] == 11):
            finish_string = str("Episode: " + str(self.current_episode+1) + ". Score: " + str(self.agent.score))
            finish_string = finish_string + str("\t\tCorrect location reached!")
        print(finish_string)
        print("\n\n\n\n\n\n\n\n\n")
        self.agent.score = 0
        self.current_episode = self.current_episode + 1
        self.agent.finish_episode()
        self.location = [3,0]
        self.agent.set_agent_location(self.location)



class qlearning():
    location = [None,None]
    epsilon = 0
    q_table = {}
    rand = None
    step_size = 0
    discount = 0
    score = 0
    scores = []
    
    def __init__(self, starting_location, epsilon, step_size, discount,epsilon_decay=0.05, epsilon_minumum=0.01, minimum_score=-250):
        self.location = [int(starting_location[0]), int(starting_location[1])]
        self.epsilon = epsilon
        self.rand = random.Random()
        self.step_size = step_size
        self.discount = discount
        self.epsilon_decay = epsilon_decay
        self.epsilon_minimum = epsilon_minumum
        self.minimum_score = minimum_score
        
    def get_possible_actions(self, location=None):
        """
        Returns all valid moves from a location
        If location = None then the agent's current location is used.
        """
        if location == None:
            location = self.location 
        possible_actions = []  
        #vertical moves
        if location[0] != 0 and location[0] != 3:
            action = [1,0]
            possible_actions.append(action)
            action = [-1,0]
            possible_actions.append(action)
        elif location[0] == 0:
            action = [1,0]
            possible_actions.append(action)
        elif location[0] == 3:
            action = [-1,0]
            possible_actions.append(action)
            
        #horizontal moves
        if location[1] != 0 and location[1] != 11:
            action = [0,1]
            possible_actions.append(action)
            action = [0,-1]
            possible_actions.append(action)
        elif location[1] == 0:
            action = [0,1]
            possible_actions.append(action) 
        elif location[1] == 11:
            action = [0,-1]
            possible_actions.append(action)
        return possible_actions
    
    def make_move(self): 
        """
        Gather all possible moves, then chooses either the move with maximum predicted reward or a random move.
        """  
        possible_actions = self.get_possible_actions()
        state_string = str(self.location[0]) + "," + str(self.location[1])
        for x in range(len(possible_actions)):
            key_string = state_string + "," + str(possible_actions[x][0]) + "," + str(possible_actions[x][1])
            possible_actions[x].append(self.query(key_string))
        
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
        """
        Returns a predicted value for a state-action pair.
        """
        #Checks table for value of state
        if state not in self.q_table:
                self.q_table[state] = 0
        return self.q_table[state]
    
    def update_table(self, previous_state, action, reward):
        """
        Updates the look-up table that is queried make moves
        """
        #Partially updates recorded reward of a state by the stepsize
        table_key = str(previous_state[0]) + "," + str(previous_state[1]) + "," + str(action[0]) + "," + str(action[1])
        possible_actions = self.get_possible_actions()
        potential_rewards = []
        state_string = str(self.location[0]) + "," + str(self.location[1])
        for x in range(len(possible_actions)):
            key_string = state_string + "," + str(possible_actions[x][0]) + "," + str(possible_actions[x][1])
            potential_rewards.append(self.query(key_string))
        self.q_table[table_key] = self.q_table[table_key] + (self.step_size * (reward + (self.discount * max(potential_rewards) - self.q_table[table_key])))
        x=0
        
    
    def set_agent_location(self, location):
        """
        Moves the agent to a located determined by input parameters.
        """
        self.location[0] = location[0]
        self.location[1] = location[1]
    
    def update_score(self, score):
        """
        Updates the agent score based on the reward received.
        """
        self.score = self.score + score     
        
    def finish_episode(self):
        """
        Performs the actions related to the ending of an episode.
        """
        self.decay_epsilon()
        return False   
    
    def decay_epsilon(self):
        """
        Decays the epsilon by a fixed amount defined during construction, if epsilon is above epsilon minimum
        """
        if self.epsilon > self.epsilon_minimum:
            self.epsilon = self.epsilon * (1 - self.epsilon_decay)
            if self.epsilon < self.epsilon_minimum:
                self.epsilon = self.epsilon_minimum
        
#Set up for q_learning grid
q_grid = Grid(2000)
qlearn = qlearning(q_grid.location, 1, 0.2, discount=0.99, epsilon_decay=0.01, epsilon_minumum=0.01, minimum_score=-250)
q_grid.set_agent(qlearn)

print(str(q_grid) + "\n\n\n\n\n\n\n\n\n\n\n")
while (q_grid.current_episode < q_grid.episodes):
    q_grid.make_move()

#Saves results to .csv file.
try:
    filename = (str("results/qlearn_") + str(datetime.datetime.now()) + str(".csv"))
    file = open(filename, "w+", newline="\n")
    for x in range(0, len(qlearn.scores)):
        file.write(str(x+1) + "," + str(qlearn.scores[x]) + "\n")
    file.close()
except:
    print("ERROR: creating results file. Have you created the `results` directory?")

