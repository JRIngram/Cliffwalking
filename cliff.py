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



class qlearning():
    location = [None,None]
    epsilon = 0
    q_table = {}
    rand = None
    score = 0
    step_size = 0
    discount = 0
    
    def __init__(self, starting_location, epsilon, step_size, discount):
        self.location = [int(starting_location[0]), int(starting_location[1])]
        self.epsilon = epsilon
        self.rand = random.Random()
        self.step_size = step_size
        self.discount = discount
        
    def get_possible_actions(self, location=None):
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
        possible_actions = self.get_possible_actions()
        state_string = str(self.location[0]) + "," + str(self.location[1])
        for x in range(len(possible_actions)):
            key_string = state_string + "," + str(possible_actions[x][0]) + "," + str(possible_actions[x][1])
            self.query(key_string)
            possible_actions[x].append(self.q_table[key_string])
        
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
        if state not in self.q_table:
                self.q_table[state] = 0
        return self.q_table[state]
    
    def update_table(self, previous_state, action, reward, terminal=False):
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
        self.location[0] = location[0]
        self.location[1] = location[1]
    
    def update_score(self, score):
        self.score = self.score + score     
        
    def finish_episode(self,previous_state, action, reward):
        return False   
        
class td_lambda():
    x=1
    
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

class q_approx():
    location = [None,None]
    epsilon = 0
    q_table = {}
    rand = None
    score = 0
    step_size = 0
    discount = 0
    target_net = None
    current_net = None
    event_memory = None
    
    def __init__(self, starting_location, epsilon, step_size, discount):
        self.location = [int(starting_location[0]), int(starting_location[1])]
        self.epsilon = epsilon
        self.rand = random.Random()
        self.step_size = step_size
        self.discount = discount
        self.event_memory = []
        
        #Set up 'old' network
        self.target_net = Sequential()
        self.target_net.add(Dense(4, input_dim=4, activation='tanh'))
        self.target_net.add(Dense(2, activation='tanh'))
        self.target_net.add(Dense(1, activation='linear'))
        self.target_net.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error', 'accuracy'])
        self.current_net = deepcopy(self.target_net)
        
    def get_possible_actions(self, location=None):
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
        possible_actions = self.get_possible_actions()
        for x in range(len(possible_actions)):
            q_key = [self.location[0],self.location[1], possible_actions[x][0],possible_actions[x][1]]
            approximated_value = self.query(q_key).item()
            possible_actions[x].append(approximated_value)

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
        #Checks the ANN for value of state
        state = np.array([[state[0],state[1], state[2], state[3]]])
        approximate_value = self.target_net.predict(state, batch_size=None)[0,0]
        return approximate_value
    
    def update_table(self, previous_state, action, reward,terminal_state=False):
        memory_entry = [previous_state, [action[0],action[1]], reward, deepcopy(self.location),terminal_state]
        self.event_memory.append(memory_entry)
        #Prepares a memory batch for event recall.
        if len(self.event_memory) >= 32:
            memory_batch = random.sample(self.event_memory, 30)
        else:
            memory_batch = random.sample(self.event_memory, len(self.event_memory) )
        
        #For each memory in the batch
        #Look at the next state and possible rewards from that state
        #Choose the max possible reward from the next state
        #Then perform: target = reward + (self.discount * MAX PREDICTION)
        for x in range(0, len(memory_batch)):
            predicted_action_values = []
            previous_state = memory_batch[x][0]
            action = memory_batch[x][1]
            reward = memory_batch[x][2]
            next_state = memory_batch[x][3]
            memory_terminal_state = memory_batch[x][4]
            possible_actions = self.get_possible_actions(previous_state)
            
            for y in range(0, len(possible_actions)):
                #For each possible action in the next state calculate the potential rewards
                potential_state = [[next_state[0], next_state[1], possible_actions[y][0], possible_actions[y][1]]]
                prepared_query = np.array(potential_state) 
                predicted_action_values.append(self.target_net.predict(prepared_query, batch_size=None)[0,0])
            
            max_prediction = max(predicted_action_values)
            target = np.array([reward])
            if memory_terminal_state == False:
                #IF NOT TERMINAL
                #FROM PREDICTED ACTION VALUES TAKE THE MAX AND PERFORM THE FOLLOWING:
                #target = reward + self.discount * MAX PREDICTION
                target = np.array([reward + (self.discount * max_prediction)])
            training_state = np.array([[previous_state[0], previous_state[1], action[0],action[1]]])
            #print(str("*****\nTarget: " + str(reward + (self.discount * max_prediction))))
            #print(str("Prediction: ") + str(self.target_net.predict(prepared_query, batch_size=None)[0,0]))
            #print(str("Reward: " + str(reward)))
            prediction=self.target_net.fit(training_state, target,epochs=1,verbose=0)
            #print(str("New Prediction: ") + str(self.target_net.predict(prepared_query, batch_size=None)[0,0]))
            y=1
                
                
    def update_score(self, score):
        self.score = self.score + score   
        
    def set_agent_location(self, location):
        self.location[0] = location[0]
        self.location[1] = location[1] 

    def finish_episode(self,previous_state, action, reward):
        self.update_table(previous_state, action, reward,True)
        self.event_memory = []
        self.epsilon = self.epsilon - (self.epsilon * 1/2500)
        return False 

episodes = 5000
#Set up for q_learning grid
q_grid = Grid(0)
qlearn = qlearning(q_grid.location, 0.05, 0.2, 1.0)
q_grid.set_agent(qlearn)

#Set up for dqn grid
q_approx_grid = Grid(episodes)
dqn = q_approx(q_approx_grid.location, 1, 0.2, 1.0)
q_approx_grid.set_agent(dqn)

print(str(q_grid) + "\n\n\n\n\n\n\n\n\n\n\n")
while (q_grid.current_episode < q_grid.episodes) or (q_approx_grid.current_episode < q_approx_grid.episodes):
    #q_learning grid
    if (q_grid.current_episode < q_grid.episodes):
        q_grid.make_move()
    
    #dqn grid
    if (q_approx_grid.current_episode < q_approx_grid.episodes):
        q_approx_grid.make_move()  
   
q_grid.make_move