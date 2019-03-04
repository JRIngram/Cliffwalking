import random
import time
import copy
from random import Random
from copy import deepcopy
import datetime

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
        #Select an action to take
        move = self.agent.make_move()
        #Execution action and observe reward
        self.location[0] = self.location[0] + move[0]
        self.location[1] = self.location[1] + move[1]
        self.agent.set_agent_location(self.location)
        new_location = copy.deepcopy(self.agent.location)
        reward = self.grid[self.location[0]][self.location[1]]
        if reward == -100 or (self.location[0] == 3 and self.location[1] == 11):
            #Terminal State
            self.agent.remember_state_action(original_location, move, reward, new_location, True)
            self.agent.update_approximater()
            self.agent.reset_approximaters()
            self.finish_episode(original_location, move, reward)
        else:
            #Non-Terminal State
            self.agent.remember_state_action(original_location, move, reward, new_location, False)
            self.agent.update_score(reward)
            self.agent.update_approximater()
            self.agent.reset_approximaters()

    def finish_episode(self, original_location, move, reward):
        self.agent.update_score(reward)
        self.agent.set_agent_location(self.location)
        print(str(self))
        finish_string = str("Episode: " + str(self.current_episode+1) + ". Score: " + str(self.agent.score))
        if(self.agent.location[0] == 3 and self.agent.location[1] == 11):
            finish_string = str("Episode: " + str(self.current_episode+1) + ". Score: " + str(self.agent.score))
            finish_string = finish_string + str("\t\tCorrect location reached!")
        print(finish_string)
        print("\n\n\n\n\n\n\n\n\n")
        #time.sleep(1.75)
        self.agent.scores.append(self.agent.score)
        self.agent.score = 0
        self.current_episode = self.current_episode + 1
        self.agent.finish_episode(self.agent.location, move, reward)
        self.location = [3,0]
        self.agent.set_agent_location(self.location)

class q_approx():
    location = [None,None]
    epsilon = 0
    epsilon_decay = 0
    epsilon_minimum = 0.05
    rand = None
    discount = 0
    target_net = None
    current_net = None
    event_memory = None
    memory_size = 0
    sample_size = 0
    steps_taken = 0
    reset_steps = 0

    scores = []
    score = 0

    def __init__(self, starting_location, epsilon, discount, epsilon_decay=0.05, memory_size=1000, sample_size=50, reset_steps = 500):
        self.location = [int(starting_location[0]), int(starting_location[1])]
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rand = random.Random()
        self.discount = discount
        self.event_memory = []
        self.memory_size = memory_size
        self.sample_size = sample_size
        self.reset_steps = reset_steps

        #Initialize action-value function Q with random weights
        self.current_net = Sequential()
        self.current_net.add(Dense(3, input_dim=4, activation='tanh'))
        self.current_net.add(Dense(1, activation='linear'))
        self.current_net.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

        #Initialize target action-value function Q
        self.target_net = deepcopy(self.current_net)

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
        #Gather all action values
        possible_actions = self.get_possible_actions()
        for x in range(len(possible_actions)):
            state_action = np.array([[self.location[0], self.location[1], possible_actions[x][0], possible_actions[x][1]]])
            possible_actions[x].append(self.query(state_action))

        choose_optimal = self.rand.random()
        move = possible_actions[0]
        if choose_optimal > self.epsilon:
            #Choose action with max predicted value
            for x in range(len(possible_actions)):
                if possible_actions[x][2] > move[2]:
                    move = possible_actions[x]
        else:
            #Choose a random action
            x = self.rand.randrange(0,len(possible_actions))
            move = possible_actions[x]
        self.steps_taken = self.steps_taken + 1
        return move

    def query(self, state_action, current_net=True):
        """
        Returns a predicted value for a state-action pair.
        If current_net is true then the current_net is used for this prediction.
        If current_net is false then the target_net is used for this prediction.
        """
        if(current_net == True):
            value_prediction = self.current_net.predict(state_action, batch_size=1)[0,0]
        else:
            value_prediction = self.target_net.predict(state_action, batch_size=1)[0,0]
        return value_prediction

    def update_approximater(self):
        """
        Replays N memories.
        Updates the current_net based on a target which is:
            reward from the state (if terminal state)
            Max predicted reward from next state (if non-terminal state)
        Gradient descent is then performed on the current_net
        """
        if len(self.event_memory) < self.sample_size:
            memory_samples = random.sample(self.event_memory, len(self.event_memory))
        else:
            memory_samples = random.sample(self.event_memory, self.sample_size)
        for memory in memory_samples:
            previous_state = memory[0]
            action = memory[1]
            reward = memory[2]
            next_state = memory[3]

            state_action = np.array([[previous_state[0], previous_state[1], action[0], action[1]]])
            if memory[4] == True:
                target = np.array([reward])
            else:
                #Calculate max potential value from next state
                next_possible_actions = self.get_possible_actions([next_state[0], next_state[1]])
                for x in range(len(next_possible_actions)):
                    state_action = np.array([[next_state[0], next_state[1], next_possible_actions[x][0], next_possible_actions[x][1]]])
                    next_possible_actions[x].append(self.query(state_action, False))

                max_value_move = next_possible_actions[0]
                for x in range(len(next_possible_actions)):
                    if next_possible_actions[x][2] > max_value_move[2]:
                        max_value_move = next_possible_actions[x]


                target = np.array([reward + (self.discount * max_value_move[2])])
            self.current_net.fit(state_action, target,verbose=0)
        return False

    def reset_approximaters(self):
        """
        Sets the target_net to the current_net every fixed amount of steps
        """
        if self.steps_taken % self.reset_steps == 0:
            self.target_net = deepcopy(self.current_net)

    def update_score(self, score):
        """
        Updates the agent score based on the reward received.
        """
        self.score = self.score + score

    def set_agent_location(self, location):
        """
        Moves the agent to a located determined by input parameters.
        """
        self.location[0] = location[0]
        self.location[1] = location[1]

    def finish_episode(self,previous_state, action, reward):
        self.decay_epsilon()
        return False

    def remember_state_action(self, previous_state, action, reward, next_state, terminal):
        """
        Adds a state-action-reward-next_state-terminal_state array to memory
        This can then be replayed in event recall.
        """
        memory = [previous_state, [action[0], action[1]], reward, next_state, terminal]
        self.event_memory.append(memory)
        if len(self.event_memory) > self.memory_size:
            self.event_memory.pop(0)
        return False

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon = self.epsilon * (1 - self.epsilon_decay)
            if self.epsilon < self.epsilon_minimum:
                self.epsilon = self.epsilon_minimum

q_approx_grid = Grid(250)
dqn = q_approx(q_approx_grid.location, 1, 0.99, epsilon_decay=0.01,memory_size=1000, sample_size=32, reset_steps = 500)
q_approx_grid.set_agent(dqn)

print(str(q_approx_grid) + "\n\n\n\n\n\n\n\n\n\n\n")
while(q_approx_grid.current_episode < q_approx_grid.episodes):
    #dqn grid
    q_approx_grid.make_move()

filename = (str("q_approx_results1/results_") + str(datetime.datetime.now()) + str(".txt"))
file = open(filename, "w+", newline="\n")
for x in range(0, len(dqn.scores)):
    file.write(str(x+1) + "," + str(dqn.scores[x]) + "\n")
file.close()

filename = (str("monitoring") + str(".txt"))
file = open(filename, "a", newline="\n")
file.write(str("DQN1 Test completed. Episode reached: " + str(q_approx_grid.current_episode)) + str(" at ") + str(datetime.datetime.now()) + str("\n"))
file.close()

print(str("Initial move up: ") + str(q_approx_grid.agent.query(np.array([[3, 0, -1, 0]]))))
print(str("Initial jump off cliff: ") + str(q_approx_grid.agent.query(np.array([[3, 0, 0, 1]]))))
print(str("Move to goal: ") + str(q_approx_grid.agent.query(np.array([[2, 11, 1, 0]]))))
