from connect4_agent import ConnectFourAgent
from connect4_generic_functions import ava_moves

import numpy as np
import random

class ConnectFourAgentGreedy(ConnectFourAgent):

    def __init__(self, player, max_memory, seed, experiment, starting_epsilon, constant_epsilon):
        super().__init__(player, max_memory, seed, experiment)

        if constant_epsilon:
            self.epsilon = starting_epsilon
        else: #Change epsilon's starting value using the formula
            episode = 1
            self.epsilon = 1 / np.log(episode + 1)
        self.constant_epsilon=constant_epsilon

        self.stateSet = set()
    
    def choose_move(self, state, winner_tag, learn, episode):
        if ( (learn) and (self.prev_move != None) ): #Load the previous transition to memory
            self.load_to_memory(self.prev_state, self.prev_move, state, self.get_reward(winner_tag))
        
        #print("episode ", episode, ", epsilon = ", self.epsilon)

        if winner_tag is not None: #End of an episode
            self.count_memory += 1

            self.prev_state = np.zeros((6, 7))
            self.prev_move = None
            
            if (not self.constant_epsilon): #Epsilon is not constant
                next_episode = episode + 1
                self.epsilon = 1 / np.log(next_episode + 1) #Update epsilon
                
            #print( "updated epsilon = " + str(self.epsilon) ) #Check if epsilon is updating

            if ( (learn) and (self.count_memory == self.max_memory) ):
                states_len = self.get_states_len()
                print("Number of states explored: ", states_len, "\n")
                self.statistics["Training - States Explored"] = states_len

                self.count_memory = 0
                # Offline training
                self.model.learn_batch(self.memory)
                self.memory = []
            
            return None

        column = None
        if learn: #Training
            column = self.epsilon_greedy(state)
        else: #Testing
            column = self.choose_optimal_move(state)

        self.prev_state = state
        self.prev_move = column

        return column
    
    def get_states_len(self):
        return len(self.stateSet)

    def epsilon_greedy(self, state):
        state_tuple = tuple(map(tuple, state))
        self.stateSet.add(state_tuple)

        p = random.uniform(0, 1)
        
        #print("p =", p)
        if p < self.epsilon: #Random
            #print("random move")
            available_moves = ava_moves(state)
            return random.choice(available_moves)
        else: #Optimal
            #print("optimal move")
            return self.choose_optimal_move(state) 

