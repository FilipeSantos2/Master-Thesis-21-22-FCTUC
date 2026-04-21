from connect4_generic_functions import *
from connect4_model import ConnectFourModel

import numpy as np

from pathlib import Path
import pickle

class ConnectFourAgent:

    def __init__(self, player, max_memory, seed, experiment):
        if player == 1:
            self.tag = 1
            self.player = 1
            self.player_string = 'First'
        else:
            self.tag = -1
            self.player = 2
            self.player_string = 'Second'

        self.prev_state = np.zeros((6, 7))
        self.prev_move = None
        self.memory = []
        self.count_memory = 0
        self.max_memory = max_memory

        self.seed = seed
        self.experiment = experiment
        
        self.experiment = experiment
        self.statistics = self.load_statistics()

        experiment_done = self.is_experiment_done()
        self.model = ConnectFourModel(self.tag, self.player, self.player_string, self.seed, self.experiment, experiment_done)

    def choose_optimal_move(self, state):
        available_moves = ava_moves(state)
        qs_list = self.model.get_qs_list(state, available_moves)

        index = np.argmax(qs_list)
        move = available_moves[index]

        #print("choose_optimal_move - Available Moves: ", available_moves)
        #print("choose_optimal_move - Qs list:\n", qs_list)
        #print("choose_optimal_move - Move: ", move, "\n")
        return move

    def get_reward(self, winner_tag):
        if winner_tag is None: #Game isn't over
            reward = 0
        else: #Game is over
            if winner_tag == -self.tag: #Loss
                reward = -1
                self.statistics["Training_Wins_List"].append(0)
            elif winner_tag == self.tag: #Win
                reward = 1
                self.statistics["Training_Wins_List"].append(1)
            elif winner_tag == 0: #Draw
                reward = 0.5
                self.statistics["Training_Wins_List"].append(0)

            self.statistics["Training_Rewards_List"].append(reward)
            self.statistics["Training_States_List"].append(self.get_states_len())
        
        return reward
    
    def get_states_len(self): #The child classes of this parent class (ConnectFourAgent) override this function
        pass
    
    def load_to_memory(self, state, move, next_state, reward):
        #print("state\n", state)
        #print("move:", move)
        #print("next_state\n", next_state)
        #print("reward:", reward, "\n")

        afterstate = make_state_from_move(state, move, self.tag)
        self.memory.append([afterstate, next_state, reward])

    def save_statistics(self, learn, wins, losses, draws, move_count_all_episodes, mean_moves, time):
        training_or_testing = "Testing"
        if learn:
            training_or_testing = "Training"
            self.statistics["Training_Moves_List"] = move_count_all_episodes
        else: #The agent finished training and testing
            self.statistics["Finished_Experiment"] = True

        self.statistics[training_or_testing + " - Wins"] = wins
        self.statistics[training_or_testing + " - Losses"] = losses
        self.statistics[training_or_testing + " - Draws"] = draws
        self.statistics[training_or_testing + " - Number of Moves"] = mean_moves
        self.statistics[training_or_testing + " - Time"] = time

        s = 'Agents/' + self.experiment + '/Q_Learning_Agent_Going_' + self.player_string + '_Seed_' + self.seed + '_Statistics.pkl'

        try:
            os.remove(s)
        except:
            pass

        with open(s, 'wb') as output:
            pickle.dump(self.statistics, output)
            
    def load_statistics(self):
        s = 'Agents/' + self.experiment + '/Q_Learning_Agent_Going_' + self.player_string + '_Seed_' + self.seed + '_Statistics.pkl'
        statistics = Path(s)
        if statistics.is_file(): #Load statistics if the agent finished training and testing
            with open(s, 'rb') as input_:
                statistics = pickle.load(input_)
                if (statistics["Finished_Experiment"]):
                    #print('Load Statistics - Player', self.player, "- Seed", self.seed)
                    return statistics
                else: #Create new statistics
                    return self.new_statistics()
        else: #Create new statistics
            return self.new_statistics()
        
    def new_statistics(self):
        print('\nPlayer', self.player, "- Seed", self.seed)
        statistics = {"Training_Rewards_List":[], "Training_Wins_List":[],"Training_Moves_List":[],"Training_States_List":[], "Finished_Experiment":False}
        return statistics
    
    def is_experiment_done(self):
        return self.statistics["Finished_Experiment"]

