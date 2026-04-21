from connect4_agent import ConnectFourAgent
from connect4_generic_functions import ava_moves
from circuit_builder import CircuitBuilder

from qiskit import *
from qiskit.circuit.library import Diagonal, GroverOperator
from qiskit.providers.aer import AerSimulator

import numpy as np
import math
import random

from itertools import count

from pathlib import Path
import pickle

class ConnectFourAgentWithTags(ConnectFourAgent):

    def __init__(self, player, max_memory, seed, experiment, isQuantum, r, delta):
        super().__init__(player, max_memory, seed, experiment)
        
        if ( self.is_experiment_done() ):
            self.flags = self.load_flags()
        else:
            self.flags = self.new_flags()
        self.R = r
        self.delta = delta
        self.iterationsToObtainFlag = []
        self.currentIterationsToObtainFlag = 0

        self.isQuantum = isQuantum
        self.backend = AerSimulator()

    def choose_move(self, state, winner_tag, learn, episode):
        if ( (learn) and (self.prev_move != None) ): #Load the previous transition to memory
            self.load_to_memory(self.prev_state, self.prev_move, state, self.get_reward(winner_tag))
        
        if winner_tag is not None: #End of an episode
            self.count_memory += 1

            self.prev_state = np.zeros((6, 7))
            self.prev_move = None

            if ( (learn) and (self.count_memory == self.max_memory) ):
                mean = np.mean(self.iterationsToObtainFlag)
                print("Player ", self.player, "Mean number of iterations to obtain a flagged action: ", mean)
                self.statistics["Training - Mean_Iterations_Flagged_Action"] = mean
                states_len = self.get_states_len()
                print("Number of states explored: ", states_len, "\n")
                self.statistics["Training - States Explored"] = states_len

                self.count_memory = 0
                # Offline training
                self.model.learn_batch(self.memory)
                self.memory = []

                self.save_flags()
            
            return None

        column = None
        if learn: #Training
            column = self.choose_move_deliberation(state, episode)
        else: #Testing
            column = self.choose_optimal_move(state)

        self.prev_state = state
        self.prev_move = column

        return column

    def get_states_len(self):
        return len(self.flags)

    def choose_move_deliberation(self, state, episode):
        available_moves = ava_moves(state)
        q_values = np.array( self.model.get_qs_list(state, available_moves) )

        state_tuple = tuple(map(tuple, state))
        if state_tuple not in self.flags:
            self.flags[state_tuple] = available_moves

        flags = self.flags[state_tuple]

        #Exploration Policy using Softmax
        temperature = ( 0.2 + (20 - 0.2) / (1 + math.e**(0.35*(episode / self.delta))) )
        shift = q_values - np.max(q_values)
        prob = np.exp((1 / temperature) * shift) / np.sum(np.exp((1 / temperature) * shift))
        
        #Quantum Deliberation
        if (self.isQuantum):
            action, index = self.quantum_deliberation(prob, flags, available_moves)
        else:
            action, index = self.classical_deliberation(prob, flags, available_moves)

        #Update Flags
        if len(q_values) > 1:
            if q_values[index] < 0.0:
                flags = np.delete(flags, np.where(flags == action))
            else:
                if action not in flags:
                    flags = np.append(flags, action)  

            if flags.size == 0:
                f = available_moves
                flags = np.delete(f, np.where(f == action))

        self.flags[state_tuple] = flags

        return action

    def classical_deliberation(self, prob, flags, ava_moves):
        prob = np.concatenate(prob, axis=0) #Turn it into a 1d array
        action = None

        for i_reflection in count():
            index = np.random.choice(prob.size, p=prob)
            action = ava_moves[index]

            self.currentIterationsToObtainFlag += 1

            if action in flags:
                self.iterationsToObtainFlag.append(self.currentIterationsToObtainFlag)
                self.currentIterationsToObtainFlag = 0

            if action in flags or i_reflection + 1 >= self.R:
                break

        return action, index

    def quantum_deliberation(self, prob, flags, ava_moves):
        if (prob.size == 1):
            num_qubits = 1
        else: 
            num_qubits = math.ceil(math.log2(prob.size))

        if prob.size != 2**num_qubits:
            prob = np.append(prob, [0] * (2**num_qubits - prob.size))
        
        len_ava_moves = len(ava_moves)
        epsilon = 0.0
        for i in range(len_ava_moves):
            move = ava_moves[i] 
            if (move in flags): #Move in Flags
                epsilon += prob[i]

        if epsilon >= 1.0:
            epsilon = 1.0

        #print("Available Moves: ", ava_moves)
        #print("prob: ", prob)
        #print("flags: ", flags)
        k = math.ceil(1 / math.sqrt(epsilon))

        U = CircuitBuilder(self.backend).get_U(num_qubits, self.prob_to_angles(prob))

        for i_reflection in count():
            qreg = QuantumRegister(num_qubits, name='q')
            circ = QuantumCircuit(qreg)

            circ.append(U.to_instruction(), qreg)

            m = random.randint(0, k)

            if m > 0:
                grover = GroverOperator(
                    oracle=Diagonal([-1 if i in flags else 1 for i in range(2**num_qubits)]), 
                    state_preparation=U).repeat(m)

                circ.append(grover.to_instruction(), qreg)

            circ.measure_all()

            result = execute(circ, backend=self.backend, shots=1).result()
            counts = result.get_counts(circ)
            #print("counts: ", counts)
            index = int(max(counts, key=counts.get), 2)

            action = ava_moves[index]

            self.currentIterationsToObtainFlag += 1

            if action in flags:
                self.iterationsToObtainFlag.append(self.currentIterationsToObtainFlag)
                self.currentIterationsToObtainFlag = 0

            if action in flags or i_reflection + 1 >= self.R:
                break

        return action, index
    
    def prob_to_angles(self, prob, previous=1.0):
        "Calculates the angles to encode the given probabilities"

        if len(prob) == 2:
            if previous != 0.0:
                return [self.calc_angle(prob[0] / previous)]  
            else:
                return [0.0]

        lhs, rhs = np.split(prob, 2)

        angles = np.array([
            self.calc_angle((np.sum(lhs)/ previous) if previous != 0.0 else 0.0)
        ])

        angles = np.append(angles, self.prob_to_angles(lhs, previous=np.sum(lhs)))
        angles = np.append(angles, self.prob_to_angles(rhs, previous=np.sum(rhs)))

        return angles

    def calc_angle(self, x):
        try:
            return 2 * math.acos(math.sqrt(x))
        except:
            print(x)
            raise()

    def save_flags(self):
        s = 'Agents/' + self.experiment + '/Q_Learning_Agent_Going_' + self.player_string + '_Seed_' + self.seed + '_Flags.pkl'

        try:
            os.remove(s)
        except:
            pass

        with open(s, 'wb') as output:
            pickle.dump(self.flags, output)
            
    def load_flags(self):
        s = 'Agents/' + self.experiment + '/Q_Learning_Agent_Going_' + self.player_string + '_Seed_' + self.seed + '_Flags.pkl'
        flags = Path(s)
        if flags.is_file():
            #print('Load Flags - Player', self.player, "- Seed", self.seed)
            with open(s, 'rb') as input_:
                flags = pickle.load(input_)
                return flags
        else:
            self.new_flags()
        
    def new_flags(self):
        print('New Flags - Player', self.player, '- Seed', self.seed + '\n')
        return dict()

