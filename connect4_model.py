from connect4_generic_functions import *

import keras.layers as Kl
import keras.models as Km
import tensorflow

from pathlib import Path

class ConnectFourModel:

    def __init__(self, tag, player, player_string, seed, experiment, experiment_done):
        self.tag = tag
        self.player = player
        self.player_string = player_string
        self.seed = seed
        self.experiment = experiment

        self.alpha = 0.8
        self.gamma = 1

        if (experiment_done):
            self.model = self.load_model()
        else:
            self.model = self.create_model()

    def save_model(self):
        s = 'Agents/' + self.experiment + '/Q_Learning_Agent_Going_' + self.player_string + '_Seed_' + self.seed + '_Model.h5'

        try:
            os.remove(s)
        except:
            pass

        self.model.save(s)  

    def load_model(self):        
        s = 'Agents/' + self.experiment + '/Q_Learning_Agent_Going_' + self.player_string + '_Seed_' + self.seed + '_Model.h5'
        model_file = Path(s)

        if model_file.is_file():
            model = Km.load_model(s)
            #print('Load Model: ' + s)
        else:
            model = self.create_model()
        return model

    def create_model(self):
        print('New Model')

        model = self.model_A()

        #model.summary()
        
        return model

    def model_A(self):
        model = Km.Sequential()
        model.add(Kl.Conv2D(20, (4, 4), padding='same', input_shape=(6, 7, 1)))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(20, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(20, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Conv2D(30, (4, 4), padding='same'))
        model.add(Kl.LeakyReLU(alpha=0.3))

        model.add(Kl.Flatten())
        model.add(Kl.Dense(49))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(7))
        model.add(Kl.LeakyReLU(alpha=0.3))
        model.add(Kl.Dense(7))
        model.add(Kl.LeakyReLU(alpha=0.3))

        model.add(Kl.Dense(1, activation='linear'))
        opt = tensorflow.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

        return model  
    
    def get_qs_list(self, state, available_moves):
        afterstates = []
        for move in available_moves:
            afterstate = make_state_from_move(state, move, self.tag) #Create the afterstate
            afterstates.append(afterstate)

        tensors = np.array([afterstate for afterstate in afterstates])
        qs_list = self.model(tensors)

        return qs_list

    def learn_batch(self, memory):
        print('Start learning player', self.player)
        print('Data length:', len(memory), '\n')

        afterstates = np.array([transition[0] for transition in memory])
        afterstates_qs_list = self.model.predict(afterstates)

        x_train = []
        y_train = []

        for index, (afterstate, next_state, reward) in enumerate(memory):
            q_value = afterstates_qs_list[index] #q_value for the current afterstate

            max_q_next_transitions = 0

            if reward == 0: #Game isn't over yet -> calculate max value of the possible transitions
                available_moves = ava_moves(next_state)
                qs_list = self.get_qs_list(next_state, available_moves)

                max_q_next_transitions = np.max(qs_list)

            #If the game is over, then max_q_next_transitions stays as 0, since there are no more transitions

            q_value += self.alpha * (reward + self.gamma * max_q_next_transitions - q_value)

            x_train.append(afterstate)
            y_train.append(q_value)

        self.model.fit(np.array(x_train), np.array(y_train), epochs=5, batch_size=32, verbose=0)

        self.save_model()

