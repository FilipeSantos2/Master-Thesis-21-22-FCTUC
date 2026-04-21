from connect4_generic_functions import make_state_from_move

import numpy as np
import datetime

class ConnectFour:

    def __init__(self, player1, player2, episodes, learn):
        self.players = {1: player1,
                        -1: player2}

        self.state, self.winner_tag, self.winner, self.turn = self.init_game()
        self.memory = {}
        
        self.episodes = episodes+1
        self.learn = learn
        
    def init_game(self):
        return np.zeros((6, 7)), None, None, 1    
        
    def play_multiple_games(self):
        start = datetime.datetime.now()

        statistics = {1: 0, -1: 0, 0: 0}
        move_count_all_episodes = []
        for episode in range(1, self.episodes):
            print( "Game " + str(episode) )
            move_count = self.play_game(episode)
            move_count_all_episodes.append(move_count)
            statistics[self.winner_tag] = statistics[self.winner_tag] + 1

            self.state, self.winner_tag, self.winner, self.turn = self.init_game()
        
        mean_moves = np.mean(move_count_all_episodes)

        end = datetime.datetime.now()
        time = str(end-start)

        self.players[1].save_statistics(self.learn, statistics[1], statistics[-1], statistics[0], move_count_all_episodes, mean_moves, time) #Player 1
        self.players[-1].save_statistics(self.learn, statistics[-1], statistics[1], statistics[0], move_count_all_episodes, mean_moves, time) #Player 2

        self.printStatistics(statistics, mean_moves, time)


    def printStatistics(self, statistics, mean_moves, time):
        print('Statistics')
        print('Player 1 wins:', statistics[1])
        print('Player 2 wins:', statistics[-1])
        print('Draws:', statistics[0])
        print('Number of Moves (mean):', mean_moves)
        print('\nTotal Time = ' + time + '\n')

    def play_game(self, episode):
        move_count = 0

        while self.winner is None:
            player = self.players[self.turn]
            move = player.choose_move(self.state, self.winner_tag, self.learn, episode)

            self.state = make_state_from_move(self.state, move, self.turn)
            self.game_winner()

            self.next_player()
            move_count += 1

        self.play_move(episode) #Add last transition to memory after the game ends
        self.next_player()
        self.play_move(episode) #Add last transition to memory after the game ends (for the other player)
            
        return move_count

    def play_move(self, episode):
        player = self.players[self.turn]
        player.choose_move(self.state, self.winner_tag, self.learn, episode)

    def next_player(self):
        self.turn = -self.turn

    def game_winner(self):
        for i in range(len(self.state[:,0])-3): #Rows
            for j in range(len(self.state[0, :])-3): #Columns
                self.square_winner(self.state[i:i+4, j:j+4]) #Check Winner for the 4x4 grid
                if self.winner is not None:
                    print(self.state)
                    print('Winner: player', self.winner, '\n')
                    return

        if np.min(np.abs(self.state[0, :])) != 0: #Check if first row is full
            self.winner = 0
            self.winner_tag = 0
            print(self.state)
            print('No winner\n')

    def square_winner(self, square):
        #np.sum(square, axis=0) -> Sum along each column
        #np.sum(square, axis=1) -> Sum along each row
        #np.trace(square) -> Sum along the diagonal
        #np.flip(square,axis=1).trace() -> Sum along the other diagonal
        s = np.append([np.sum(square, axis=0), np.sum(square, axis=1)],
                      [np.trace(square), np.flip(square,axis=1).trace()])
        if np.max(s) == 4:
            self.winner = 1
            self.winner_tag = 1
        elif np.min(s) == -4:
            self.winner = 2
            self.winner_tag = -1

