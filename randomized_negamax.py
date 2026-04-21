from connect4_generic_functions import *

import random
import math

class RandomizedNegamax:

    def __init__(self, tag, lookahead):
        self.tag = tag
        self.random_choice_chance = 0.3
        self.starting_depth = lookahead

    def choose_move(self, state, winner, learn, episode):
        if winner is not None:
            return
        
        p = random.uniform(0, 1)
        if p < self.random_choice_chance: #Randomized Negamax
            return self.randomizedNegamax(state, self.starting_depth, self.tag)[0]
        
        #Else -> Negamax
        return self.negamax(state, self.starting_depth, -math.inf, math.inf, self.tag)[0]

    def randomizedNegamax(self, state, depth, tag):
        #There is already a winner or depth is 0, so return the evaluation
        if ( (depth == 0) or (self.game_winner(state) is not None) ): 
            return None, self.heuristic(state, tag)

        moves = ava_moves(state)
        winning_moves = []
        losing_moves = []
        evaluations = {}
    
        max_evaluation = -math.inf
        min_evaluation = math.inf
        best_move = None

        for move in moves:
            new_state = make_state_from_move(state, move, tag)
            evaluation = -self.randomizedNegamax(new_state, depth-1, -tag)[1]
            evaluations.update({move: evaluation})

            if evaluation > max_evaluation:
                max_evaluation = evaluation
                best_move = move
            if evaluation < min_evaluation:
                min_evaluation = evaluation

            if evaluation > 0: #Winning Move
                winning_moves.append(move)
            else: #Losing Move
                losing_moves.append(move)

        #Random probability to select a non-optimal move
        if( (depth == self.starting_depth) and (max_evaluation < 10000) and (min_evaluation > -10000) ):
            move = None
            if (len(winning_moves) > 0): #Choose from the winning moves
                move = random.choice(winning_moves)
            else:
                move = random.choice(losing_moves) #Choose from the losing moves

            return move, evaluations[move]
        
        return best_move, max_evaluation
    
    def negamax(self, state, depth, alpha, beta, tag):
        #There is already a winner or depth is 0, so return the evaluation
        if ( (depth == 0) or (self.game_winner(state) is not None) ): 
            return None, self.heuristic(state, tag)

        moves = ava_moves(state)
    
        max_evaluation = -math.inf
        best_move = None

        for move in moves:
            new_state = make_state_from_move(state, move, tag)
            evaluation = -self.negamax(new_state, depth-1, -beta, -alpha, -tag)[1]

            if evaluation > max_evaluation:
                max_evaluation = evaluation
                best_move = move

            alpha = max(alpha, max_evaluation)
            if alpha >= beta:
                break
        
        return best_move, max_evaluation

    def game_winner(self, state):
        winner = None
        for i in range(len(state[:,0])-3): #Rows
            for j in range(len(state[0, :])-3): #Columns
                winner = self.square_winner(state[i:i+4, j:j+4]) #Check Winner for the 4x4 grid
                if winner is not None:
                    return winner

        if np.min(np.abs(state[0, :])) != 0: #Check if first row is full
            winner = 0
        
        return winner

    def square_winner(self, square):
        #np.sum(square, axis=0) -> Sum along each column
        #np.sum(square, axis=1) -> Sum along each row
        #np.trace(square) -> Sum along the diagonal
        #np.flip(square,axis=1).trace() -> Sum along the other diagonal
        s = np.append([np.sum(square, axis=0), np.sum(square, axis=1)],
                      [np.trace(square), np.flip(square,axis=1).trace()])
        if np.max(s) == 4:
            winner = 1
        elif np.min(s) == -4:
            winner = -1
        else:
            winner = None
        return winner

    def heuristic(self, state, tag):
        winner = self.game_winner(state)
        if (winner == tag):
            return 10000
        elif (winner == -tag): #Opponent Wins
            return -10000

        max_heuristic = 0
        min_heuristic = 0
        for i in range(len(state[:,0])-3): #Rows
            for j in range(len(state[0, :])-3): #Columns
                new_max_heuristic, new_min_heuristic = self.connect4(state[i:i+4, j:j+4]) #Check the 4x4 grid
                if (new_max_heuristic > 1):
                    max_heuristic += new_max_heuristic*2
                if (new_min_heuristic < -1):
                    min_heuristic += new_min_heuristic*2

        if tag == 1:
            return (max_heuristic + min_heuristic)
        else:
            return -(max_heuristic + min_heuristic)

    def connect4(self, square):
        #np.sum(square, axis=0) -> Sum along each column
        #np.sum(square, axis=1) -> Sum along each row
        #np.trace(square) -> Sum along the diagonal
        #np.flip(square,axis=1).trace() -> Sum along the other diagonal
        s = np.append([np.sum(square, axis=0), np.sum(square, axis=1)],
                      [np.trace(square), np.flip(square,axis=1).trace()])

        return np.max(s), np.min(s)
    
    #For compatibility with the Agent classes
    def save_statistics(self, learn, wins, losses, draws, move_count_all_episodes, mean_moves, time):
        pass

    #For compatibility with the Agent classes
    def is_experiment_done(self):
        return False
    
