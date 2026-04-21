import numpy as np

def ava_moves(state):
    moves = np.where(state[0, :] == 0)[0]
    return moves

def make_state_from_move(state, move, tag):
    row = np.where(state[:, move] == 0)[0][-1] #Get the first empty space of the column (from the bottom)

    new_state = np.array(state)
    new_state[row, move] = tag #Fill that space

    return new_state

