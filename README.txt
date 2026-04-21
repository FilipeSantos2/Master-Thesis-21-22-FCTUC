Connect4.ipynb -> Notebook that imports the other python files.

circuit_builder.py -> Constructs the quantum circuits. Code from the quantum tagged action selection used for Checkers [1].

connect4.py -> Runs the Connect Four games. Based on [2].

connect4_agent.py -> The classical epsilon-greedy Q-Learning agent. Based on [2].

connect4_agent_with_tags.py -> The classical and quantum tagged action selection agents. Based on the previous python file and also adapts and modifies some code from [1].

connect4_generic_functions.py -> Some generic functions.

connect4_model.py -> Takes care of the Deep learning involving the Neural Network. Based on [2].

randomized_negamax.py -> The Randomized Negamax opponent.

Other python files -> Old files that were used in preliminary tests, but weren't used in the final version. Note that there are still a couple of imports and calls to functions related with these files, but none of them are being used and can safely be removed (they were not removed in order to guarantee that the code still runs everything properly without giving any import error, for instance). The code related with the MCTS and the random agents was adapted from [2].

1. https://github.com/ajmcastro/quantum-reinforcement-learning
2. https://github.com/giladariel/Connect4