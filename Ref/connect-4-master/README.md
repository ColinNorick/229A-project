# connect-4

Version of connect-4 and a neural network in Python.

Implemented are two alpha-beta-negamax algorithms, a random move generator and a neural network for playing connect 4.

There are two alpha-beta search algorithms implemented because the first evolved out of a minimax algorithm and the second one is much clearer.

The Neural Network is trained on the UCI connect-4 dataset, that contains all legal 8-ply positionsin which neither player has won yet, and in which the next move is not forced.

