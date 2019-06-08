import numpy as np
from graphics import *

DEFAULT_FILENAMES = [
    # "Data/Test_L1_R1.txt",
    # "Data/Test_L1_R2.txt",
    # "Data/Test_L1_R3.txt",
    # "Data/Test_L2_R1.txt",
    # "Data/Test_L2_R2.txt",
    # "Data/Test_L3_R1.txt"
]

UCI_FILENAMES = ["Data/connect-4_8plytest.csv"]


'''
    X-elem = np array 6 x 7 of (-1, 0, 1) 1 = x
    y-elem = (-1 0 1) 1 = x win
'''

def process(filenames=DEFAULT_FILENAMES):
    X = []
    y = []

    if len(filenames) == 1:
        with open(filename, "r") as f:
            for line in f:
                raw_data = line.split(',')
                rew = 
                board =translate_game(raw_data).flatten()


                X.append(board)
                y.append(int(score))

        print(f"Finished {filename}")

    for filename in filenames:
        with open(filename, "r") as f:
            for line in f:
                raw_data, score = line.split()
                board =translate_game(raw_data).flatten()


                X.append(board)
                y.append(int(score))

        print(f"Finished {filename}")

    return X, y

 # np.concatenate((a,b))