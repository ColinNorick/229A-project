import numpy as np
from graphics import *

DEFAULT_FILENAMES = [
    "Data/Test_L1_R1.txt",
    "Data/Test_L1_R2.txt",
    "Data/Test_L1_R3.txt",
    "Data/Test_L2_R1.txt",
    "Data/Test_L2_R2.txt",
    "Data/Test_L3_R1.txt"
]

def process(filenames=DEFAULT_FILENAMES):
    X = []
    y = []

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