import numpy as np

# -1 = black/O
# 1 = white/X
# blank space = 0
# assuming the game is a top down matrix or array of 6*7 
def draw_game(game):
    print(" \n \n ")
    counter = 0
    for i in np.nditer(game):
        counter +=1
        if counter % 6 ==1:
            print("\n |", end ="")
        if i ==  -1:
            print (" O |", end ="")
        if i ==  1:
            print (" X |", end ="")
        if i ==  0:
            print (" _ |", end ="")
