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
        if counter % 7 ==1:
            print("\n |", end ="")
        if i ==  -1:
            print (" O |", end ="")
        if i ==  1:
            print (" X |", end ="")
        if i ==  0:
            print (" _ |", end ="")

def translate_game(game):
    game = np.array([int(x) for x in game])
    board = np.zeros((6,7))
    position = {}
    player = 0
    for i in  np.nditer(game):
        i = int(i) - 1

        if i not in position.keys():
            position[i] = 5

        else:
            position[i] = position[i] - 1
        if player % 2 == 0:
            
            board[position[i]][i] = 1
        else:

            board[position[i]][i] = -1
        player = player + 1
    #print (board)
    return board

def conv(c):
    if c == 'o':
        return -1
    elif c == 'x':
        return 1
    elif c == 'b':
        return 0
    else:
        return int(c)

#x,b,c, (x2) ,win / ,draw / ,loss
def translate_uci_game(game):
    board = np.zeros((6, 7))
    for i in range(0, 42):
        board[i%6][i//7] = conv(game[i])
    return board


