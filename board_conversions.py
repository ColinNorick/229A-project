import numpy as np
import random

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

def validate_board(board_state):
	'''
	Validates a board in two ways:
	#123123232421
	1. Is there no more than 6 moves in any 1 column
	2. Has no one won yet?
	'''
	if not board_state:
		return True
	if np.max(np.bincount([int(ch) for ch in board_state])) > 6:
		return False
	return not checkWinForValue(board_state, -1) and not checkWinForValue(board_state, 1)

def generate_random_board_state(ply=43):
	'''
	Returns a random string of digits 1-7, which are interpreted as connect four moves
	The parameter ply controls how many moves are given
	There are only 42 moves possible, so a default value of 43 denotes a desire for
	a randomly generated ply

	generates a list with random.sample and join it into a string
	'''
	if ply == 43:
		ply = random.randint(3, 30)
	ints = random.choices(range(1, 8), k=ply)
	potential_board = ''.join(map(str, ints))
	if validate_board(potential_board):
		return potential_board
	else:
		return generate_random_board_state(ply)

def get_all_boards_le(ply):
		return get_boards_recur('', ply)

def get_boards_recur(partial, ply_left):
	stub = [partial]
	if ply_left == 0:
		return stub
	for i in range(1, 8):
		stub += get_boards_recur(partial + str(i), ply_left-1)
	return stub

def checkWinForValue(board, val):
	'''
	Identifies whether or not there is a 4-in-a-row of val in the 2d numpy array board or 1d list of moves
	'''
	if np.shape(board) != (6, 7):
		#we assume that the board is a list of the integer moves made
		board = translate_game(board)

	[boardWidth, boardHeight] = np.shape(board)
	# check horizontal spaces
	for y in range(boardHeight):
		for x in range(boardWidth - 3):
			if board[x, y] == val and board[x+1, y] == val and board[x+2, y] == val and board[x+3, y] == val:
				return True

	# check vertical spaces
	for x in range(boardWidth):
		for y in range(boardHeight - 3):
			if board[x, y] == val and board[x, y+1] == val and board[x, y+2] == val and board[x, y+3] == val:
				return True

	# check / diagonal spaces
	for x in range(boardWidth - 3):
		for y in range(3, boardHeight):
			if board[x, y] == val and board[x+1, y-1] == val and board[x+2, y-2] == val and board[x+3, y-3] == val:
				return True

	# check \ diagonal spaces
	for x in range(boardWidth - 3):
		for y in range(boardHeight - 3):
			if board[x, y] == val and board[x+1, y+1] == val and board[x+2, y+2] == val and board[x+3, y+3] == val:
				return True

	return False



