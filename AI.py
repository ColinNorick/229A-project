import sklearn
import board_conversions as bc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

def undo_what_we_did_before(pos, board_rank):
	if len(pos) % 2 == 1:
		return [-x for x in board_rank]
	return board_rank

def who_wins(board_rank):
	max_score = np.max(board_rank)
	if max_score > 0:
		return 1
	elif max_score < 0:
		return -1
	else:
		return 0


def find_all_moves(board_rank):
	max_score = np.max(board_rank)
	return [int(x == max_score) for x in board_rank]

def find_all_moves_idx(board_rank):
	max_score = np.max(board_rank)
	r = []
	for i in range(len(board_rank)):
		if board_rank[i] == max_score:
			r.append(i)
	return r

def get_data(filename, use_multilabel=False, use_weak=True):
	X = []
	y = []
	with open(filename, 'r') as f:
		for line in f:
			line_data = line.split(',')
			
			# add the board to the data
			board = bc.translate_game(line_data[0])
			X.append(board.flatten())
			board_ranks = undo_what_we_did_before(line_data[0], [ int(x) for x in line_data[1:] ])
			y.append(board_ranks)
			# y.append([0 if i != max_rank else 1 for i in range(7)])
	return X, y

def translate_data_to_multilabel(y):
	'''
	Translates y data of the form:
	-3 4 -3 -3 -3 2 0 (scores)
	into y data of the form:
	1 where the data is 1 for a forced win,
	0 for a draw, and -1 for a forced loss
	'''
	for ranks in y:
		max_ranks = find_all_moves_idx(board_ranks)
		y.append(max_ranks)
	mlb = MultiLabelBinarizer()
	y2 = mlb.fit_transform(y)

def translate_data_to_whowins(y):
	'''
	Translates y data of the form:
	-3 4 -3 -3 -3 2 0 (scores)
	into y data of the form:
	[0 1 0 0 0 1 0] where the data is 1 for
	a move that acheives the best possible outcome,
	and a 1 otherwise
	'''
	y2 = []
	for ranks in y:
		winner = who_wins(board_ranks)
		y2.append(winner)


def mlp_ai(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	print("Small subset of data")
	print(X[:5])
	print(y[:5])
	param={'max_iter' : 250, 'hidden_layer_sizes': (100, 50)}
	mlp = MLPClassifier(**param)
	mlp.fit(X_train, y_train)
	print("Training set score: %f" % mlp.score(X_train, y_train))
	print("Test set score: %f" % mlp.score(X_test, y_test))
	
	return mlp


def main():
	X,y = get_data('bulk_one-ALL.csv')
	y_whowins = translate_data_to_whowins(y)
	y_multilabel = translate_data_to_multilabel(y)
	print('starting mlp whowins!')
	mlp = mlp_ai(X, y_whowins)
	with open('mlp-trained-bulk-whowins.mlp', 'wb') as f:
		pickle.dump(mlp, f)

	print('starting mlp multilabel!')
	mlp = mlp_ai(X, y_whowins)
	with open('mlp-trained-bulk-multilabel.mlp', 'wb') as f:
		pickle.dump(mlp, f)



if __name__ == '__main__':
	main()