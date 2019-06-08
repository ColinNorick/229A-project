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

def get_data(filename, use_multilabel=True):
	X = []
	y = []
	with open(filename, 'r') as f:
		for line in f:
			line_data = line.split(',')
			
			# add the board to the data
			board = bc.translate_game(line_data[0])
			X.append(board.flatten())
			board_ranks = undo_what_we_did_before(line_data[0], [ int(x) for x in line_data[1:] ])
			if not use_multilabel:
				# add the output to the data
				max_rank = np.argmax(board_ranks)
				y.append(max_rank)
			else:
				max_ranks = find_all_moves_idx(board_ranks)
				y.append(max_ranks)
			# y.append([0 if i != max_rank else 1 for i in range(7)])
	if use_multilabel:		
		print('pre-multilabel stuff:')
		print(y[:5])
		mlb = MultiLabelBinarizer()
		y = mlb.fit_transform(y)
		print('post-multilabel stuff:')
		print(y[:5])
	return X, y

def mlp_ai(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	print("Small subset of data")
	print(X[:5])
	print(y[:5])
	param={'max_iter' : 250, 'hidden_layer_sizes': (100, 100)}
	mlp = MLPClassifier(**param)
	mlp.fit(X_train, y_train)
	print("Training set score: %f" % mlp.score(X_train, y_train))
	print("Test set score: %f" % mlp.score(X_test, y_test))
	return mlp


def main():
	X,y = get_data('bulk_one-ALL.csv')
	print('starting mlp!')
	mlp = mlp_ai(X, y)
	with open('mlp-trained-bulk.mlp', 'wb') as f:
		pickle.dump(mlp, f)



if __name__ == '__main__':
	main()