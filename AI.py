import sklearn
import board_conversions as bc
import numpy as np
import matplotlib.pyplot as plt
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
	[0 1 0 0 0 1 0] where the data is 1 for
	a move that acheives the best possible outcome,
	and a 1 otherwise
	
	'''
	for rank in y:
		max_ranks = find_all_moves_idx(rank)
		y.append(max_ranks)	
	mlb = MultiLabelBinarizer()
	y2 = mlb.fit_transform(y)

def translate_data_to_whowins(y):
	'''
	Translates y data of the form:
	-3 4 -3 -3 -3 2 0 (scores)
	into y data of the form:
	1 where the data is 1 for a forced win,
	0 for a draw, and -1 for a forced loss
	'''
	y2 = []
	for rank in y:
		winner = who_wins(rank)
		y2.append(winner)

	return y2

def mlp_ai(X, y, param):
	print("Small subset of data")
	print(X[:5])
	print(y[:5])
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#param={'max_iter' : 250, 'hidden_layer_sizes': (100, 50)}
	mlp = MLPClassifier(**param)
	mlp.fit(X_train, y_train)
	print("Training set score: %f" % mlp.score(X_train, y_train))
	print("Test set score: %f" % mlp.score(X_test, y_test))

	return mlp

def make_graphs(X, y):
	params = [
	{},
	  {'hidden_layer_sizes': (100,),'learning_rate_init': .0001, 'activation' :'logistic'},
	  #{'hidden_layer_sizes': (100, 50),'learning_rate_init': .0001, 'activation' :'logistic'},
	  #{'hidden_layer_sizes': (100, 100),'learning_rate_init': .0001, 'activation' :'logistic'},
	  #{'hidden_layer_sizes': (100, 200),'learning_rate_init': .0001, 'activation' :'logistic'},

	  {'hidden_layer_sizes': (100,),'learning_rate_init': .0001, 'activation' :'tanh'},
	  #{'hidden_layer_sizes': (100,),'learning_rate_init': .0001, 'activation' :'tanh'},
	  #{'hidden_layer_sizes': (100,),'learning_rate_init': .0001, 'activation' :'tanh'},
	  #{'hidden_layer_sizes': (100,),'learning_rate_init': .0001, 'activation' :'tanh'},

	  {'hidden_layer_sizes': (100,),'learning_rate_init': .0001, 'activation' :'relu'},

	]

	labels = ["default", "logistic", "tanh","relu",
	  ]

	plot_args = [{'c': 'red', 'linestyle': '-'},
	 {'c': 'green', 'linestyle': '-'},
	 {'c': 'black', 'linestyle': '-'},
	 {'c': 'orange', 'linestyle': '-'},
	 ]

	# for each dataset, plot learning for each learning strategy
	fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	#for ax, name in zip(axes.ravel(), ['Loss of different Archectectures', 'Low Learning Rate']):
	ax = axes.ravel()[0]	
	ax.set_title("Loss of different Archectectures")
	#X = MinMaxScaler().fit_transform(X)
	mlps = []
	for label, param in zip(labels, params):
		print("training: %s" % label)
		mlp = mlp_ai(X, y, param)	  
		mlps.append(mlp)
	
	for mlp, label, args in zip(mlps, labels, plot_args):
		ax.plot(mlp.loss_curve_, label=label, **args)

	fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
	plt.show()


def main():
	print('getting data!')
	X,y = get_data('bulk_one-ALL.csv')
	print('traslating data!')
	y_whowins = translate_data_to_whowins(y)
	# print(y_whowins)
	print('Doing the actual work now!')
	make_graphs(X, y_whowins)
	#y_multilabel = translate_data_to_multilabel(y)
	#print('starting mlp whowins!')
	# mlp = mlp_ai(X, y_whowins)
	# with open('mlp-trained-bulk-whowins.mlp', 'wb') as f:
	# 	pickle.dump(mlp, f)

	# print('starting mlp multilabel!')
	# mlp = mlp_ai(X, y_whowins)
	# with open('mlp-trained-bulk-multilabel.mlp', 'wb') as f:
	# 	pickle.dump(mlp, f)



if __name__ == '__main__':
	main()