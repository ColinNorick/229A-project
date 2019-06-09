import sklearn
import board_conversions as bc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
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

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#param={'max_iter' : 250, 'hidden_layer_sizes': (100, 50)}
	mlp = MLPClassifier(**param)
	mlp.fit(X_train, y_train)
	print("Training set score: %f" % mlp.score(X_train, y_train))
	print("Test set score: %f" % mlp.score(X_test, y_test))
	
	return mlp

def make_graphs(X, y):
	params1 = [
	  {},
	  {'hidden_layer_sizes': (100,100,100),'learning_rate_init': .001},
	  {'hidden_layer_sizes': (100,100,100,100),'learning_rate_init': .001},
	  {'hidden_layer_sizes': (100,100,100,100,100),'learning_rate_init': .001},
	  {'hidden_layer_sizes': (50,50,50,50,50,50,50),'learning_rate_init': .001},
	]

	params2 = [
	{'hidden_layer_sizes': (100,50),'learning_rate_init': .001, 'early_stopping': True},
	  {'hidden_layer_sizes': (100,100),'learning_rate_init': .001, 'early_stopping': True},
	  {'hidden_layer_sizes': (100,150),'learning_rate_init': .001, 'early_stopping': True},
	  {'hidden_layer_sizes': (100,200),'learning_rate_init': .001, 'early_stopping': True},
	  {'hidden_layer_sizes': (50,100),'learning_rate_init': .001, 'early_stopping': True},
	  {'hidden_layer_sizes': (150,100),'learning_rate_init': .001, 'early_stopping': True},
	  {'hidden_layer_sizes': (200, 100),'learning_rate_init': .001, 'early_stopping': True},
	]

	labels = ["Default","100,100,100","100,100,100,100","100,100,100,100,100","50,50,50,50,50,50,50",
	  ]

	plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'},
             {'c': 'orange', 'linestyle': '-'}]

	params_list = [params1, params2]

	#fig, axes = plt.subplots(2, 2, figsize=(15, 10))
	plt.figure()
	#for ax, name, params in zip(axes.ravel(), ['Relu Layers', 'Default Layers'], params_list):	
	plt.title("Neural Network Loss Curves")
	#X = MinMaxScaler().fit_transform(X)
	mlps = []
	for label, param in zip(labels, params1):
		print("training: %s" % label)
		mlp = mlp_ai(X, y, param)	  
		mlps.append(mlp)
	
	for mlp, label, args in zip(mlps, labels, plot_args):
		plt.plot(mlp.loss_curve_, label=label, **args)

		# if params == params1:
		# 	ax = axes.ravel()[2]
		# 	ax.set_title("Relu_Layers_validation loss")
		# 	for mlp, label, args in zip(mlps, labels, plot_args):
		# 		ax.plot(mlp.validation_scores_, label=label, **args)
		# elif params == params2:
		# 	ax = axes.ravel()[3]
		# 	ax.set_title("Default_Layers_validation loss")
		# 	for mlp, label, args in zip(mlps, labels, plot_args):
		# 		ax.plot(mlp.validation_scores_, label=label, **args)

	
	mlp_list = open('mlp_list', 'ab') 
    # source, destination 
	pickle.dump(mlps, mlp_list)                      
	mlp_list.close()	

	plt.legend(labels, ncol=3, loc="upper center")
	plt.xlabel("Iterations")
	plt.ylabel("Loss")
	plt.show()

# from skikit-learn
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
 
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.001,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.001, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot():
	labels = ["Default","100,50","100,100","100,150","100,200","50,100","150,100","200,100",
	  ]

	plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'},
             {'c': 'orange', 'linestyle': '-'}]
	plt.figure()
	plt.title("Neural Network Loss Curves")

	data = open('mlplist', 'rb')

	# dump information to that file
	mlp_list = pickle.load(data)

	# close the file
	mlp_list.close()
	for mlp, label, args in zip(mlps, labels, plot_args):
		plt.plot(mlp.loss_curve_, label=label, **args)


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
	#title = "test"
	#estimator = MLPClassifier(max_iter = 1800, alpha = .000001)
	#plot_learning_curve(estimator, title, X, y_whowins, cv = 5, n_jobs=-1)
	plt.show()


if __name__ == '__main__':
	main()