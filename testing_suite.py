# Everything is run from this file
import numpy as np
from graphics import *
import data_process as dp
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

#a = np.ones((6,7))
#print (a)   
#gh.draw_game(a)
# test = "5554224333234511764415115"
# game = translate_game(test)
# draw_game(game)



#processing the data
X,y = dp.process()


params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.4},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .0,
          'learning_rate_init': .0001},
        ]

labels = ["high learning rate", "low learning rate ",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]


def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    #X = MinMaxScaler().fit_transform(X)
    mlps = []
    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=1,
                            max_iter=200, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)

#Where things start


fig, axes = plt.subplots(2, 2, figsize=(15, 10))



for ax, name in zip(axes.ravel(), ['High Learning Rate', 'Low Learning Rate',
                                                    'momentum +', 'momentum -']):
    plot_on_dataset(X, y, ax=ax, name=name)

fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
plt.show()
# mlp = MLPClassifier(verbose=0, random_state=0)
# mlp.fit(X, y)   
# print("Training set score: %f" % mlp.score(X, y))
# print("Training set loss: %f" % mlp.loss_)

