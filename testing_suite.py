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


mlp = MLPClassifier(verbose=0, random_state=0)
mlp.fit(X, y)
print("Training set score: %f" % mlp.score(X, y))
print("Training set loss: %f" % mlp.loss_)

