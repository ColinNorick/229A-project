import NN
import numpy as np

'''
Takes a string representation of a connect-4 situation from the database and reshapes it.
'''
def StrToBoard(string):
    ret = [[],[],[],[],[],[]]
    for i in range(7):
        for j in range(6):
            z = 5-j+6*i
            ret[j].append(" " if string[z]=="b" else string[z])
    return [ret, string[42]]

'''
Shows the string representation of a connect-4 situation from the database.
'''
def displayStr(string):
    display_board(StrToBoard(string)[0])


'''
Make string representation of a connect-4 situation from the database usable for the neural network.
'''
def list4NN(string):
	# unlisted
	# A= [[1,2],[3,4]] -> A= [1,2,3,4]
	# sum(StrToBoard(string)[0])
    ret = []
    for i in range(7):
        for j in range(6):
            z = 5-j+6*i
            ret.append(0 if string[z]=="b" else (-1 if string[z]=="o" else 1))
    target = 0 if string[42]=="b" else (-1 if string[42]=="o" else 1)
    return [ret, target]

'''
Make a game situation usable for the neural network.
'''
def board2NN(board):
    tmp = np.array(board)
    tmp = tmp.T
    tmp = tmp.tolist()
    tmp = sum(tmp,[])
    tmp = [1 if x=="O" else (-1 if x=="X" else 0) for x in tmp]
    return tmp

'''
Convert neural network data to game situation.
'''
def NN2board(li):
    tmp = ["X" if x==1 else ("O" if x==-1 else " ") for x in li]
    tmp = np.array(tmp)
    tmp.shape = (7,6)
    tmp = tmp.T
    return tmp

################################################################

try:
    nn = NN.openNN()
    print "NN successfully opened"

except:
    print "not able to open NN"
    print "create new one"

    inp = []
    out = []

    fobj = open("database.txt")
    counter=0

    for line in fobj:
        #print line.rstrip()
        #displayStr(line.rstrip())
        tmp = list4NN(line.rstrip())
        
        inp.append(tmp[0])
        out.append([tmp[1]])
        # counter for stopping if its too long
        #counter = counter+1
        if counter > 100:
            break
    fobj.close()
    print "Got the Data!"
    nn = NN.NeuralNetwork(inp,out,hidden_nodes=[40])
    print "created a Network"
    nn.train()
    print "network trained"





