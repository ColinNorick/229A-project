import numpy as ny
from scipy.special import expit
import cPickle as pickle

class NeuralNetwork:

    def __init__(self, inputs, targets, hidden_nodes = [2], learning_rate=0.01, momentum = 0.9, activation_type = 1):
        """
        Constructor.
        inputs       -- Array with training data.
        targets      -- Array with targets to the training data.
        hidden_nodes -- Number of nodes in the hidden layer.
        learning_rate-- Learning spee.
        momentum     -- 
        activation_function -- 1 for tanh, 2 for sigmoid
        """

        self.inputs = ny.array( inputs )
        self.targets = ny.array( targets )

        self.nodes_in = len( inputs[0] )
        self.nodes_out = len( targets[0] )
        self.data_amount = len( inputs )

        self.nodes_hidden = hidden_nodes
        self.learning_rate = learning_rate
        self.momentum = momentum

        # range tanh from -1 to 1, range sigmoid from 0 to 1
        self.activation_type = activation_type
        if activation_type == 1:
            self.activation_function = ny.tanh
            #self.derivative_function = lambda x : 1-ny.square(self.activation_function(x))
        else:
            self.activation_function = expit
            #self.derivative_function = lambda (x): self.activation_function(x)*(1-self.activation_function(x))

        self.number_hidden_layers = len(hidden_nodes)

        self._init_weights()

    def _init_weights(self):
        """
        Randomly initialize weights.
        There are to weight layers: One from the input nodes
        to the hidden nodes, and one from the hidden nodes to
        the output nodes.
        """

        self.weights_layer = [ny.random.rand( self.nodes_hidden[0], self.nodes_in )]
        self.bias = []

        for i in range(self.number_hidden_layers-1):
            self.weights_layer.append(ny.random.rand( self.nodes_hidden[i+1], self.nodes_hidden[i] ))
            self.bias.append( ny.ones( (self.nodes_hidden[i],1) ) )

        self.bias.append( ny.ones((self.nodes_hidden[self.number_hidden_layers-1],1)))
        self.bias.append( ny.ones( (self.nodes_out, 1) ) )
        self.weights_layer.append(ny.random.rand( self.nodes_out, self.nodes_hidden[self.number_hidden_layers-1] ))


    def calc_activation(self, inp):
        """
        Calculate a output for an input.
        Save calculated activation values for each layer in List a.
        """
        inp_rightform = ny.matrix( inp ).T
        self.a = [inp_rightform]
        tmp = ny.dot( self.weights_layer[0], inp_rightform ) + self.bias[0]
        tmp = self.activation_function(tmp)

        self.a.append(tmp)

        for i in range(self.number_hidden_layers-1):

            tmp = ny.dot( self.weights_layer[i+1], tmp ) + self.bias[i+1]
            tmp = self.activation_function(tmp)
            self.a.append(tmp)

        tmp = ny.dot( self.weights_layer[self.number_hidden_layers], tmp )+self.bias[self.number_hidden_layers]
        tmp = self.activation_function(tmp)

        self.a.append(tmp)
        #eventuell muss shape von tmp angepasst werden

    def activate(self, inp):
        """
        Calculate output for an input.
        """
        inp_rightform = ny.matrix( inp ).T
        tmp = ny.dot( self.weights_layer[0], inp_rightform ) + self.bias[0]
        tmp = self.activation_function(tmp)

        for i in range(self.number_hidden_layers):
            tmp = self.activation_function(ny.dot( self.weights_layer[i+1], tmp ) + self.bias[i+1])

        #tmp = self.activation_function(ny.dot( self.weights_layer[self.number_hidden_layers], tmp )+self.bias[self.number_hidden_layers])
        #eventuell muss shape von tmp angepasst werden
        return tmp

    def train(self, alpha=0.5):
        """
        Does backpropagation on every single Dataset.
        """
        for i in range(self.data_amount):
            self.backprop(self.inputs[i],self.targets[i],alpha)


    def backprop(self, inp, target, alpha):
        """
        Backpropagation for one specific input and target.
        First calculate activation values for each layer, then calculate the deltas backwards.
        In the end calculate the error and correct the weights and biases.

        alpha = learning_rate
        """

        # set derivative function
        if self.activation_type==1:
            derivative = lambda a: 1-ny.square(a)
        else:
            derivative = lambda a: a*(1.0-a)

        self.calc_activation(inp)

        # correct target's format
        target_rightform = ny.matrix( target ).T

        # calculate deltas
        tmp = self.a[-1] - target_rightform

        tmp = ny.multiply(tmp, derivative(self.a[-1]))

        self.delta = [tmp]

        for i in range(self.number_hidden_layers):
            tmp = (ny.dot(self.weights_layer[-1-i].T, self.delta[i]))
            tmp = ny.multiply(tmp, derivative(self.a[-1-1-i]))

            self.delta.append(tmp)

        # correct weights and biases with the deltas
        for i in range(len(self.weights_layer)):
            self.weights_layer[i] -= alpha*(ny.dot(self.delta[-1-i], self.a[i].T))

        for i in range(len(self.bias)):
            self.bias[i] -= alpha * self.delta[-1-i]


    def full_batch_backprop(self, alpha, lamb):
        """
        Backpropagation with batch_size = data_amount

        alpha = learning rate, lambda = weight_decay
        """
        if self.activation_type==1:
            derivative = lambda a: 1-ny.square(a)
        else:
            derivative = lambda a: a*(1.0-a)


        delta_W = []
        delta_b = []
        for i in range(self.number_hidden_layers+1):
            delta_W.append(0)
            delta_b.append(0)

        for i in range(self.data_amount):

            inp = self.inputs[i]
            target = self.targets[i]

            self.calc_activation(inp)

            target_rightform = ny.matrix( target ).T
            tmp = self.a[-1] - target_rightform

            tmp = ny.multiply(tmp, derivative(self.a[-1]))

            self.delta = [tmp]

            for i in range(self.number_hidden_layers):
                tmp = (ny.dot(self.weights_layer[-1-i].T, self.delta[i]))
                tmp = ny.multiply(tmp, derivative(self.a[-1-1-i]))

                self.delta.append(tmp)

            for i in range(len(self.weights_layer)):
                delta_W[i] += (ny.dot(self.delta[-1-i], self.a[i].T))
                delta_b[i] += self.delta[-1-i]
                #self.weights_layer[i] -= alpha*(ny.dot(self.delta[-1-i], self.a[i].T))

            #for i in range(len(self.bias)):
                #delta_b[i] += self.delta[-1-i]
                #self.bias[i] -= alpha * self.delta[-1-i]

        for i in range(len(self.weights_layer)):
            self.weights_layer[i] -= alpha*(delta_W[i]/self.data_amount + lamb*self.weights_layer[i])
            self.bias[i] -= alpha * delta_b[i] / self.data_amount


    def batch_backprop(self, alpha, lamb, batch_size):
        """
        Batch-Backpropagation with flexible batch_size

        alpha = learning rate, lambda = weight_decay
        """
        # init derivated function
        if self.activation_type==1:
            derivative = lambda a: 1-ny.square(a)
        else:
            derivative = lambda a: a*(1.0-a)

        # init deltas
        delta_W = []
        delta_b = []

        z = self.data_amount if batch_size > self.data_amount else self.data_amount / batch_size
        for k in range(z):

            for i in range(self.number_hidden_layers+1):
                delta_W.append(0)
                delta_b.append(0)

            for j in range(batch_size):

                i = j + batch_size*k

                inp = self.inputs[i]
                target = self.targets[i]

                self.calc_activation(inp)

                target_rightform = ny.matrix( target ).T
                tmp = self.a[-1] - target_rightform

                tmp = ny.multiply(tmp, derivative(self.a[-1]))

                self.delta = [tmp]

                for i in range(self.number_hidden_layers):
                    tmp = (ny.dot(self.weights_layer[-1-i].T, self.delta[i]))
                    tmp = ny.multiply(tmp, derivative(self.a[-1-1-i]))

                    self.delta.append(tmp)

                for i in range(len(self.weights_layer)):
                    delta_W[i] += (ny.dot(self.delta[-1-i], self.a[i].T))
                    delta_b[i] += self.delta[-1-i]
                    

            for i in range(len(self.weights_layer)):
                self.weights_layer[i] -= alpha*(delta_W[i]/self.data_amount + lamb*self.weights_layer[i])
                self.bias[i] -= alpha * delta_b[i] / self.data_amount


    def test_acc(self, part=0.1):
        '''
        Test the networks accurracy on its own dataset.
        Not the whole dataset but the part of the dataset.
        '''
        nr = self.data_amount*part
        err = 0
        for i in range(int(round(nr))):
            err += self.targets[i] - self.activate(self.inputs[i])
        return err/nr

    
    def save(self, filename = "savedNetwork.obj", with_data=False):
        '''
        Save the network in filename with pickle.
        TODO
        '''
        fobj = open(filename, 'w')
        pickle.dump(self,fobj)

        '''
        fobj.write("in ")
        fobj.write(str(self.nodes_in)+"\n")
        
        fobj.write("out ")
        fobj.write(str(self.nodes_out)+"\n")
        
        fobj.write("nrhidden ")
        fobj.write(str(self.number_hidden_layers) + "\n")
        
        fobj.write("hidden ")
        fobj.write(str(self.nodes_hidden)+"\n")
        
        fobj.write("lr ")
        fobj.write(str(self.learning_rate)+"\n")
        
        fobj.write("mom ")
        fobj.write(str(self.momentum+"\n")
        
        fobj.write("actype ")
        fobj.write(str(self.activation_type+"\n")

        if with_data:
            fobj.write("data\n")
            pass
        else:
            fobj.write("nodata\n")
            pass
            '''

def openNN(filename = "savedNetwork.obj", with_data=False):
    '''
    Open the network from filename with pickle.
    '''
    fobj = open(filename, 'r')
    return pickle.load(fobj)



##############################################################################

'''

inp = [[0,0],[0,1],[1,0],[1,1]]
out = [[0],[1],[1],[0]]
nn = NeuralNetwork(inp,out,hidden_nodes=[3])
nn.activate(ny.array([1,1]))

'''

'''
numpy.array([[ 0],
       [ 1],
       [ 2],
       [ 3]])
'''


