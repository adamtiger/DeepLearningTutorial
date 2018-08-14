import numpy as np
from mutlilayer_nn import activations, losses, optimizers


class Mlp:
    
    def __init__(self, optimizer, loss, initializer, regularizer):
        self.optimizer = optimizer
        self.loss = loss
        self.initializer = initializer
        self.regularizer = regularizer 

        self.theta = []
        self.activations = []
        self.gradients = []

        self.layers = 0 # number of layers
    
    def add_layer(self, nodes, input_length=0, activation=activations.Linear()):
        if input_length==0:
            if len(self.theta) > 0:
                input_length = self.theta[-1].shape[0]
            else:
                assert "Missing input_length at first layer!"

        w = self.initializer.create((input_length, nodes))
        self.theta.append(w)
        self.activations.append(activation)
        self.layers += 1
    
    def __forward(self, x):
        '''
        x - an input vector, only one sample
        '''
        w_times_xs = []
        hs = []
        x_current = x
        for w, a in zip(self.theta, self.activations):
            w_times_x = np.matmul(w, x_current)
            h = a.activate(w_times_x)
            w_times_xs.append(w_times_x)
            hs.append(h)
            x_current = h
        return x_current, w_times_xs, hs

    def __backward(self, w_times_xs, hs, y_real):
        '''
        w_times_xs - w product x for all hidden layers (list)
        hs - outputs of each layer (list)
        y_real - a real output from the training set (one sample)
        '''
        y_predicted = hs[-1]
        delta = self.loss.delta_last(y_predicted, y_real)
        for l in range(self.layers-1, -1, -1): # walking the list backward direction
            wx = w_times_xs[l]
            h = hs[l]
            df = self.activations[l].d_activate
            df_wx = np.matmul(df(wx), delta)
            self.gradients[l] += np.outer(df_wx, h) # calculate gradient for delta
            delta = np.matmul(df_wx, self.theta[l]) # calculate new delta

    def __init_gradients(self):
        if len(self.gradients) == 0:
            for w in self.theta:
                self.gradients.append(np.zeros_like(w))
        else:
            for idx in range(self.layers):
                self.gradients[idx] *= 0.0
    
    def __multiply_gradient_list(self, factor):
        self.gradients = list(map(lambda x: factor * x, self.gradients))


    