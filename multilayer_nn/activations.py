import numpy as np


class Activation:
    '''
    Base class for activations.
    '''
    def activate(self, x):
        '''
        The original activation function.
        '''
        pass

    def d_activate(self, x):
        '''
        The derivated function.
        x - input tensor
        '''
        pass


class Linear(Activation):

    def __init__(self, a=1.0, b=0.0):
        self.a = a
        self.b = b
    
    def activate(self, x):
        return self.a * x + self.b
    
    def d_activate(self, x):
        return np.diag(np.ones_like(x) * self.a)


class Relu(Activation):

    def activate(self, x):
        temp = np.zeros((x.shape[0], 2))
        temp[:, 0] = x
        return np.max(temp, axis=-1)
    
    def d_activate(self, x):
        zeros = np.zeros_like(x)
        bools = np.greater_equal(x, zeros)
        dv = bools.astype(np.float32)
        return np.diag(dv)


class Tanh(Activation):

    def activate(self, x):
        return np.tanh(x)
    
    def d_activate(self, x):
        chx = np.cosh(x)
        return np.diag(1.0 / np.square(chx))


class Sigmoid(Activation):

    def activate(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def d_activate(self, x):
        return np.diag(self.activate(x) * (1.0 - self.activate(x)))


class Softmax(Activation):

    def activate(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def d_activate(self, x):
        s = self.activate(x)
        return np.diag(s) - np.outer(s, s)


class Elu(Activation):

    def activate(self, x):
        mask = np.greater_equal(0, x)
        return (np.exp(x) - 1) * mask + x * (1 - mask)
    
    def d_activate(self, x):
        mask = np.greater_equal(0, x)
        dv = (np.exp(x)) * mask + 1 * (1 - mask)
        return np.diag(dv)


class SoftPlus(Activation):

    def activate(self, x):
        return np.log(np.exp(x) + 1)
    
    def d_activate(self, x):
        dv = np.exp(x) / (1 + np.exp(x))
        return np.diag(dv)
