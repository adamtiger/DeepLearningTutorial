import numpy as np

class Loss:

    def loss(self, y_predicted, y):
        pass

    def delta_last(self, y_predicted, y):
        pass
    

class MeanSquaredError(Loss):

    def loss(self, y_predicted, y):
        L = 0
        for y_p, y_ in zip(y_predicted, y):
            L += np.dot(y_p - y_, y_p - y_)
        return 0.5 / len(y) * L

    def delta_last(self, y_predicted, y):
        delta = np.zeros_like(y[0])
        for y_p, y_ in zip(y_predicted, y):
            delta += y_p - y_
        return 1.0 / len(y) * delta


class CrossEntropy(Loss):

    def loss(self, y_predicted, y):
        L = 0
        for y_p, y_ in zip(y_predicted, y):
            L += np.dot(y_p, np.log(y_))
        return -1.0 / len(y) * L

    def delta_last(self, y_predicted, y):
        delta = np.zeros_like(y[0])
        for y_p, y_ in zip(y_predicted, y):
            delta += y_ / y_p
        return -1.0 / len(y) * delta


class Huber(Loss):

    def __init__(self, delta=1.0):
        self.delta = delta

    def loss(self, y_predicted, y):
        L = 0.0
        for y_p, y_ in zip(y_predicted, y):
            diff = np.abs(y_ - y_p)
            mask = np.greater_equal(self.delta, diff)
            _mask = np.greater(diff, self.delta)
            sqr = 0.5 * np.square(y_ - y_p)
            lin = self.delta * diff - 0.5 * self.delta**2
            L += np.sum(sqr * mask + lin * _mask)
        return L / len(y) 

    def delta_last(self, y_predicted, y):
        d = np.zeros_like(y[0])
        for y_p, y_ in zip(y_predicted, y): # solution to avoid if-else
            abs_diff = np.abs(y_ - y_p)
            mask = np.greater_equal(self.delta, abs_diff) # True: 1, False: 0
            _mask = np.greater(abs_diff, self.delta)
            diff = y_ - y_p
            c = np.sign(y_p - y_) * self.delta
            d += diff * mask + c * _mask
        return d / len(y)

