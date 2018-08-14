import numpy as np

class Loss:

    def loss(self, ys_predicted, ys):
        pass

    def delta_last(self, y_predicted, y):
        pass
    

class MeanSquaredError(Loss):

    def loss(self, ys_predicted, ys):
        L = 0
        for y_p, y in zip(ys_predicted, ys):
            L += np.dot(y_p - y, y_p - y)
        return 0.5 / len(ys) * L

    def delta_last(self, y_predicted, y):
        delta = np.zeros_like(y)
        delta += y_predicted - y
        return delta


class CrossEntropy(Loss):

    def loss(self, ys_predicted, ys):
        L = 0
        for y_p, y in zip(ys_predicted, ys):
            idx = np.argmax(y)
            L += np.log(y_p[idx])
        return -1.0 / len(ys) * L

    def delta_last(self, y_predicted, y):
        delta = np.zeros_like(y)
        delta += y / y_predicted
        return -delta


class Huber(Loss):

    def __init__(self, delta=1.0):
        self.delta = delta

    def loss(self, ys_predicted, ys):
        L = 0.0
        for y_p, y in zip(ys_predicted, ys):
            diff = np.abs(y - y_p)
            mask = np.greater_equal(self.delta, diff)
            _mask = np.greater(diff, self.delta)
            sqr = 0.5 * np.square(y - y_p)
            lin = self.delta * diff - 0.5 * self.delta**2
            L += np.sum(sqr * mask + lin * _mask)
        return L / len(ys) 

    def delta_last(self, y_predicted, y):
        d = np.zeros_like(y)
        # solution to avoid if-else
        abs_diff = np.abs(y - y_predicted)
        mask = np.greater_equal(self.delta, abs_diff) # True: 1, False: 0
        _mask = np.greater(abs_diff, self.delta)
        diff = y - y_predicted
        c = np.sign(y_predicted - y) * self.delta
        d += diff * mask + c * _mask
        return d 

