import numpy as np


class Regularizer:
    '''
    Regularization is for preventing the network from overfitting.
    ws - the list of weights, each weight is a numpy matrix
    '''
    def __init__(self, beta):
        self.beta = beta

    def reg_term(self, ws):
        pass

    def delta_last(self, ws):
        pass


class ZeroRegularizer(Regularizer):

    def __init__(self):
        super().__init__(0.0)

    def reg_term(self, ws):
        return 0

    def delta_last(self, ws):
        return 0


class L1(Regularizer):

    def __init__(self, beta):
        super().__init__(beta)
    
    def reg_term(self, ws):
        l1 = 0.0
        for w in ws:
            l1 += np.sum(np.abs(w))
        return self.beta * l1
    
    def delta_last(self, ws):
        d_ws = []
        for w in ws:
            dw = np.greater_equal(w, 0) - np.less(w, 0)
            d_ws.append(dw)
        return d_ws


class L2(Regularizer):

    def __init__(self, beta):
        super().__init__(beta)

    def reg_term(self, ws):
        l2 = 0.0
        for w in ws:
            l2 += np.sum(w * w)
        return 0.5 * self.beta * l2
    
    def delta_last(self, ws):
        return list(map(lambda x: self.beta * x, ws))
