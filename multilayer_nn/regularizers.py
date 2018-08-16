import numpy as np


class Regularizer:

    def __init__(self, beta):
        self.beta = beta

    def reg_term(self, ys_predicted, ys):
        pass

    def delta_last(self, y_predicted, y):
        pass


class ZeroRegularizer(Regularizer):

    def __init__(self):
        super().__init__(0.0)

    def reg_term(self, ys_predicted, ys):
        return 0

    def delta_last(self, y_predicted, y):
        return 0