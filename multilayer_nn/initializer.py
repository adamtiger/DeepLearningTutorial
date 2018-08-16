import numpy as np


class Initializer:

    def __init__(self, seed):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def create(self, shape):
        pass


class Zeros(Initializer):

    def __init__(self):
        super().__init__(None)

    def create(self, shape):
        return np.zeros(shape)


class Uniform(Initializer):

    def __init__(self, low=0.0, high=1.0, seed=None):
        super().__init__(seed)
        self.low = low
        self.high = high

    def create(self, shape):
        return np.random.uniform(self.low, self.high, size=shape)


class Normal(Initializer):

    def __init__(self, loc=0.0, scale=1.0, seed=None):
        super().__init__(seed)
        self.loc = loc
        self.scale = scale
    
    def create(self, shape):
        return np.random.normal(self.loc, self.scale, size=shape)


class Xavier(Initializer):

    def __init__(self, seed=None):
        super().__init__(seed)

    def create(self, shape):
        d = np.sqrt(6.0 / (shape[0] + shape[1] + 1.0))
        return np.random.uniform(-d, d, shape)
