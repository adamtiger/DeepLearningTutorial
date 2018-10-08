import numpy as np


class Optimizer:
     
    def __init__(self, lr):
        self.lr = lr

    def optimizer_step(self, theta, gradients):
        pass


class SGD(Optimizer):

    def __init__(self, lr):
        '''
        The vanilla SGD (Mini-batch gradient descent).
        lr - learning rate
        '''
        super().__init__(lr)
    
    def optimizer_step(self, theta, gradients):

        for idx, (w, grad) in enumerate(zip(theta, gradients)):
            theta[idx] = w - self.lr * grad


class Momentum(Optimizer):

    def __init__(self, gamma, lr):
        '''
        Momentum method uses a momentum which stores the previous gradients.
        It makes the algorithm a bit more robust.
        '''
        super().__init__(lr)
        self.gamma = gamma
        self.v_prev = [0]

    def optimizer_step(self, theta, gradients):
        if len(self.v_prev) == 1:
            self.v_prev = [0] * len(theta)
        for idx in range(len(theta)):
            self.v_prev[idx] = self.gamma * self.v_prev[idx] + self.lr * gradients[idx]
            theta[idx] = theta[idx] - self.v_prev[idx]


class Adam(Optimizer):
    '''
    Article: https://arxiv.org/pdf/1412.6980.pdf
    '''
    def __init__(self, beta1, beta2, lr):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.beta1_t = beta1
        self.beta2_t = beta2
    
    def optimizer_step(self, theta, gradients):
        if self.m is None and self.v is None:
            self.m = [np.zeros(gradients[0].shape)] * len(theta)
            self.v = [np.zeros(gradients[0].shape)] * len(theta)
        for idx, g in enumerate(gradients):
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g
            m_hat = self.m[idx] / (1 - self.beta1_t)
            v_hat = self.v[idx] / (1 - self.beta2_t)
            theta[idx] = theta[idx] - self.lr * 1.0 / np.sqrt(v_hat + 1e-8) * m_hat
            self.beta1_t = self.beta1_t * self.beta1
            self.beta2_t = self.beta2_t * self.beta2


class Adadelta(Optimizer):
    '''
    Article: https://arxiv.org/pdf/1212.5701.pdf
    '''
    def __init__(self, gamma):
        '''
        Here the lr is not used.
        '''
        super().__init__(1.0)
        self.gamma = gamma
        self.G = None         # Accumulated gradients
        self.delta_Th = None  # The delta theta squares
    
    def optimizer_step(self, theta, gradients):

        def rms(x):
            return np.sqrt(x * x) + 1e-10

        if self.G is None and self.delta_Th is None:
            self.G = [np.zeros(gradients[0].shape)] * len(theta)
            self.delta_Th = [np.zeros(gradients[0].shape)] * len(theta)
        for idx, g in enumerate(gradients):
            self.G[idx] += self.gamma * self.G[idx] + (1 - self.gamma) * g * g
            delta = -rms(self.delta_Th[idx]) / rms(self.G[idx]) * g
            self.delta_Th[idx] = self.gamma * self.delta_Th[idx] + self.gamma * delta * delta
            theta[idx] = theta[idx] + delta
        

class RMSProp(Optimizer):
    '''
    Reference: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    '''
    def __init__(self, gamma, lr):
        super().__init__(lr)
        self.gamma = gamma
        self.G = None

    def optimizer_step(self, theta, gradients):
        if self.G is None:
           self.G = [np.zeros(gradients[0].shape)] * len(theta)
        for idx, g in enumerate(gradients):
            g = g / np.linalg.norm(g)
            self.G[idx] += self.gamma * self.G[idx] + (1 - self.gamma) * g * g
            theta[idx] = theta[idx] - self.lr / np.sqrt(self.G[idx]) * g 


class Adagrad(Optimizer):
    
    def __init__(self, lr):
        super().__init__(lr)
        self.G = None
    
    def optimizer_step(self, theta, gradients):
        if self.G is None:
            self.G = [np.zeros(gradients[0].shape)] * len(theta)
        for idx, g in enumerate(gradients):
            self.G[idx] += g * g
            theta[idx] = theta[idx] - self.lr * 1.0 / np.sqrt(self.G[idx])
