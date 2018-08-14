

class Optimizer:
     
    def __init__(self, lr):
        self.lr = lr

    def optimizer_step(self, theta, gradients):
        pass


class SGD(Optimizer):

    def __init__(self, lr):
        super().__init__(lr)
    
    def optimizer_step(self, theta, gradients):

        for idx, (w, grad) in enumerate(zip(theta, gradients)):
            theta[idx] = w + self.lr * grad
