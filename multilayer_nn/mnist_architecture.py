from multilayer_nn import mlp, activations, losses, regularizers, initializers, optimizers
from matplotlib import pyplot as plt
from pckutils import mnist, utils
import seaborn as sns
import pandas as pd
import numpy as np
import os


def load_data():
    # loading MNIST data
    data = mnist.load_mnist('data')

    # add bias to x and one-hot encode y
    X = utils.add_bias_to_vectors(utils.scale(data.X_train, 1/255.0))
    Y = utils.one_hot_encode(data.Y_train, 10)
    return X, Y


class Architecture:

    def __init__(self, regularizer, X, Y):
        self.regularizer = regularizer
        self.X = X
        self.Y = Y

        # other elements of the model
        self.loss = losses.CrossEntropy()
        self.initializer = initializers.Xavier()
        self.optimizer = optimizers.SGD(0.1)

        self._build_model()
        self._fit()
    
    def _build_model(self):
        self.nn = mlp.Mlp(self.optimizer, self.loss, self.initializer, self.regularizer)

        self.nn.add_layer(30, input_length=28*28+1, activation=activations.Relu())
        self.nn.add_layer(20, activation=activations.Tanh())
        self.nn.add_layer(10, activation=activations.Softmax())
    
    def _fit(self):
        # callback for gathering the error rates during training
        self.history = {'errors': [], 'losses': []}
        def performance_monitor(batch_y_p, batch_y, loss, ep, itr):
            if itr % 2 == 0:
                self.history['losses'].append(loss)
                err = utils.error_rate(batch_y_p, batch_y)
                self.history['errors'].append(err)
        
        self.nn.fit(self.X, self.Y, 10, 6000, verbose=True, callback=performance_monitor)
    
    def show_learning_curve(self):
        iterations = [i for i in range(len(self.history['errors']))]
        plt.plot(iterations, self.history['errors'], 'ro')
        plt.show()

    def show_activations(self):
        activations = []
        def sniffer(x):
            activations.append(x)
            return x
        
        self.nn.sniffer = sniffer

        self.nn.predict_batch(self.X, 'raw')

        average_activations = [0, 0]  # Only the two layers at the middle are important
        for activation in activations:
            average_activations[0] += activation[1]  # second hidden layer
            average_activations[1] += activation[2]  # third layer
        
        average_activations[0] = average_activations[0] / np.max(np.abs(average_activations[0]))
        average_activations[1] = average_activations[1] / np.max(np.abs(average_activations[1]))
        
        data = {'layer': [], 'y': [], 'activation': []}
        data['layer'] = [1] * 30 + [2] * 20
        data['y'] = [0.2 * i for i in range(30)] + [1.0 + 0.2 * i for i in range(20)]
        data['activation'] = average_activations[0].tolist() + average_activations[1].tolist()
        df = pd.DataFrame(data)
        
        sns.set()
        cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
        ax = sns.scatterplot(x="layer", y="y", size="activation",
                     palette=cmap, sizes=(10, 200),
                     data=df)





    

