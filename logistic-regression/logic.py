import numpy as np

# This is an implementation of logistic regression.

def sigmoid(theta, x):
    z = np.dot(theta, x)
    return 1.0 / (1.0 + np.exp(-z))


def loss(theta, xs, ys):
    '''
    xs - the input values in a list, each value is a numpy array
    ys - the correct label for each input in a list
    '''
    J = 0.0
    for x, y in zip(xs, ys):
        h_theta = sigmoid(theta, x)
        J += y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta)
    return J


def grad(theta, xs, ys):
    '''
    xs - the input values in a list, each value is a numpy array
    ys - the correct label for each input in a list
    '''
    g = np.zeros_like(xs[0])
    for x, y in zip(xs, ys):
        h_theta = sigmoid(theta, x)
        g += (y - h_theta) * x
    return g


def update(theta, lr, gradient):
    return theta + lr * gradient


def init(theta_size):
    return np.random.uniform(theta_size)


def logistic_regression(data, lr, max_iter, verbose=False):
    pass

