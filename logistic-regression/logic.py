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


def deduce_batch(xs, ys, size):
    batch_x = []
    batch_y = []

    indices = np.array([t for t in range(len(xs))])
    np.random.shuffle(indices)
    for idx in indices.tolist():
        batch_x.append(xs[idx])
        batch_y.append(ys[idx])

    return batch_x, batch_y


def logistic_regression(data, lr, max_iter, batch_size, epoch=5, split_ratio=0.8, verbose=False):
    '''
    data - a tuple containing two lists, first for the input and second for the labels,
           each element of the input list is a 1-D numpy array
    lr - learning rate
    max_iter - number of iterations during optimization
    batch_size - size of a batch
    epoch - number of repetition for a batch
    split_ratio - the ratio of the training examples from data
    verbose - show progress during run
    '''
    xs, ys = data
    theta_size = xs[0].shape[0] + 1 # +1 due to bias
    theta = init(theta_size)

    # extend xs wih 1s
    for i, x in enumerate(xs):
        temp = np.ones(theta_size)
        temp[1:] = x
        xs[i] = temp
    
    # split the data
    indices = np.array([i for i in range(len(xs))])
    np.random.shuffle(indices)
    break_point = int(len(xs) * split_ratio)

    xs_train = []
    ys_train = []
    for i in range(break_point):
        xs_train.append(xs[indices[i]])
        ys_train.append(ys[indices[i]])

    xs_test = []
    ys_test = []
    for i in range(break_point, len(xs)):
        xs_test.append(xs[indices[i]])
        ys_test.append(ys[indices[i]])

    # initializing list for losts
    training_losses = []
    test_losses = []
    
    batch_x, batch_y = deduce_batch(xs, ys, batch_size)
    for i in range(max_iter):

        if (i + 1) % (max_iter // 10) == 0:
            test_loss = loss(theta, xs_test, ys_test) / (1 - split_ratio)
            test_losses.append(test_loss)
        
        if i % epoch == 0:
            batch_x, batch_y = deduce_batch(xs, ys, batch_size)
        train_loss = loss(theta, batch_x, batch_y) / batch_size
        training_losses.append(train_loss)

        nabla_J = grad(theta, batch_x, batch_y)
        theta = update(theta, lr, nabla_J)

        if (i + 1) % (max_iter/20) == 0 and verbose:
                print("Iterating: [%d%%]\r" %int((i+1)/max_iter * 100), end="")
    if verbose:
        print("")

    return theta, training_losses, test_losses


def predict(theta, x):
    temp = np.ones(x.shape[0]+1)
    temp[1:] = x
    return int(sigmoid(theta, temp) > 0.5)