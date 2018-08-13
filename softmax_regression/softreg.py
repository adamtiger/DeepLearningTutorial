import numpy as np

# This is an implementation of softmax regression.
beta = 0.01

def softmax(theta, x):
    z = beta * np.matmul(theta, x)
    return np.exp(z)/np.sum(np.exp(z))


def error_rate(theta, xs, ys):
    '''
    Error rate
    '''
    errors = 0
    for x, y in zip(xs, ys):
        if np.argmax(softmax(theta, x)) != np.argmax(y):
            errors += 1
    return errors / len(xs)
    

def grad(theta, xs, ys):
    '''
    xs - the input values in a list, each value is a numpy array
    ys - the correct label for each input in a list
    beta is for avoiding overflow
    '''
    grad = np.zeros_like(theta)
    for x, y in zip(xs, ys):
        grad += beta * np.outer((y - softmax(theta, x)), x)
    return grad/np.linalg.norm(grad)


def update(theta, lr, gradient):
    return theta + lr * gradient


def init(theta_size, k):
    '''
    theta - weight matrix
    k - number of classes
    '''
    return np.random.uniform(size=(k, theta_size))


def deduce_batch(xs, ys, size):
    batch_x = []
    batch_y = []

    indices = np.array([t for t in range(len(xs))])
    np.random.shuffle(indices)
    for idx in indices.tolist():
        batch_x.append(xs[idx])
        batch_y.append(ys[idx])

    return batch_x, batch_y


def copy(data, fn=lambda x: x):
    '''
    creates a copy of data in order to leave the original input untouched
    and 
    data - a tuple of x, y
    '''
    xs_new, ys_new = [], []
    xs_o, ys_o = data
    for x, y in zip(xs_o, ys_o):
        xs_new.append(fn(np.copy(x)))
        ys_new.append(np.copy(y))
    return xs_new, ys_new


def softmax_regression(data, k, lr, max_iter, batch_size, epoch=5, verbose=False):
    '''
    data - a tuple containing two lists, first for the input and second for the labels,
           each element of the input list is a 1-D numpy array
    lr - learning rate
    max_iter - number of iterations during optimization
    batch_size - size of a batch
    epoch - number of repetition for a batch
    verbose - show progress during run
    '''
    xs, ys = copy(data, lambda t: t.astype(np.float32)/255.0)
    theta_size = xs[0].shape[0] + 1 # +1 due to bias
    theta = init(theta_size, k)

    # extend xs wih 1s
    for i, x in enumerate(xs):
        temp = np.ones(theta_size)
        temp[1:] = x
        xs[i] = temp
    
    # one-hot-encode y
    for i, y in enumerate(ys):
        temp = np.zeros(k)
        temp[y] = 1
        ys[i] = temp

    # initializing list for error rates
    training_err_rate = []
    
    batch_x, batch_y = deduce_batch(xs, ys, batch_size)
    for i in range(max_iter):
        
        if i % epoch == 0:
            batch_x, batch_y = deduce_batch(xs, ys, batch_size)

        train_errors = error_rate(theta, batch_x, batch_y)
        training_err_rate.append(train_errors)

        nabla_J = grad(theta, batch_x, batch_y)
        theta = update(theta, lr, nabla_J)

        if (i + 1) % (max_iter/20) == 0 and verbose:
                print("Iterating: [%d%%]\r" %int((i+1)/max_iter * 100), end="")
    if verbose:
        print("")

    return theta, training_err_rate


def predict(theta, x):
    '''
    x - the length is smaller with 1 compared to theta
    '''
    temp = np.zeros(theta.shape[1])
    temp[1:] = x
    return softmax(theta, temp)