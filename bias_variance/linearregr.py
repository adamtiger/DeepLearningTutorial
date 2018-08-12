import numpy as np

# An implementation for the linear regression with gradient descent.

def data_generator(n, fn, x_min=0.0, x_max=1.0, sigma=0.01):
    delta_x = (x_max - x_min) / (n - 1) # in case of n points the number of intervals are n-1
    
    x = np.array([x_min + t * delta_x for t in range(n)], dtype=float)
    y = fn(x)
    
    mean = [0, 0]
    cov = [[sigma, 0], [0, sigma]]
    x_noise, y_noise= np.random.multivariate_normal(mean, cov, n).T

    return x + x_noise, y + y_noise


# Initialize the weights
def init():
    return np.random.uniform(size=2)


# Calculate loss
def loss(theta, train_x, train_y):
    X = np.dot(train_x, theta)
    X = X - train_y
    X = np.square(X)
    return np.sum(X)/X.shape[0]


# Calculate gradient
def gradient(theta, train_x, train_y):
    grad = np.dot(train_x, theta)
    grad = grad - train_y
    grad = np.dot(grad, train_x)
    return grad/np.linalg.norm(grad)


# Update the weights
def update(theta, grad, lr=0.01):
    theta = theta - lr * grad
    return np.copy(theta)


def linear_regressor(data, lr=0.1, max_iter=100, split_ratio=0.8, verbose=False):
    
    '''
    data - a tuple containing the input (x) and the target (y)
    split_ratio - the ratio of the training
    verbose - to show the loss during training
    '''
    x, y = data[0], data[1]

    # Split the data randomly
    DATA_LEN = len(data[0])
    indices = np.array([x for x in range(DATA_LEN)], dtype=int)
    np.random.shuffle(indices)
    training_indices = indices[0:int(DATA_LEN*split_ratio)].tolist()
    test_indices = indices[int(DATA_LEN*split_ratio):-1].tolist()

    train_x = np.array([x[i] for i in training_indices])
    train_y = np.array([y[i] for i in training_indices])

    test_x = np.array([x[i] for i in test_indices])
    test_y = np.array([y[i] for i in test_indices])

    # Insert 1 at the end
    temp_x = np.ones((len(train_x), 2))
    temp_x[:, 0] = train_x
    train_x = temp_x

    temp_x = np.ones((len(test_x), 2))
    temp_x[:, 0] = test_x
    test_x = temp_x

    # Train
    theta = init()
    train_losses = []
    test_losses = []

    for itr in range(max_iter):
        train_L = loss(theta, train_x, train_y)
        test_L = loss(theta, test_x, test_y)
        grad = gradient(theta, train_x, train_y)
        theta = update(theta, grad, lr)
        train_losses.append((itr, train_L))
        test_losses.append((itr, test_L))
        if verbose and (itr + 1) % (max_iter // 10) == 0:
            print("Iterating: [%d%%]\r" %int(100*(itr + 1)/max_iter), end="")
    if verbose:
        print("")
    
    # Return: a, b, train_loss list(iter, loss), test_loss (iter, loss)
    return theta[0], theta[1], train_losses, test_losses


# Calculating the bias, variance and noise
def bias_var(ys_pred, y):
    '''
    ys_pred - numpy array containing the predictions with each function
    y - measured y in x
    '''
    bias = (np.average(ys_pred) - y)**2
    var = np.var(ys_pred)
    return bias, var


