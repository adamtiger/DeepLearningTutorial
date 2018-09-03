import numpy as np
import json


def print_progress(status, max_iter, verbose):
    '''
    Prints the current progress as a precentage.
    '''
    if verbose:
        print("Iterating: [%d%%]\r" %int(status/max_iter * 100), end="")
        if int((status + 1)/max_iter) == 1:
            print("")

def scale(xs, factor):
    xs_scaled = []
    for x in xs:
        xs_scaled.append(x * factor)
    return xs_scaled

def add_bias_to_vectors(xs):
    '''
    The bias is taken into account by adding a 1 to the beginning of the vectors.
    '''
    xs_bias = []
    for x in xs:
        temp = np.ones(x.shape[0] + 1)
        temp[1:] = x
        xs_bias.append(temp)
    return xs_bias

def one_hot_encode(ys, k):
    '''
    Create the one-hot-encoded representation of a label.
    E.g.: k= 3, y = 2 => y_encoded = [0, 0, 1]
    ys - list of lables
    k - number of values
    '''
    ys_encoded = []
    for y in ys:
        temp = np.zeros(k)
        temp[y] = 1
        ys_encoded.append(temp)
    return ys_encoded

def create_binary_image(xs):
    '''
    Changes the values between 0 and 255 to 0 and 1.
    xs - list of input images (28x28 numpy arrays)
    '''
    xs_bin = []
    for x in xs:
        x_bin = (x > 20).astype(np.float32)
        xs_bin.append(x_bin)
    return xs_bin

def error_rate(ys_pred, ys):
    '''
    ys_pred - list of numpy arrays with probabilities of falling into a class
    ys - list of one-hot-encoded lables
    '''
    errors = 0.0
    for y_pred, y in zip(ys_pred, ys):
        if np.argmax(y_pred) != np.argmax(y):
            errors += 1.0
    return errors / len(ys)

def save_parameters(thetas, path):
    '''
    Saves the parameters of a model.
    thetas - list of parameters to be saved, the parameters are numpy matrices
    path - the path to the weight file
    '''
    thetas_as_list = []
    for theta in thetas:
        thetas_as_list.append(theta.tolist())
    with open(path, 'w') as file:
        json.dump(thetas_as_list, file)

def load_parameters(path):
    '''
    Loads the parameters for a model.
    path - the path to the saved weights
    '''
    with open(path, 'r') as file:
        thetas = json.load(file)
    return list(map(np.array, thetas))
