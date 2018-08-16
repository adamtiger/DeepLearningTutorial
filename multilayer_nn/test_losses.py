import numpy as np
from multilayer_nn import losses


def test_MSE():
    '''
    Test MSE loss function. Derivative is too simple therefore test is omitted for it.
    '''
    ys_pred = [[1.1, 2.2, 1.4], [0.5, 1.7, 2.5], [1.4, 24.3, 17.9]]
    ys_pred = list(map(np.array, ys_pred))

    ys = [[1.4, 1.5, 2.5], [0.8, 5.6, 3.6], [3.6, 21.6, 11.1]]
    ys = list(map(np.array, ys))

    # sure implementation
    L_expected = 0.0
    for yp_vec, y_vec in zip(ys_pred, ys):
        for yp, y in zip(yp_vec, y_vec):
            L_expected += (yp - y)**2
    L_expected /= 6.0

    # calculated
    L_calc = losses.MeanSquaredError().loss(ys_pred, ys)
    
    assert abs(L_calc - L_expected) < 0.000001, "Wrong MSE loss."
    print('MSE: OK')

test_MSE()

def test_CrossEntropy():
    '''
    Derivative is omitted here as well.
    '''
    ys_pred = [[0.05, 0.7, 0.25], [0.5, 0.5, 0.0], [0.2, 0.25, 0.55]]
    ys_pred = list(map(np.array, ys_pred))

    ys = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    ys = list(map(np.array, ys))

    # sure implementation
    L_expected = 0.0
    for yp_vec, y_vec in zip(ys_pred, ys):
        idx = np.argmax(y_vec)
        L_expected += np.log(yp_vec[idx])
    L_expected /= -3.0

    # calculated
    L_calc = losses.CrossEntropy().loss(ys_pred, ys)

    assert abs(L_calc - L_expected) < 0.000001, "Wrong CrossEntropy loss."
    print('CrossEntropy: OK')

test_CrossEntropy()

def test_Huber():
    
    ys_pred = [[0.05, 0.7, 0.25], [0.5, 0.5, 0.0], [0.2, 0.25, 0.55]]
    ys_pred = list(map(np.array, ys_pred))

    ys = [[0.2, 0.9, 0.24], [1.0, 0.1, 0.7], [0.3, 0.45, 1.1]]
    ys = list(map(np.array, ys))

    # sure implementation
    delta = 0.2
    L_expected = 0.0
    for yp_vec, y_vec in zip(ys_pred, ys):
        for yp, y in zip(yp_vec, y_vec):
            if abs(yp - y) <= delta:
                L_expected += 0.5 * (yp - y)**2
            else:
                L_expected += delta * (abs(yp - y) - 0.5 * delta)
    L_expected /= len(ys)

    # calculated
    L_calc = losses.Huber(delta=0.2).loss(ys_pred, ys)

    assert abs(L_calc - L_expected) < 0.000001, "Wrong Huber loss."
    print('Huber: OK')

test_Huber()
