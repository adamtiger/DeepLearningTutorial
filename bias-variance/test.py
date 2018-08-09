import linearregr as L
import numpy as np
from matplotlib import pyplot as plt

def test_loss():
    theta = np.array([1.0, 0.0])
    x = np.array([[0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0], [0.5, 1.0]])
    y = np.array([0.05, 0.3, 0.35, 0.42, 0.54])
    loss_0 = L.loss(theta, x, y)
    loss_1 = 0.0
    for i in range(y.shape[0]):
        y_ = theta[0] * x[i][0] + theta[1]
        loss_1 += (y_ - y[i])**2
    loss_1 = loss_1 / y.shape[0]
    
    assert abs(loss_0 - loss_1) < 0.001, "Wrong loss value!"
    print("Loss: Correct!")

test_loss()

def test_gradient():
    theta = np.array([1.0, 0.0])
    x = np.array([[0.1, 1.0], [0.2, 1.0], [0.3, 1.0], [0.4, 1.0], [0.5, 1.0]])
    y = np.array([0.05, 0.3, 0.35, 0.42, 0.54])
    grad_0 = L.gradient(theta, x, y)
    grad_1 = np.array([0.0, 0.0])
    for i in range(y.shape[0]):
        y_ = theta[0] * x[i][0] + theta[1]
        grad_1 += (y_ - y[i]) * x[i]
    grad_1 /= np.linalg.norm(grad_1)
    
    assert np.allclose(grad_0, grad_1), "Wrong gradient value!"
    print("Gradient: Correct!")

test_gradient()

def test_generator():
    x, y = L.data_generator(30, lambda x: 1.0 *x + 0.0, sigma=0.001)
    plt.plot(x, y, 'ro')
    plt.show()

test_generator()

def test_bias_var_noise():
    y_q = 1.1
    ys_pred = np.array([1.05, 1.0, .97, .99, 1.08])

    b, v = L.bias_var(ys_pred, y_q)
    eps = np.average(np.square(ys_pred - y_q))
    assert abs(b+v - eps) < 0.000001, "Wrong bias-variance calculation!"
    print("Bias-var: Correct!")

test_bias_var_noise()
