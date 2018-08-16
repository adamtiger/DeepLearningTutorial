import numpy as np
import math as m
from multilayer_nn import activations


def check(name, y_exp, y_calc, dy_exp, dy_calc):
    assert np.allclose(y_exp, y_calc), "Wrong activation in " + name
    assert np.allclose(dy_exp, dy_calc), "Wrong derivative of activation in " + name
    print(name + ": OK")

def test_linear():
    
    x = np.array([2, 1, 0.6, 0.8, 1.5])
    a, b = 1.6, 1.2

    # expected
    y_expected = a * x + b
    d_y_expected = np.zeros((5, 5))
    for i in range(5):
        d_y_expected[i, i] = a

    # calculated
    act = activations.Linear(a, b)
    y_calc = act.activate(x)
    d_y_calc = act.d_activate(x)

    check("Linear", y_expected, y_calc, d_y_expected, d_y_calc)

test_linear()

def test_relu():

    x = np.array([-1.4, 0.6, 1.7, -0.5, 0.01])
    
    # expected
    y_expected = np.zeros_like(x)
    for i, x_ in enumerate(x):
        if x_ < 0.0:
            y_expected[i] = 0.0
        else:
            y_expected[i] = x_
    
    d_y_expected = np.zeros((5, 5))
    for i, x_ in enumerate(x):
        if x_ < 0.0:
            d_y_expected[i, i] = 0
        else:
            d_y_expected[i, i] = 1
    
    # calc
    act = activations.Relu()
    y_calc = act.activate(x)
    d_y_calc = act.d_activate(x)

    check("Relu", y_expected, y_calc, d_y_expected, d_y_calc)

test_relu()

def test_tanh():

    x = np.array([-4.2, -1.6, 0.12, 0.78, 2.3])

    # expected
    y_expected = np.zeros_like(x)
    for i, x_ in enumerate(x):
        y_expected[i] = (m.exp(x_) - m.exp(-x_)) / (m.exp(x_) + m.exp(-x_))
    
    d_y_expected = np.zeros((5, 5))
    for i, x_ in enumerate(x):
        chx = (m.exp(x_) + m.exp(-x_)) / 2.0
        d_y_expected[i, i] = 1.0 / chx**2
    
    # calculated
    act = activations.Tanh()
    y_calc = act.activate(x)
    d_y_calc = act.d_activate(x)

    check("Tanh", y_expected, y_calc, d_y_expected, d_y_calc)

test_tanh()

def test_sigmoid():

    x = np.array([-4.2, -1.6, 0.12, 0.78, 2.3])

    # expected
    y_expected = np.zeros_like(x)
    for i, x_ in enumerate(x):
        y_expected[i] = 1.0 / (1.0 + m.exp(-x_))
    
    d_y_expected = np.zeros((5, 5))
    for i, x_ in enumerate(x):
        s = 1.0 / (1.0 + m.exp(-x_))
        d_y_expected[i, i] = s * (1-s)
    
    # calculated
    act = activations.Sigmoid()
    y_calc = act.activate(x)
    d_y_calc = act.d_activate(x)

    check("Sidmoid", y_expected, y_calc, d_y_expected, d_y_calc)

test_sigmoid()

def test_softmax():

    x = np.array([-4.2, -1.6, 0.12, 0.78, 2.3])

    # expected
    y_expected = np.zeros_like(x)
    denom = 0.0
    for i, x_ in enumerate(x):
        denom += m.exp(x_)
        y_expected[i] = m.exp(x_)
    y_expected /= denom
    
    d_y_expected = np.zeros((5, 5))
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if i == j:
                d_y_expected[i, j] = m.exp(xi)/denom * (1.0 - m.exp(xj)/denom)
            else:
                d_y_expected[i, j] = -m.exp(xi)/denom * m.exp(xj)/denom

    # calculated
    act = activations.Softmax()
    y_calc = act.activate(x)
    d_y_calc = act.d_activate(x)

    check("Softmax", y_expected, y_calc, d_y_expected, d_y_calc)

test_softmax()
