import numpy as np
from matplotlib.pyplot import plot, show

# An implementation for the linear regression with gradient descent.

def data_generator(n, a, b, x_min, x_max, sigma=0.01):
    delta_x = (x_max - x_min) / (n - 1) # in case of n points the number of intervals are n-1
    
    x = np.array([x_min + t * delta_x for t in range(n)], dtype=float)
    y = a* x + b
    
    mean = [0, 0]
    cov = [[sigma, 0], [0, sigma]]
    x_noise, y_noise = np.random.multivariate_normal(mu, cov, n)

    return x + x_noise, y + y_noise


def linear_regressor(data, split_ratio=0.8, verbose=False):
    # Split the data randomly

    # Insert 1 at the end
    
    # Initialize the weights

    # Calculate loss

    # Calculate gradient

    # Update the weights

    # Train

    # Test

    # Return: a, b, train_loss list(iter, loss), test_loss (one value)
    pass

    



