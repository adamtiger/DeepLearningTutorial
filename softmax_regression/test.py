import numpy as np

def multiplication():
    a = np.array([[1, 3], [2, 1], [5, 3]])
    b = np.array([1, 2])
    print(a, np.matmul(a, b))

multiplication()