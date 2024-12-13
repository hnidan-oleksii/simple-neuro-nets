import numpy as np


def activation(x):
    x = np.clip(x, -500, 500)
    return 1/(1 + np.exp(-x))


def activation_der(x):
    x = np.clip(x, -500, 500)
    return np.exp(-x) / np.power(1 + np.exp(-x), 2)


def activation_par_der(x, w):
    s = np.dot(x, w)
    return np.dot(x, activation_der(s))
