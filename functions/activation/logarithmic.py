import numpy as np


def activation(x):
    return np.log(1 + np.exp(x))


def activation_der(x):
    return 1 / (1 + np.exp(-x))
