import numpy as np


def loss(true, pred):
    return np.mean(np.square(true - pred))


def loss_der(true, pred):
    return -true + pred
