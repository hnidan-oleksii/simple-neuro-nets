import numpy as np


def loss(true, pred):
    return np.mean(np.abs(true - pred))


def loss_der(true, pred):
    return np.where(pred > true, 1, -1)
