import numpy as np


def MLPL2Loss(w, lamb):
    ret = lamb * w
    return ret


def MLPL1Loss(w, lamb):
    ret = lamb * (w >= 0) * np.ones(w.shape) + lamb * (w < 0) * (-np.ones(w.shape))
    return ret
