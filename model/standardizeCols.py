import numpy as np


# 对每一列进行标准化
def standardizeCols(M, mu=None, sigma2=None):
    # function [S,mu,sigma2] = standardize(M, mu, sigma2)
    # Make each column of M be zero mean, std 1.
    #
    # If mu, sigma2 are omitted, they are computed from M

    (nrows, ncols) = M.shape
    M.astype('float')
    if mu is None:
        mu = np.average(M, axis=0)
    if sigma2 is None:
        sigma2 = np.std(M, axis=0, ddof=1)
        sigma2 = (sigma2 < 2e-16) * 1 + (sigma2 >= 2e-16) * sigma2

    S = M - np.tile(mu, (nrows, 1))
    if ncols > 0:
        S = S / np.tile(sigma2, (nrows, 1))

    return S, mu, sigma2
