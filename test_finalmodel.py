import numpy as np
import scipy.io as scio
from MLPclassificationPredict import *
from standardizeCols import *

if __name__ == '__main__':

    para = scio.loadmat('para.mat')
    w = para['w']
    b = para['b']
    mu = para['mu']
    sigma = para['sigma']
    nHidden = para['nHidden'].reshape([1])
    nLabels = int(para['nLabels'])

    Xtest = np.load('X_test.npy')
    ytest = np.load('y_test.npy')
    Xtest, _, _ = standardizeCols(Xtest, mu, sigma)
    n = Xtest.shape[0]

    yhat = MLPclassificationPredict(w, b, Xtest, nHidden, nLabels)
    yhat = yhat.reshape(yhat.shape[0], 1)
    print('Test error with final model = {error}'.format(
        error=float(sum(yhat != ytest) / n)))
