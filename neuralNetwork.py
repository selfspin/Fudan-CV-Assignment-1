import scipy.io as scio
import numpy as np
from model.standardizeCols import *
from model.MLPclassificationPredict import *
from model.MLPclassificationLoss import *
from model.MLPRegularLoss import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def train(learning_rate=0.0001, layer_size=300, regular=0.01, maxIter=700000, curve=False):
    if curve:
        train_error = []
        train_loss = []
        valid_error = []
        valid_loss = []
        times = []

    np.random.seed(0)

    X = np.load('./data/X_train.npy')
    y = np.load('./data/y_train.npy')

    Xtest = np.load('./data/X_test.npy')
    ytest = np.load('./data/y_test.npy')

    X, Xvalid, y, yvalid = train_test_split(X, y, test_size=0.1, random_state=0)

    (n, d) = X.shape
    nLabels = y.max() + 1
    t = Xvalid.shape[0]
    t2 = Xtest.shape[0]

    # Standardize columns and add bias
    X, mu, sigma = standardizeCols(X)
    # Make sure to apply the same transformation to the validation/test data
    Xvalid, _, _ = standardizeCols(Xvalid, mu, sigma)
    Xtest, _, _ = standardizeCols(Xtest, mu, sigma)

    # Choose network structure
    nHidden = np.array([layer_size])

    # Count number of parameters and initialize weights 'w'
    nParams = d * nHidden[0]
    bParams = nHidden[0]
    for h in range(1, nHidden.shape[0]):
        nParams += nHidden[h - 1] * nHidden[h]
        bParams += nHidden[h]

    nParams += nHidden[-1] * nLabels
    bParams += nLabels

    # data = scio.loadmat('w.mat')
    # w = data['w']
    w = np.random.randn(nParams, 1)
    b = np.random.randn(bParams, 1)

    Mw = np.zeros(w.shape)
    Gw = np.zeros(w.shape)
    Mb = np.zeros(b.shape)
    Gb = np.zeros(b.shape)

    # Train with stochastic gradient
    stepSize = learning_rate
    beta1 = 0.9
    beta2 = 0.99
    lamb = regular
    minibatch = 8
    eps = 1e-9

    # 提前停止
    max_not_decay = 20
    not_decay = 0
    min_error = 1

    print('Training with learning_rate = {a}, layer_size = {b}, regular = {c}'.
          format(a=learning_rate, b=layer_size, c=regular))

    for iter in range(maxIter):
        if iter % 5000 == 0:
            if curve:
                yhat, prob = MLPclassificationPredict(w, b, Xvalid, nHidden, nLabels, loss=True)
                yhat = yhat.reshape(yhat.shape[0], 1)
                error = float(sum(yhat != yvalid) / t)
                print('Training iteration = {iter}, validation error = {error}'.format(
                    iter=iter, error=error))
                valid_error.append(error)
                loss = 0
                for i in range(t):
                    loss += - np.log(prob[i][yvalid[i]])
                valid_loss.append(loss / t)

                yhat, prob = MLPclassificationPredict(w, b, X, nHidden, nLabels, loss=True)
                yhat = yhat.reshape(yhat.shape[0], 1)
                train_error.append(float(sum(yhat != y) / n))
                loss = 0
                for i in range(n):
                    loss += - np.log(prob[i][y[i]])
                train_loss.append(loss / n)

                times.append(iter)

            else:
                yhat = MLPclassificationPredict(w, b, Xvalid, nHidden, nLabels)
                yhat = yhat.reshape(yhat.shape[0], 1)
                error = float(sum(yhat != yvalid) / t)
                print('Training iteration = {iter}, validation error = {error}'.format(
                    iter=iter, error=error))

            if error < min_error:
                min_error = error
                best_b = b.copy()
                best_w = w.copy()
                not_decay = 0
            else:
                not_decay += 1
                if not_decay >= max_not_decay:
                    break

        i = np.floor(np.random.rand(minibatch) * n).astype(int)
        gw, gb = MLPclassificationLoss(w, b, X[i, :], y[i], nHidden, nLabels)
        gw += MLPL2Loss(w, lamb)

        iter += 1

        Mw = beta1 * Mw + (1 - beta1) * gw
        Gw = beta2 * Gw + (1 - beta2) * gw * gw
        w = w - stepSize / np.sqrt(Gw / (1 - beta2 ** iter) + eps) * Mw / (1 - beta1 ** iter)

        Mb = beta1 * Mb + (1 - beta1) * gb
        Gb = beta2 * Gb + (1 - beta2) * gb * gb
        b = b - stepSize / np.sqrt(Gb / (1 - beta2 ** iter) + eps) * Mb / (1 - beta1 ** iter)

    yhat = MLPclassificationPredict(best_w, best_b, X, nHidden, nLabels)
    yhat = yhat.reshape(yhat.shape[0], 1)
    print('Train error with final model = {error}'.format(
        error=float(sum(yhat != y) / n)))

    yhat = MLPclassificationPredict(best_w, best_b, Xvalid, nHidden, nLabels)
    yhat = yhat.reshape(yhat.shape[0], 1)
    va_error = float(sum(yhat != yvalid) / t)
    print('Validation error with final model = {error}'.format(error=error))

    # Evaluate test error
    yhat = MLPclassificationPredict(best_w, best_b, Xtest, nHidden, nLabels)
    yhat = yhat.reshape(yhat.shape[0], 1)
    print('Test error with final model = {error}\n'.format(
        error=float(sum(yhat != ytest) / t2)))

    if args.save_model:
        scio.savemat('para.mat', mdict={'w': best_w, 'b': best_b, 'nHidden': nHidden,
                                        'nLabels': nLabels, 'mu': mu, 'sigma': sigma})

    if curve:
        return train_error, train_loss, valid_error, valid_loss, times

    return best_w, best_b, va_error, nHidden, nLabels, mu, sigma


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--iter', type=int, default=700000,
                        help='iteration times')
    parser.add_argument('--layer', type=int, default=300,
                        help='size for the hidden layer')
    parser.add_argument('--regular', type=float, default=0.01,
                        help='coefficient of L2 regularization')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='use if need to save the model checkpoint')

    args = parser.parse_args()

    train_error, train_loss, valid_error, valid_loss, times = train(args.lr, args.layer, args.regular, args.iter, True)

    # scio.savemat('curve.mat', mdict={'train_error': train_error, 'train_loss': train_loss,
    #                                 'valid_error': valid_error, 'valid_loss': valid_loss,
    #                                 'times': times})

    plt.plot(times, np.log(train_error), label='Error on the training set')
    plt.plot(times, np.log(valid_error), label='Error on the validation set')
    plt.xlabel('Iteration')
    plt.ylabel('log Error')
    plt.legend()
    plt.savefig('error.jpg')

    plt.figure()
    plt.plot(times, np.log(train_loss), label='Loss on the training set')
    plt.plot(times, np.log(valid_loss), label='Loss on the validation set')
    plt.xlabel('Iteration')
    plt.ylabel('log Loss')
    plt.legend()
    plt.savefig('loss.jpg')
