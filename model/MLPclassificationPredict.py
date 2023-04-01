import numpy as np

def MLPclassificationPredict(w, b, X, nHidden, nLabels, loss=False):
    nInstances, nVars = X.shape
    # Form Weights
    inputWeights = w[0:(nVars * nHidden[0])].reshape(nVars, nHidden[0], order='F')
    offset = nVars * nHidden[0]
    hiddenWeights = {}

    inputBias = b[0:nHidden[0]].reshape(1, nHidden[0], order='F')
    b_offset = nHidden[0]
    hiddenBias = {}

    for h in range(1, nHidden.shape[0]):
        hiddenWeights[h - 1] = w[offset:offset + nHidden[h - 1] * nHidden[h]].reshape(nHidden[h - 1], nHidden[h],
                                                                                      order='F')
        offset += nHidden[h - 1] * nHidden[h]

        hiddenBias[h - 1] = b[b_offset:b_offset + nHidden[h]].reshape(1, nHidden[h], order='F')
        b_offset += nHidden[h]

    outputWeights = w[offset:offset + nHidden[-1] * nLabels]
    outputWeights = outputWeights.reshape(nHidden[-1], nLabels, order='F')

    outputBias = b[b_offset:b_offset + nLabels].reshape(1, nLabels, order='F')

    # Compute Output
    y = np.zeros([nInstances, nLabels])
    ip = {}
    fp = {}
    ip[0] = X.dot(inputWeights) + inputBias
    fp[0] = np.tanh(ip[0])
    for h in range(1, nHidden.shape[0]):
        ip[h] = fp[h - 1].dot(hiddenWeights[h - 1]) + hiddenBias[h - 1]
        fp[h] = np.tanh(ip[h])
    y = fp[nHidden.shape[0] - 1].dot(outputWeights) + outputBias

    prob = np.exp(y)
    prob = prob / prob.sum(axis=1).reshape(prob.shape[0], 1)

    if loss:
        return prob.argmax(axis=1), prob

    return prob.argmax(axis=1)
