import numpy as np


def MLPclassificationLoss(w, b, X, y, nHidden, nLabels):
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

    g = None
    gb = None
    gHidden = {}
    gInput = np.zeros(inputWeights.shape)
    for h in range(1, nHidden.shape[0]):
        gHidden[h - 1] = np.zeros(hiddenWeights[h - 1].shape)
    gOutput = np.zeros(outputWeights.shape)

    gb_Input = np.zeros(inputBias.shape)
    gb_Hidden = {}
    for h in range(1, nHidden.shape[0]):
        gb_Hidden[h - 1] = np.zeros(hiddenBias[h - 1].shape)
    gb_Output = np.zeros(outputBias.shape)

    # Compute Output
    ip = {}
    fp = {}

    ip[0] = X.dot(inputWeights) + inputBias
    fp[0] = np.tanh(ip[0])
    for h in range(1, nHidden.shape[0]):
        ip[h] = fp[h - 1].dot(hiddenWeights[h - 1]) + hiddenBias[h - 1]
        fp[h] = np.tanh(ip[h])

    yhat = fp[nHidden.shape[0] - 1].dot(outputWeights) + outputBias

    prob = np.exp(yhat)
    prob = prob / prob.sum(axis=1).reshape(prob.shape[0], 1)

    err = prob
    for i in range(nInstances):
        err[i, y[i]] -= 1
    # Output Weights
    gOutput = np.dot(fp[nHidden.shape[0] - 1].transpose(), err)
    gb_Output = err.sum(axis=0)

    if nHidden.shape[0] > 1:
        # Last Layer of Hidden Weights
        backprop = np.dot(err, outputWeights.transpose()) * (1 - np.tanh(ip[nHidden.shape[0] - 1]) ** 2)
        gHidden[nHidden.shape[0] - 2] = fp[nHidden.shape[0] - 2].transpose().dot(backprop)
        gb_Hidden[nHidden.shape[0] - 2] = backprop.sum(axis=0)

        # Other Hidden Layers
        l = list(range(0, nHidden.shape[0] - 2))
        l.reverse()
        for h in l:
            backprop = backprop.dot(hiddenWeights[h + 1].transpose()) * (1 - np.tanh(ip[h + 1]) ** 2)
            gHidden[h] = fp[h].transpose().dot(backprop)
            gb_Hidden[h] = backprop.sum(axis=0)

        # Input Weights
        backprop = backprop.dot(hiddenWeights[0].transpose()) * (1 - np.tanh(ip[0]) ** 2)
        gInput = X.transpose().dot(backprop)
        gb_Input = backprop.sum(axis=0)

    else:
        # Input Weights
        backprop = np.dot(err, outputWeights.transpose()) * (1 - np.tanh(ip[0]) ** 2)
        gInput = X.transpose().dot(backprop)
        gb_Input = backprop.sum(axis=0)

    # Put Gradient into vector
    g = np.zeros(w.shape)
    g[0:nVars * nHidden[0]] = gInput.reshape([gInput.size, 1], order='F')
    offset = nVars * nHidden[0]
    for h in range(1, nHidden.shape[0]):
        g[offset:offset + nHidden[h - 1] * nHidden[h]] = gHidden[h - 1].reshape([gHidden[h - 1].size, 1], order='F')
        offset = offset + nHidden[h - 1] * nHidden[h]
    g[offset:offset + nHidden[-1] * nLabels] = gOutput.reshape([gOutput.size, 1], order='F')

    gb = np.zeros(b.shape)
    gb[0:nHidden[0]] = gb_Input.reshape([gb_Input.size, 1], order='F')
    boffset = nHidden[0]
    for h in range(1, nHidden.shape[0]):
        gb[boffset:boffset + nHidden[h]] = gb_Hidden[h - 1].reshape([gb_Hidden[h - 1].size, 1], order='F')
        boffset += nHidden[h]
    gb[boffset:boffset + nLabels] = gb_Output.reshape([gb_Output.size, 1])

    return g, gb
