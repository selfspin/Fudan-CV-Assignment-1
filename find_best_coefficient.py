import argparse
from neuralNetwork import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', nargs='+', type=float, default=[1e-2, 1e-3, 1e-4, 1e-5],
                        help='learning rate')
    parser.add_argument('--iter', type=int, default=700000,
                        help='max iteration times')
    parser.add_argument('--layer', nargs='+', type=int, default=[50, 100, 200, 300],
                        help='size for the hidden layer')
    parser.add_argument('--regular', nargs='+', type=float, default=[1e-2, 1e-3, 1e-4, 1e-5],
                        help='coefficient of L2 regularization')

    args = parser.parse_args()

    min_error = 1

    for learning_rate in args.lr:
        for layer_size in args.layer:
            for regular in args.regular:
                w, b, valid_error, nHidden, nLabels, mu, sigma = train(learning_rate, layer_size,
                                                                       regular, args.iter)
                if valid_error < min_error:
                    min_error = valid_error
                    scio.savemat('./checkpoints/para.mat', mdict={'w': w, 'b': b, 'nHidden': nHidden,
                                                    'nLabels': nLabels, 'mu': mu, 'sigma': sigma})
                    scio.savemat('./checkpoints/train_para.mat', mdict={'lr': learning_rate, 'layer': layer_size,
                                                          'regular': regular})

    print('\nProcess is complete')
    data = scio.loadmat('train_para.mat')
    print('Best model training with learning_rate = {a}, layer_size = {b}, regular = {c}'.
          format(a=float(data['lr']), b=int(data['layer']), c=float(data['regular'])))
