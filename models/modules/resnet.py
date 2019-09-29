from keras import layers
from keras.layers import Activation, Conv1D, \
    MaxPooling1D


def convolutional_block(X, f, filters, kernel_sizes, stride, net_id):
    conv_name_base = 'res_' + net_id + "_" + str(2) + 'a' + '_branch'

    F1, F2, F3, F4 = filters
    K1, K2, K3, K4 = kernel_sizes

    X_shortcut = X

    X = Conv1D(F1, K1, name=conv_name_base + '2a',
               padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(K2, padding='same')(X)

    X = Conv1D(F2, f, name=conv_name_base + '2b',
               padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(K3, padding='same')(X)

    X_shortcut = Conv1D(F4, kernel_size=K4, strides=stride, padding='same')(X_shortcut)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def res_net(X_input, net_id, args):
    X = convolutional_block(X_input, f=args['f'], filters=args['filters'], kernel_sizes=args['kernel_sizes'],
                            stride=args['s'], net_id=net_id)
    X = MaxPooling1D(pool_size=args['p'], padding="same")(X)
    return X
