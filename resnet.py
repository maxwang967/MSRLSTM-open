from keras import layers
from keras.layers import Activation, Conv1D, \
    MaxPooling1D


def convolutional_block(X, f, filters, stage, block, net_id, ):

    # defining name basis
    conv_name_base = 'res_' + net_id + "_" + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv1D(F1, 3, name=conv_name_base + '2a',
               padding='same')(X)
    # X = BatchNormalization(name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(2, padding='same')(X)

    # Second component of main path (≈3 lines)
    X = Conv1D(F2, f, name=conv_name_base + '2b',
               padding='same')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(2, padding='same')(X)

    # SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv1D(128, kernel_size=4, strides=4, padding='same')(X_shortcut)
    # X_shortcut = MaxPooling1D(2, padding='same')(X_shortcut)
    # X_shortcut = Conv1D(F2, 3, padding='same')(X_shortcut)
    # X_shortcut = MaxPooling1D(2, padding='same')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def res_net(X_input, net_id):
    X = convolutional_block(X_input, f=3, filters=[64, 128, 128], stage=2, block="a", net_id=net_id)
    X = MaxPooling1D(pool_size=2, padding="same")(X)
    return X
