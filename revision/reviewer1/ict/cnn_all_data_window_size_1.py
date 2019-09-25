# -*- coding: utf-8 -*-
# @Time    : 2018-08-15 9:24
# @Author  : morningstarwang
# @FileName: merge_data_1_window_size.py
# @Blog    ：https://morningstarwang.github.io
__author__ = 'morningstarwang'

import numpy as np
import numpy.linalg as la
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv1D, concatenate, Flatten, MaxPooling1D, Dropout, PReLU, BatchNormalization, LSTM
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
path = "/public/lhy/data/npz/"
model_name = 'ict_cnn.h5'
window_size = 450
num_classes = 6
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

train_data = np.load(path + "pt-la-ne-train-450-0.000000-new-2018-11-14.npz")
test_data = np.load(path + "pt-la-ne-test-450-0.000000-new-2018-11-14.npz")

train_x = train_data["x"]
train_y = train_data["y"]

test_x = test_data["x"]
test_y = test_data["y"]
"""
    column_names = ['accx', 'accy', 'accz',
                    'laccx', 'laccy', 'laccz',
                    'gyrx', 'gyry', 'gyrz',
                    'magx', 'magy', 'magz',
                    'pressure', 'cid', 'signal']
"""
gyr_x = train_x[:, :, 6:7]
gyr_y = train_x[:, :, 7:8]
gyr_z = train_x[:, :, 8:9]

lacc_x = train_x[:, :, 3:4]
lacc_y = train_x[:, :, 4:5]
lacc_z = train_x[:, :, 5:6]

mag_x = train_x[:, :, 9:10]
mag_y = train_x[:, :, 10:11]
mag_z = train_x[:, :, 11:12]
pressure = train_x[:, :, 12:13]

gyr_x_v = test_x[:, :, 6:7]
gyr_y_v = test_x[:, :, 7:8]
gyr_z_v = test_x[:, :, 8:9]
lacc_x_v = test_x[:, :, 3:4]
lacc_y_v = test_x[:, :, 4:5]
lacc_z_v = test_x[:, :, 5:6]
mag_x_v = test_x[:, :, 9:10]
mag_y_v = test_x[:, :, 10:11]
mag_z_v = test_x[:, :, 11:12]
pressure_v = test_x[:, :, 12:13]


def build_model():
    gyrx = Input(shape=(window_size, 1), dtype='float32', name='gyrx_input')
    gyry = Input(shape=(window_size, 1), dtype='float32', name='gyry_input')
    gyrz = Input(shape=(window_size, 1), dtype='float32', name='gyrz_input')
    laccx = Input(shape=(window_size, 1), dtype='float32', name='laccx_input')
    laccy = Input(shape=(window_size, 1), dtype='float32', name='laccy_input')
    laccz = Input(shape=(window_size, 1), dtype='float32', name='laccz_input')
    magx = Input(shape=(window_size, 1), dtype='float32', name='magx_input')
    magy = Input(shape=(window_size, 1), dtype='float32', name='magy_input')
    magz = Input(shape=(window_size, 1), dtype='float32', name='magz_input')
    pres = Input(shape=(window_size, 1), dtype='float32', name='pres_input')
    #
    # all_inputs = concatenate([gyrx, gyry, gyrz, laccx, laccy, laccz, magx, magy, magz], axis=1)
    # print(all_inputs.shape)
    convgyrx = Conv1D(64, kernel_size=3, activation='relu')(gyrx)
    convgyrx = MaxPooling1D(2)(convgyrx)
    convgyrx = Conv1D(128, kernel_size=3, activation='relu')(convgyrx)
    convgyrx = MaxPooling1D(2)(convgyrx)

    convgyry = Conv1D(64, kernel_size=3, activation='relu')(gyry)
    convgyry = MaxPooling1D(2)(convgyry)
    convgyry = Conv1D(128, kernel_size=3, activation='relu')(convgyry)
    convgyry = MaxPooling1D(2)(convgyry)

    convgyrz = Conv1D(64, kernel_size=3, activation='relu')(gyrz)
    convgyrz = MaxPooling1D(2)(convgyrz)
    convgyrz = Conv1D(128, kernel_size=3, activation='relu')(convgyrz)
    convgyrz = MaxPooling1D(2)(convgyrz)

    convlaccx = Conv1D(64, kernel_size=3, activation='relu')(laccx)
    convlaccx = MaxPooling1D(2)(convlaccx)
    convlaccx = Conv1D(128, kernel_size=3, activation='relu')(convlaccx)
    convlaccx = MaxPooling1D(2)(convlaccx)

    convlaccy = Conv1D(64, kernel_size=3, activation='relu')(laccy)
    convlaccy = MaxPooling1D(2)(convlaccy)
    convlaccy = Conv1D(128, kernel_size=3, activation='relu')(convlaccy)
    convlaccy = MaxPooling1D(2)(convlaccy)

    convlaccz = Conv1D(64, kernel_size=3, activation='relu')(laccz)
    convlaccz = MaxPooling1D(2)(convlaccz)
    convlaccz = Conv1D(128, kernel_size=3, activation='relu')(convlaccz)
    convlaccz = MaxPooling1D(2)(convlaccz)

    convmagx = Conv1D(64, kernel_size=3, activation='relu')(magx)
    convmagx = MaxPooling1D(2)(convmagx)
    convmagx = Conv1D(128, kernel_size=3, activation='relu')(convmagx)
    convmagx = MaxPooling1D(2)(convmagx)

    convmagy = Conv1D(64, kernel_size=3, activation='relu')(magy)
    convmagy = MaxPooling1D(2)(convmagy)
    convmagy = Conv1D(128, kernel_size=3, activation='relu')(convmagy)
    convmagy = MaxPooling1D(2)(convmagy)

    convmagz = Conv1D(64, kernel_size=3, activation='relu')(magz)
    convmagz = MaxPooling1D(2)(convmagz)
    convmagz = Conv1D(128, kernel_size=3, activation='relu')(convmagz)
    convmagz = MaxPooling1D(2)(convmagz)

    convpres = Conv1D(64, kernel_size=3, activation='relu')(pres)
    convpres = MaxPooling1D(2)(convpres)
    convpres = Conv1D(128, kernel_size=3, activation='relu')(convpres)
    convpres = MaxPooling1D(2)(convpres)
    convpres = Conv1D(128, kernel_size=3, activation='relu')(convpres)
    convpres = MaxPooling1D(2)(convpres)
    # #
    b = concatenate([convgyrx, convgyry, convgyrz])
    b = Conv1D(32, kernel_size=3, activation='relu')(b)
    b = MaxPooling1D(2)(b)

    b2 = concatenate([convlaccx, convlaccy, convlaccz])
    b2 = Conv1D(32, kernel_size=3, activation='relu')(b2)
    b2 = MaxPooling1D(2)(b2)

    c = concatenate([convmagx, convmagy, convmagz])
    c = Conv1D(32, kernel_size=3, activation='relu')(c)
    c = MaxPooling1D(2)(c)
    # b2 = concatenate([laccx, laccy, laccz])
    # b2 = Conv1D(64, kernel_size=3, activation='relu')(b2)
    # b2 = MaxPooling1D(2)(b2)
    # b2 = Conv1D(128, kernel_size=3, activation='relu')(b2)
    # b2 = MaxPooling1D(2)(b2)
    # b2 = Conv1D(256, kernel_size=3, activation='relu')(b2)
    # b2 = MaxPooling1D(2)(b2)
    x = concatenate([b, b2, c, convpres])

    # c2 = Flatten()(b2)
    c2 = Flatten()(x)
    c2 = Dense(128, activation='relu', kernel_initializer='truncated_normal')(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Dense(256, activation='relu', kernel_initializer='truncated_normal')(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Dense(512, activation='relu', kernel_initializer='truncated_normal')(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Dense(1024, activation='relu', kernel_initializer='truncated_normal')(c2)
    c2 = Dropout(0.2)(c2)
    # c2 = Dense(256, kernel_initializer='truncated_normal')(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = PReLU()(c2)
    # c2 = Dense(512, kernel_initializer='truncated_normal')(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = PReLU()(c2)
    # c2 = Dense(512, kernel_initializer='truncated_normal')(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = PReLU()(c2)
    # c2 = Dense(256, kernel_initializer='truncated_normal')(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = PReLU()(c2)
    # c2 = Dense(128, kernel_initializer='truncated_normal')(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = PReLU()(c2)
    # c2 = Dense(64, kernel_initializer='truncated_normal')(c2)
    # c2 = BatchNormalization()(c2)
    # c2 = PReLU()(c2)
    output = Dense(num_classes, activation='softmax', name='output')(c2)
    model = Model(inputs=[
        gyrx, gyry, gyrz,
        laccx, laccy, laccz,
        magx, magy, magz, pres
    ],
        outputs=[output])
    # model.layers.insert(0, LSTM(units=24, input_shape=(window_size, 1)))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_model()

print(model.summary())
# checkpoint
# filepath = "best_model.h5"
checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit({
    'gyrx_input': gyr_x, 'gyry_input': gyr_y, 'gyrz_input': gyr_z, 'laccx_input': lacc_x, 'laccy_input': lacc_y, 'laccz_input': lacc_z, 'magx_input': mag_x, 'magy_input': mag_y, 'magz_input': mag_z, 'pres_input': pressure}, {'output': train_y}, callbacks=callbacks_list, epochs=100, shuffle=True, batch_size=1024, validation_data=({ 'gyrx_input': gyr_x_v, 'gyry_input': gyr_y_v, 'gyrz_input': gyr_z_v,
                         'laccx_input': lacc_x_v, 'laccy_input': lacc_y_v, 'laccz_input': lacc_z_v,
                         'magx_input': mag_x_v, 'magy_input': mag_y_v, 'magz_input': mag_z_v, 'pres_input': pressure_v
                     }, {'output': test_y})

    # validation_split=0.1)
)
# model.save(model_name)