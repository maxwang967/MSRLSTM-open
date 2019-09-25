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
from keras.layers import Dense, Input, Conv1D, concatenate, Flatten, MaxPooling1D, Dropout, PReLU, BatchNormalization, \
    LSTM, Conv2D
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# np.savez("train_data_%d_in_%d.npz" % ((current_piece + 1), pieces), x=train_x, y=train_y)

# path = "/public/lhy/data/npz/"
path = "/public/lhy/data/npz/"
model_name = 'htc_dcl.h5'
window_size = 450
num_classes = 8

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

# train_x = np.array([])
# train_y = np.array([])
# test_x = np.array([])
# test_y = np.array([])
# train_indexes = [1, 2, 3, 4, 6, 7, 8, 9]
# test_indexes = [5, 10]
# for i in train_indexes:
train_data = np.load(path + "htc-train-450-0.000000.npz")
test_data = np.load(path + "htc-test-450-0.000000.npz")

train_x  = train_data["x"]
train_y = train_data["y"]

test_x = test_data["x"]
test_y = test_data["y"]


gyr_x = train_x[:, :, 6:7]
gyr_y = train_x[:, :, 7:8]
gyr_z = train_x[:, :, 8:9]

lacc_x = train_x[:, :, 3:4]
lacc_y = train_x[:, :, 4:5]
lacc_z = train_x[:, :, 5:6]

mag_x = train_x[:, :, 9:10]
mag_y = train_x[:, :, 10:11]
mag_z = train_x[:, :, 11:12]
# pressure = train_x[:, :, 19:20]

gyr_x_v = test_x[:, :, 6:7]
gyr_y_v = test_x[:, :, 7:8]
gyr_z_v = test_x[:, :, 8:9]
lacc_x_v = test_x[:, :, 3:4]
lacc_y_v = test_x[:, :, 4:5]
lacc_z_v = test_x[:, :, 5:6]
mag_x_v = test_x[:, :, 9:10]
mag_y_v = test_x[:, :, 10:11]
mag_z_v = test_x[:, :, 11:12]


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
    # pres = Input(shape=(window_size, 1), dtype='float32', name='pres_input')


    all_inputs = concatenate([gyrx, gyry, gyrz, laccx, laccy, laccz, magx, magy, magz], axis=2)
    print(all_inputs.shape)
    conv1 = Conv1D(64, kernel_size=5, padding='SAME', activation='relu')(all_inputs)
    conv1 = Conv1D(64, kernel_size=5, padding='SAME', activation='relu')(conv1)
    conv1 = Conv1D(64, kernel_size=5, padding='SAME', activation='relu')(conv1)
    conv1 = Conv1D(64, kernel_size=5, padding='SAME', activation='relu')(conv1)
    lstm1 = LSTM(128, input_shape=(300, 64), return_sequences=True)(conv1)
    lstm1 = LSTM(128, input_shape=(300, 64), return_sequences=False)(lstm1)
    output = Dense(8, activation='softmax', kernel_initializer='truncated_normal', name='output')(lstm1)
    model = Model(inputs=[
        gyrx, gyry, gyrz,
        laccx, laccy, laccz,
        magx, magy, magz
    ],
        outputs=[output])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_model()

print(model.summary())
# checkpoint

checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit({
    'gyrx_input': gyr_x, 'gyry_input': gyr_y, 'gyrz_input': gyr_z, 'laccx_input': lacc_x, 'laccy_input': lacc_y, 'laccz_input': lacc_z, 'magx_input': mag_x, 'magy_input': mag_y, 'magz_input': mag_z},
    {'output': train_y}, callbacks=callbacks_list, epochs=100, shuffle=True, batch_size=1024,
    validation_data=({'gyrx_input': gyr_x_v, 'gyry_input': gyr_y_v, 'gyrz_input': gyr_z_v,
                         'laccx_input': lacc_x_v, 'laccy_input': lacc_y_v, 'laccz_input': lacc_z_v,
                         'magx_input': mag_x_v, 'magy_input': mag_y_v, 'magz_input': mag_z_v},
                     {'output': test_y})
)
# model.save(model_name)
