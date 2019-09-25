import data
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
print("real one")
from revision.reviewer1.attention_ict import MSRLSTMNetwork

path = "/public/lhy/data/npz/"

window_size = 450
lstm_input = 36, 128
model_name = 'attention-g-ict.h5'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)


if __name__ == "__main__":
    # data = data.MSData(train_path, validate_path)
    model = MSRLSTMNetwork(window_size, lstm_input).build_model()
    print(model.summary())
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

    checkpoint = ModelCheckpoint(
        model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    history = model.fit({
        'gyrx_input': gyr_x, 'gyry_input': gyr_y, 'gyrz_input': gyr_z, 'laccx_input': lacc_x, 'laccy_input': lacc_y,
        'laccz_input': lacc_z, 'magx_input': mag_x, 'magy_input': mag_y, 'magz_input': mag_z, 'pres_input': pressure},
        {'output': train_y}, callbacks=callbacks_list, epochs=100, shuffle=True, batch_size=512,
        validation_data=({'gyrx_input': gyr_x_v, 'gyry_input': gyr_y_v, 'gyrz_input': gyr_z_v,
                          'laccx_input': lacc_x_v, 'laccy_input': lacc_y_v, 'laccz_input': lacc_z_v,
                          'magx_input': mag_x_v, 'magy_input': mag_y_v, 'magz_input': mag_z_v,
                          'pres_input': pressure_v},
                         {'output': test_y})
    )
    print(history)
