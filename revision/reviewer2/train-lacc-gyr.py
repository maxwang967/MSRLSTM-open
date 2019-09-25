from keras.utils import plot_model

import msrlstm_lacc_gyr
import data
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
train_path = '/public/lhy/data/npz/all_data_train_0.8_window_450_no_his_no_smooth.npz'
validate_path = '/public/lhy/data/npz/all_data_test_0.2_window_450_no_his_no_smooth.npz'
window_size = 450
model_name = 'lacc_gyr.h5'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

if __name__ == "__main__":
    data = data.MSData(train_path, validate_path)
    model = msrlstm_lacc_gyr.MSRLSTMNetwork(window_size, (55, 128)).build_model()
    print(model.summary())
    # plot_model(model, to_file='model1.png', show_shapes=True)

    lacc_x, lacc_x_v = data.get_lacc_x_data()
    lacc_y, lacc_y_v = data.get_lacc_y_data()
    lacc_z, lacc_z_v = data.get_lacc_z_data()

    gyr_x, gyr_x_v = data.get_gyr_x_data()
    gyr_y, gyr_y_v = data.get_gyr_y_data()
    gyr_z, gyr_z_v = data.get_gyr_z_data()
    #
    # mag_x, mag_x_v = data.get_mag_x_data()
    # mag_y, mag_y_v = data.get_mag_y_data()
    # mag_z, mag_z_v = data.get_mag_z_data()
    #
    # pressure, pressure_v = data.get_pressure_data()

    train_y = data.get_train_y_data()
    validate_y = data.get_validate_y_data()

    checkpoint = ModelCheckpoint(
        model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit({
        'gyrx_input': gyr_x,
        'gyry_input': gyr_y,
        'gyrz_input': gyr_z,
        'laccx_input': lacc_x,
        'laccy_input': lacc_y,
        'laccz_input': lacc_z,
        # 'magx_input': mag_x,
        # 'magy_input': mag_y,
        # 'magz_input': mag_z,
        # 'pres_input': pressure
    },
        {'output': train_y}, callbacks=callbacks_list, epochs=100, shuffle=True, batch_size=512,
        validation_data=({
                             'gyrx_input': gyr_x_v,
                             'gyry_input': gyr_y_v,
                             'gyrz_input': gyr_z_v,
                             'laccx_input': lacc_x_v,
                             'laccy_input': lacc_y_v,
                             'laccz_input': lacc_z_v,
                             # 'magx_input': mag_x_v,
                             # 'magy_input': mag_y_v,
                             # 'magz_input': mag_z_v,
                             # 'pres_input': pressure_v
                         },
                         {'output': validate_y})
    )
