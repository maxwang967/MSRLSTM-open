import os
import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
import numpy as np
from tf.keras.callbacks import ModelCheckpoint


class Trainer:
    def __init__(self, train_args):
        self.train_args = train_args

    def train(self):
        train_args = self.train_args
        batch_size, data_type, device_id, epoch, model_name, train_path, validate_path, window_size = self._read_args(train_args)
        self._initial_gpu_env(device_id)
        model_module = __import__('models')
        model = getattr(model_module, train_args['model'])
        model = model(window_size, train_args['model_args'], data_type).build_model()
        print(model.summary())
        test_x, train_x = self._load_data(train_path, validate_path)
        self._choose_dataset(data_type, test_x, train_x)
        callbacks_list = self._initial_checkpoint(model_name)
        self._model_fit(batch_size, callbacks_list, epoch, model, data_type)

    def _read_args(self, train_args):
        epoch = train_args['epoch']
        batch_size = train_args['batch_size']
        model_name = train_args['model_name']
        train_path = train_args['train_path']
        validate_path = train_args['validate_path']
        window_size = train_args['window_size']
        device_id = train_args['device']
        data_type = train_args["datatype"]
        return batch_size, data_type, device_id, epoch, model_name, train_path, validate_path, window_size

    def _initial_gpu_env(self, device_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         sess = tf.Session(config=config)
#         KTF.set_session(sess)

    def _load_data(self, train_path, validate_path):
        train_data = np.load(train_path)
        test_data = np.load(validate_path)
        train_x = train_data["x"]
        self.train_y = train_data["y"]
        test_x = test_data["x"]
        self.test_y = test_data["y"]
        return test_x, train_x

    def _initial_checkpoint(self, model_name):
        checkpoint = ModelCheckpoint(
            model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        return callbacks_list

    def _choose_dataset(self, data_type, test_x, train_x):
        if "SHL" == data_type:
            self._initial_shl_dataset(test_x, train_x)
        elif "HTC" == data_type:
            self._initial_htc_dataset(test_x, train_x)
        elif "ICT" == data_type:
            self._initial_ict_dataset(test_x, train_x)

    def _model_fit(self, batch_size, callbacks_list, epoch, model, data_type):
        if "HTC" == data_type:
            model.fit({
                'gyrx_input': self.gyr_x,
                'gyry_input': self.gyr_y,
                'gyrz_input': self.gyr_z,
                'laccx_input': self.lacc_x,
                'laccy_input': self.lacc_y,
                'laccz_input': self.lacc_z,
                'magx_input': self.mag_x,
                'magy_input': self.mag_y,
                'magz_input': self.mag_z,
                'pres_input': self.pressure},
                {'output': self.train_y},
                callbacks=callbacks_list,
                epochs=epoch,
                shuffle=True,
                batch_size=batch_size,
                validation_data=({
                                     'gyrx_input': self.gyr_x_v,
                                     'gyry_input': self.gyr_y_v,
                                     'gyrz_input': self.gyr_z_v,
                                     'laccx_input': self.lacc_x_v,
                                     'laccy_input': self.lacc_y_v,
                                     'laccz_input': self.lacc_z_v,
                                     'magx_input': self.mag_x_v,
                                     'magy_input': self.mag_y_v,
                                     'magz_input': self.mag_z_v,
                                     'pres_input': self.pressure_v
                                 },
                                 {'output': self.test_y})
            )
        else:
            model.fit({
                'gyrx_input': self.gyr_x,
                'gyry_input': self.gyr_y,
                'gyrz_input': self.gyr_z,
                'laccx_input': self.lacc_x,
                'laccy_input': self.lacc_y,
                'laccz_input': self.lacc_z,
                'magx_input': self.mag_x,
                'magy_input': self.mag_y,
                'magz_input': self.mag_z},
                {'output': self.train_y},
                callbacks=callbacks_list,
                epochs=epoch,
                shuffle=True,
                batch_size=batch_size,
                validation_data=({
                                     'gyrx_input': self.gyr_x_v,
                                     'gyry_input': self.gyr_y_v,
                                     'gyrz_input': self.gyr_z_v,
                                     'laccx_input': self.lacc_x_v,
                                     'laccy_input': self.lacc_y_v,
                                     'laccz_input': self.lacc_z_v,
                                     'magx_input': self.mag_x_v,
                                     'magy_input': self.mag_y_v,
                                     'magz_input': self.mag_z_v,
                                 },
                                 {'output': self.test_y})
            )

    def _initial_ict_dataset(self, test_x, train_x):
        self.gyr_x = train_x[:, :, 6:7]
        self.gyr_y = train_x[:, :, 7:8]
        self.gyr_z = train_x[:, :, 8:9]
        self.lacc_x = train_x[:, :, 3:4]
        self.lacc_y = train_x[:, :, 4:5]
        self.lacc_z = train_x[:, :, 5:6]
        self.mag_x = train_x[:, :, 9:10]
        self.mag_y = train_x[:, :, 10:11]
        self.mag_z = train_x[:, :, 11:12]
        self.pressure = train_x[:, :, 12:13]
        self.gyr_x_v = test_x[:, :, 6:7]
        self.gyr_y_v = test_x[:, :, 7:8]
        self.gyr_z_v = test_x[:, :, 8:9]
        self.lacc_x_v = test_x[:, :, 3:4]
        self.lacc_y_v = test_x[:, :, 4:5]
        self.lacc_z_v = test_x[:, :, 5:6]
        self.mag_x_v = test_x[:, :, 9:10]
        self.mag_y_v = test_x[:, :, 10:11]
        self.mag_z_v = test_x[:, :, 11:12]
        self.pressure_v = test_x[:, :, 12:13]

    def _initial_htc_dataset(self, test_x, train_x):
        self.gyr_x = train_x[:, :, 6:7]
        self.gyr_y = train_x[:, :, 7:8]
        self.gyr_z = train_x[:, :, 8:9]
        self.lacc_x = train_x[:, :, 3:4]
        self.lacc_y = train_x[:, :, 4:5]
        self.lacc_z = train_x[:, :, 5:6]
        self.mag_x = train_x[:, :, 9:10]
        self.mag_y = train_x[:, :, 10:11]
        self.mag_z = train_x[:, :, 11:12]
        self.gyr_x_v = test_x[:, :, 6:7]
        self.gyr_y_v = test_x[:, :, 7:8]
        self.gyr_z_v = test_x[:, :, 8:9]
        self.lacc_x_v = test_x[:, :, 3:4]
        self.lacc_y_v = test_x[:, :, 4:5]
        self.lacc_z_v = test_x[:, :, 5:6]
        self.mag_x_v = test_x[:, :, 9:10]
        self.mag_y_v = test_x[:, :, 10:11]
        self.mag_z_v = test_x[:, :, 11:12]

    def _initial_shl_dataset(self, test_x, train_x):
        self.gyr_x = train_x[:, :, 6:7]
        self.gyr_y = train_x[:, :, 7:8]
        self.gyr_z = train_x[:, :, 8:9]
        self.lacc_x = train_x[:, :, 9:10]
        self.lacc_y = train_x[:, :, 10:11]
        self.lacc_z = train_x[:, :, 11:12]
        self.mag_x = train_x[:, :, 12:13]
        self.mag_y = train_x[:, :, 13:14]
        self.mag_z = train_x[:, :, 14:15]
        self.pressure = train_x[:, :, 19:20]
        self.gyr_x_v = test_x[:, :, 6:7]
        self.gyr_y_v = test_x[:, :, 7:8]
        self.gyr_z_v = test_x[:, :, 8:9]
        self.lacc_x_v = test_x[:, :, 9:10]
        self.lacc_y_v = test_x[:, :, 10:11]
        self.lacc_z_v = test_x[:, :, 11:12]
        self.mag_x_v = test_x[:, :, 9:10]
        self.mag_y_v = test_x[:, :, 10:11]
        self.mag_z_v = test_x[:, :, 11:12]
        self.pressure_v = test_x[:, :, 19:20]
