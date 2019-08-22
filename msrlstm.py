from keras.models import Model
from keras.layers import Dense, Input, Conv1D, concatenate, Dropout, \
    LSTM, Activation
import resnet


class MSRLSTMNetwork:

    @staticmethod
    def simple_cnn(X_input, net_id):
        X = Conv1D(filters=32, kernel_size=3,
                   kernel_initializer="glorot_uniform", name="simple_conv1_%s_" % net_id)(X_input)
        X = Activation("relu")(X)
        return X

    def __init__(self, window_size, lstm_input):
        self.window_size = window_size
        self.lstm_input = lstm_input

    def build_model(self):
        # Linear Acceleration Input Layer
        lacc_x = Input(shape=(self.window_size, 1),
                       dtype='float32', name='laccx_input')
        lacc_y = Input(shape=(self.window_size, 1),
                       dtype='float32', name='laccy_input')
        lacc_z = Input(shape=(self.window_size, 1),
                       dtype='float32', name='laccz_input')
        # Gyroscope Input Layer
        gyr_x = Input(shape=(self.window_size, 1),
                      dtype='float32', name='gyrx_input')
        gyr_y = Input(shape=(self.window_size, 1),
                      dtype='float32', name='gyry_input')
        gyr_z = Input(shape=(self.window_size, 1),
                      dtype='float32', name='gyrz_input')
        # Magnetometer Input Layer
        mag_x = Input(shape=(self.window_size, 1),
                      dtype='float32', name='magx_input')
        mag_y = Input(shape=(self.window_size, 1),
                      dtype='float32', name='magy_input')
        mag_z = Input(shape=(self.window_size, 1),
                      dtype='float32', name='magz_input')
        # Pressure Input Layer
        pressure = Input(shape=(self.window_size, 1),
                         dtype='float32', name='pres_input')
        # Linear Accerleration
        lacc_x_cnn = resnet.res_net(lacc_x, "single_lacc_x")
        lacc_y_cnn = resnet.res_net(lacc_y, "single_lacc_y")
        lacc_z_cnn = resnet.res_net(lacc_z, "single_lacc_z")
        # Gyroscope
        gyr_x_cnn = resnet.res_net(gyr_x, "single_gyr_x")
        gyr_y_cnn = resnet.res_net(gyr_y, "single_gyr_y")
        gyr_z_cnn = resnet.res_net(gyr_z, "single_gyr_z")
        # Magnetometer
        mag_x_cnn = resnet.res_net(mag_x, "single_mag_x")
        mag_y_cnn = resnet.res_net(mag_y, "single_mag_y")
        mag_z_cnn = resnet.res_net(mag_z, "single_mag_z")
        # Pressure
        pressure_cnn = resnet.res_net(pressure, "single_pressure")
        # End of ResNet50 Layer for Input Layers
        # Concatenate Axis For Each Sensor
        concat_lacc = concatenate(
            [lacc_x_cnn, lacc_y_cnn, lacc_z_cnn])
        concat_gyr = concatenate([gyr_x_cnn, gyr_y_cnn, gyr_z_cnn])
        concat_mag = concatenate([mag_x_cnn, mag_y_cnn, mag_z_cnn])
        concat_lacc_resnet = self.simple_cnn(concat_lacc, "concat_lacc")
        concat_gyr_resnet = self.simple_cnn(concat_gyr, "concat_gyr")
        concat_mag_resnet = self.simple_cnn(concat_mag, "concat_mag")
        concat_pressure_resnet = self.simple_cnn(pressure_cnn, "concat_pressure")
        # Concatenate All ResNet50 Layers
        all_resnet = concatenate(
            [concat_lacc_resnet, concat_gyr_resnet, concat_mag_resnet, concat_pressure_resnet])
        # all_resnet = resnet50.ResNet50(all_resnet, "all_resnet")
        lstm = LSTM(128, input_shape=self.lstm_input, activation='tanh',
                    dropout=0.2, recurrent_dropout=0.2)(all_resnet)
        fc = Dense(128, activation='relu', kernel_initializer='truncated_normal')(lstm)
        fc = Dropout(0.2)(fc)
        fc = Dense(256, activation='relu', kernel_initializer='truncated_normal')(fc)
        fc = Dropout(0.2)(fc)
        fc = Dense(512, activation='relu', kernel_initializer='truncated_normal')(fc)
        fc = Dropout(0.2)(fc)
        fc = Dense(1024, activation='relu', kernel_initializer='truncated_normal')(fc)
        fc = Dropout(0.2)(fc)
        output = Dense(8, activation='softmax', name='output')(fc)
        model = Model(inputs=[
            gyr_x, gyr_y, gyr_z,
            lacc_x, lacc_y, lacc_z,
            mag_x, mag_y, mag_z, pressure
        ],
            outputs=[output])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
