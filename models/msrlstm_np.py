from tf.keras.models import Model
from tf.keras.layers import Dense, Input, Conv1D, concatenate, Dropout, \
    LSTM, Activation, multiply

from models.modules import resnet


class MSRLSTMNP:

    def simple_cnn(self, X_input, net_id):
        X = Conv1D(filters=self.cnn_args[0], kernel_size=self.cnn_args[1],
                   kernel_initializer="glorot_uniform", name="simple_conv1_%s_" % net_id)(X_input)
        X = Activation("relu")(X)
        return X

    def __init__(self, window_size, model_args, data_type):
        self.window_size = window_size
        self.lstm_args = model_args["lstm"]
        self.cnn_args = model_args["cnn"]
        self.resnet_args = model_args["resnet"]
        self.attention_args = model_args["attention"]
        self.fc_args = model_args["fc"]
        self.dropout_args = model_args["dropout"]
        self.data_type = data_type

    def build_model(self):
        gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z = self.input_layer()
        gyr_x_cnn, gyr_y_cnn, gyr_z_cnn, lacc_x_cnn, lacc_y_cnn, lacc_z_cnn, mag_x_cnn, mag_y_cnn, mag_z_cnn = self.residual_layer(
            gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z)
        all_resnet = self.cnn_layer(gyr_x_cnn, gyr_y_cnn, gyr_z_cnn, lacc_x_cnn, lacc_y_cnn, lacc_z_cnn, mag_x_cnn,
                                    mag_y_cnn, mag_z_cnn)
        lstm = self.lstm_layer(all_resnet)
        lstm = self.attention_layer(lstm)
        output = self.mlp_layer(lstm)
        model = Model(inputs=[
            gyr_x, gyr_y, gyr_z,
            lacc_x, lacc_y, lacc_z,
            mag_x, mag_y, mag_z
        ],
            outputs=[output])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def mlp_layer(self, lstm):
        fc = Dense(self.fc_args[0], activation='relu', kernel_initializer='truncated_normal')(lstm)
        fc = Dropout(self.dropout_args)(fc)
        fc = Dense(self.fc_args[1], activation='relu', kernel_initializer='truncated_normal')(fc)
        fc = Dropout(self.dropout_args)(fc)
        fc = Dense(self.fc_args[2], activation='relu', kernel_initializer='truncated_normal')(fc)
        fc = Dropout(self.dropout_args)(fc)
        fc = Dense(self.fc_args[3], activation='relu', kernel_initializer='truncated_normal')(fc)
        fc = Dropout(self.dropout_args)(fc)
        output = Dense(self.fc_args[4], activation='softmax', name='output')(fc)
        return output

    def attention_layer(self, lstm):
        dense1 = Dense(self.attention_args[0], activation="softmax")(lstm)
        dense2 = Dense(self.attention_args[1], activation="softmax")(dense1)
        lstm = multiply([lstm, dense2])
        return lstm

    def lstm_layer(self, all_resnet):
        lstm = LSTM(self.lstm_args[0], input_shape=(self.lstm_args[1], self.lstm_args[2]), activation='tanh',
                    dropout=self.dropout_args, recurrent_dropout=self.dropout_args)(all_resnet)
        return lstm

    def cnn_layer(self, gyr_x_cnn, gyr_y_cnn, gyr_z_cnn, lacc_x_cnn, lacc_y_cnn, lacc_z_cnn, mag_x_cnn, mag_y_cnn,
                  mag_z_cnn, pressure_cnn):
        concat_lacc = concatenate(
            [lacc_x_cnn, lacc_y_cnn, lacc_z_cnn])
        concat_gyr = concatenate([gyr_x_cnn, gyr_y_cnn, gyr_z_cnn])
        concat_mag = concatenate([mag_x_cnn, mag_y_cnn, mag_z_cnn])
        concat_lacc_resnet = self.simple_cnn(concat_lacc, "concat_lacc")
        concat_gyr_resnet = self.simple_cnn(concat_gyr, "concat_gyr")
        concat_mag_resnet = self.simple_cnn(concat_mag, "concat_mag")
        all_resnet = concatenate(
            [concat_lacc_resnet, concat_gyr_resnet, concat_mag_resnet])
        return all_resnet

    def residual_layer(self, gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z, pressure):
        lacc_x_cnn = resnet.res_net(lacc_x, "single_lacc_x", self.resnet_args)
        lacc_y_cnn = resnet.res_net(lacc_y, "single_lacc_y", self.resnet_args)
        lacc_z_cnn = resnet.res_net(lacc_z, "single_lacc_z", self.resnet_args)
        gyr_x_cnn = resnet.res_net(gyr_x, "single_gyr_x", self.resnet_args)
        gyr_y_cnn = resnet.res_net(gyr_y, "single_gyr_y", self.resnet_args)
        gyr_z_cnn = resnet.res_net(gyr_z, "single_gyr_z", self.resnet_args)
        mag_x_cnn = resnet.res_net(mag_x, "single_mag_x", self.resnet_args)
        mag_y_cnn = resnet.res_net(mag_y, "single_mag_y", self.resnet_args)
        mag_z_cnn = resnet.res_net(mag_z, "single_mag_z", self.resnet_args)
        return gyr_x_cnn, gyr_y_cnn, gyr_z_cnn, lacc_x_cnn, lacc_y_cnn, lacc_z_cnn, mag_x_cnn, mag_y_cnn, mag_z_cnn

    def input_layer(self):
        lacc_x = Input(shape=(self.window_size, 1),
                       dtype='float32', name='laccx_input')
        lacc_y = Input(shape=(self.window_size, 1),
                       dtype='float32', name='laccy_input')
        lacc_z = Input(shape=(self.window_size, 1),
                       dtype='float32', name='laccz_input')
        gyr_x = Input(shape=(self.window_size, 1),
                      dtype='float32', name='gyrx_input')
        gyr_y = Input(shape=(self.window_size, 1),
                      dtype='float32', name='gyry_input')
        gyr_z = Input(shape=(self.window_size, 1),
                      dtype='float32', name='gyrz_input')
        mag_x = Input(shape=(self.window_size, 1),
                      dtype='float32', name='magx_input')
        mag_y = Input(shape=(self.window_size, 1),
                      dtype='float32', name='magy_input')
        mag_z = Input(shape=(self.window_size, 1),
                      dtype='float32', name='magz_input')
        return gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z
