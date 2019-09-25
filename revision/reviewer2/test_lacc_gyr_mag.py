import numpy as np

from keras.engine.saving import load_model
import data
import sys

validate_path = '/public/lhy/data/npz/all_data_test_0.2_window_450_no_his_no_smooth.npz'
model_name = 'lacc_gyr_mag.h5'

if __name__ == "__main__":
    data = data.MSData(None, validate_path)
    _, lacc_x_v = data.get_lacc_x_data()
    _, lacc_y_v = data.get_lacc_y_data()
    _, lacc_z_v = data.get_lacc_z_data()

    _, gyr_x_v = data.get_gyr_x_data()
    _, gyr_y_v = data.get_gyr_y_data()
    _, gyr_z_v = data.get_gyr_z_data()

    _, mag_x_v = data.get_mag_x_data()
    _, mag_y_v = data.get_mag_y_data()
    _, mag_z_v = data.get_mag_z_data()

    _, pressure_v = data.get_pressure_data()

    validate_y = data.get_validate_y_data()

    model = load_model(model_name)
    predictions = model.predict(
        {
            'gyrx_input': gyr_x_v, 'gyry_input': gyr_y_v, 'gyrz_input': gyr_z_v,
         'laccx_input': lacc_x_v, 'laccy_input': lacc_y_v, 'laccz_input': lacc_z_v,
         'magx_input': mag_x_v, 'magy_input': mag_y_v, 'magz_input': mag_z_v,
         # 'pres_input': pressure_v
        })
    predictions = [np.argmax(p) for p in predictions]
    lv = [np.argmax(t) for t in validate_y]
    accuracy = 0
    cnf = [[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]]
    for (p, t) in zip(predictions, lv):
        cnf[t][p] += 1
        if p == t:
            accuracy += 1
    accuracy /= float(len(predictions))
    print('acc', accuracy)
    print(np.array(cnf))
    print('Still: %f\nWalk: %f\nRun: %f\nBike: %f\nCar: %f\nBus: %f\nTrain: %f\nSubway: %f' % (cnf[0][0] / float(np.sum(cnf[0])),
                                                                      cnf[1][1] / float(np.sum(cnf[1])),
                                                                      cnf[2][2] / float(np.sum(cnf[2])),
                                                                      cnf[3][3] / float(np.sum(cnf[3])),
                                                                      cnf[4][4] / float(np.sum(cnf[4])),
                                                                      cnf[5][5] / float(np.sum(cnf[5])),
                                                                      cnf[6][6] / float(np.sum(cnf[6])),
                                                                      cnf[7][7] / float(np.sum(cnf[7]))))
