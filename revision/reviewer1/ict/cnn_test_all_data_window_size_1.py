# -*- coding: utf-8 -*-
import numpy as np
from keras.models import  load_model

model_name = 'ict_cnn.h5'
path = "/public/lhy/data/npz/"


def main():
    data = np.load(path + "pt-la-ne-test-450-0.000000-new-2018-11-14.npz")
    test_x = data["x"]
    test_y = data["y"]
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

    model = load_model(model_name)
    predictions = model.predict(
        {'gyrx_input': gyr_x_v, 'gyry_input': gyr_y_v, 'gyrz_input': gyr_z_v,
         'laccx_input': lacc_x_v, 'laccy_input': lacc_y_v, 'laccz_input': lacc_z_v,
         'magx_input': mag_x_v, 'magy_input': mag_y_v, 'magz_input': mag_z_v,
         'pres_input': pressure_v})

    predictions = [np.argmax(p) for p in predictions]
    lv = [np.argmax(t) for t in test_y]
    # print("pred="+str(predictions))
    # print("lv="+str(lv))菜单
    accuracy = 0
    cnf = [[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]]
    for (p, t) in zip(predictions, lv):
        cnf[t][p] += 1
        if p == t:
            accuracy += 1
    accuracy /= float(len(predictions))
    print('acc', accuracy)
    print(np.array(cnf))
    print('Still: %f\nWalk: %f\nSubway: %f\nTrain: %f\nBus: %f\nCar: %f\n' % (cnf[0][0] / float(np.sum(cnf[0])),
                                            cnf[1][1] / float(np.sum(cnf[1])),
                                            cnf[2][2] / float(np.sum(cnf[2])),
                                            cnf[3][3] / float(np.sum(cnf[3])),
                                            cnf[4][4] / float(np.sum(cnf[4])),
                                            cnf[5][5] / float(np.sum(cnf[5])),
                                            ))


if __name__ == '__main__':
    main()
