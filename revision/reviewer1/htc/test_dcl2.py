# -*- coding: utf-8 -*-
"""
交通模式识别 原始数据训练
@author 王晨星
"""
import time

import numpy as np
from keras.models import Model, load_model
model_name = 'htc_dcl2.h5'
path = "/public/lhy/data/npz/"


def main():
    data = np.load(path + "htc-test-450-0.000000.npz")
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
    print("after train=" + str(int(round(time.time() * 1000))))
    model = load_model(model_name)
    predictions = model.predict(
        { 'gyrx_input': gyr_x_v, 'gyry_input': gyr_y_v, 'gyrz_input': gyr_z_v,
                         'laccx_input': lacc_x_v, 'laccy_input': lacc_y_v, 'laccz_input': lacc_z_v,
                         'magx_input': mag_x_v, 'magy_input': mag_y_v, 'magz_input': mag_z_v})

    predictions = [np.argmax(p) for p in predictions]
    lv = [np.argmax(t) for t in test_y]
    # print("pred="+str(predictions))
    # print("lv="+str(lv))菜单
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
    print('1: %f\n2: %f\n3: %f\n4: %f\n5: %f\n6: %f\n7: %f\n8: %f' % (cnf[0][0] / float(np.sum(cnf[0])),
                                                                      cnf[1][1] / float(np.sum(cnf[1])),
                                                                      cnf[2][2] / float(np.sum(cnf[2])),
                                                                      cnf[3][3] / float(np.sum(cnf[3])),
                                                                      cnf[4][4] / float(np.sum(cnf[4])),
                                                                      cnf[5][5] / float(np.sum(cnf[5])),
                                                                      cnf[6][6] / float(np.sum(cnf[6])),
                                                                      cnf[7][7] / float(np.sum(cnf[7]))))
    print("after predict=" + str(int(round(time.time() * 1000))))


if __name__ == '__main__':
    main()
