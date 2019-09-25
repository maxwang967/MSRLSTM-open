import numpy as np

from keras.engine.saving import load_model
import data

model_name = 'attention-g-ict.h5'
import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
path = "/public/lhy/data/npz/"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

if __name__ == "__main__":
    test_data = np.load(path + "pt-la-ne-test-450-0.000000-new-2018-11-14.npz")
    test_x = test_data["x"]
    test_y = test_data["y"]
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
         'magx_input': mag_x_v, 'magy_input': mag_y_v, 'magz_input': mag_z_v, 'pres_input': pressure_v})
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