# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from tf.keras.utils import to_categorical

train_split_ratio = 0.5  # 训练集占80%
num_classes = 8


def read_data_all(path):
    """
    读取集成的训练+测试数据
    :param path: 文件路径
    :param piece: 第几份
    :param pieces: 总份数
    :return:
    """
    column_names = ['timestamp',
                    'acc_x', 'acc_y', 'acc_z',
                    'gra_x', 'gra_y', 'gra_z',
                    'gyr_x', 'gyr_y', 'gyr_z',
                    'lacc_x', 'lacc_y', 'lacc_z',
                    'mag_x', 'mag_y', 'mag_z',
                    'ori_w', 'ori_x', 'ori_y', 'ori_z',
                    'pressure', 'label']
    global data

    for i in range(8):
        current_path = path + "Label_%d.txt" % (i + 1)
        if i == 0:
            data = pd.read_csv(current_path, header=None, names=column_names, sep=",")
        else:
            data = pd.concat([data, pd.read_csv(current_path, header=None, names=column_names, sep=",")])
    return data


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    print("mu=%s" % str(mu))
    print("sigma=%s" % str(sigma))
    return (dataset - mu) / sigma


def windows(data, size, overlap):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (int(size * (1 - overlap)))  # 步长


def segment_signal(data, window_size, overlap):
    segments = np.empty((0, window_size, 20))
    labels = np.empty((0))
    data['label'] = [x - 1 for x in data['label']]
    count = 0
    for (start, end) in windows(data['timestamp'], window_size, overlap):
        count += 1
        print('segmentation: %d' % count)
        acc_x = data['acc_x'][start:end]
        acc_y = data['acc_y'][start:end]
        acc_z = data['acc_z'][start:end]
        gra_x = data['gra_x'][start:end]
        gra_y = data['gra_y'][start:end]
        gra_z = data['gra_z'][start:end]
        gyr_x = data['gyr_x'][start:end]
        gyr_y = data['gyr_y'][start:end]
        gyr_z = data['gyr_z'][start:end]
        lacc_x = data['lacc_x'][start:end]
        lacc_y = data['lacc_y'][start:end]
        lacc_z = data['lacc_z'][start:end]
        mag_x = data['mag_x'][start:end]
        mag_y = data['mag_y'][start:end]
        mag_z = data['mag_z'][start:end]
        ori_w = data['ori_w'][start:end]
        ori_x = data['ori_x'][start:end]
        ori_y = data['ori_y'][start:end]
        ori_z = data['ori_z'][start:end]
        pressure = data['pressure'][start:end]
        if len(data['timestamp'][start:end]) == window_size:
            segments_tmp = np.dstack([acc_x, acc_y, acc_z,
                                      gra_x, gra_y, gra_z,
                                      gyr_x, gyr_y, gyr_z,
                                      lacc_x, lacc_y, lacc_z,
                                      mag_x, mag_y, mag_z,
                                      ori_w, ori_x, ori_y, ori_z,
                                      pressure])
            segments = np.vstack([segments, segments_tmp])
            del segments_tmp
            labels = np.append(labels, stats.mode(data['label'][start:end])[0][0])
    return segments, labels


def main(_window_size, overlap, path):
    """
      11743701 ./Label_Label8.txt
      14864887 ./Label_Label5.txt
      12585507 ./Label_Label4.txt
      13798387 ./Label_Label1.txt
       12343944 ./Label_Label3.txt
      13120921 ./Label_Label2.txt
      15102310 ./Label_Label7.txt
      12529639 ./Label_Label6.txt
      97860000 total
    """
    print("window_size=%d" % _window_size)
    print("overlap=%d" % overlap)
    print('read data begin...')
    current_data = read_data_all(path)
    current_data.dropna(axis=0, how='any', inplace=True)
    # current_data = current_data.sample(frac=1)
    print('read data ok!')
    print('feature_normalize begin...')
    current_data['acc_x'] = feature_normalize(current_data['acc_x'])
    print('feature_normalize  acc_x ok!')
    current_data['acc_y'] = feature_normalize(current_data['acc_y'])
    print('feature_normalize  acc_y ok!')
    current_data['acc_z'] = feature_normalize(current_data['acc_z'])
    print('feature_normalize  acc_z ok!')
    current_data['gra_x'] = feature_normalize(current_data['gra_x'])
    print('feature_normalize  gra_x ok!')
    current_data['gra_y'] = feature_normalize(current_data['gra_y'])
    print('feature_normalize  gra_y ok!')
    current_data['gra_z'] = feature_normalize(current_data['gra_z'])
    print('feature_normalize  gra_z ok!')
    current_data['gyr_x'] = feature_normalize(current_data['gyr_x'])
    print('feature_normalize  gyr_x ok!')
    current_data['gyr_y'] = feature_normalize(current_data['gyr_y'])
    print('feature_normalize  gyr_y ok!')
    current_data['gyr_z'] = feature_normalize(current_data['gyr_z'])
    print('feature_normalize  gyr_z ok!')
    current_data['lacc_x'] = feature_normalize(current_data['lacc_x'])
    print('feature_normalize  lacc_x ok!')
    current_data['lacc_y'] = feature_normalize(current_data['lacc_y'])
    print('feature_normalize  lacc_y ok!')
    current_data['lacc_z'] = feature_normalize(current_data['lacc_z'])
    print('feature_normalize  lacc_z ok!')
    current_data['mag_x'] = feature_normalize(current_data['mag_x'])
    print('feature_normalize  mag_x ok!')
    current_data['mag_y'] = feature_normalize(current_data['mag_y'])
    print('feature_normalize  mag_y ok!')
    current_data['mag_z'] = feature_normalize(current_data['mag_z'])
    print('feature_normalize  mag_z ok!')
    current_data['ori_w'] = feature_normalize(current_data['ori_w'])
    print('feature_normalize  ori_w ok!')
    current_data['ori_x'] = feature_normalize(current_data['ori_x'])
    print('feature_normalize  ori_x ok!')
    current_data['ori_y'] = feature_normalize(current_data['ori_y'])
    print('feature_normalize  ori_y ok!')
    current_data['ori_z'] = feature_normalize(current_data['ori_z'])
    print('feature_normalize  ori_z ok!')
    current_data['pressure'] = feature_normalize(current_data['pressure'])
    print('feature_normalize  pressure ok!')
    print('feature_normalize ok!')
    print('segmentation begin...')
    segments, labels = segment_signal(current_data, _window_size, overlap)
    labels = to_categorical(labels, num_classes=8)
    print('segmentation ok!')
    print('train split begin...')
    train_split = np.random.rand(len(segments)) < train_split_ratio
    train_x = segments[train_split]
    train_y = labels[train_split]
    print('train split ok!')
    test_x = segments[~train_split]
    test_y = labels[~train_split]
    print('test split ok!')
    print('train saving begin...')
    np.savez("all_data_train_0.5_window_%d_overlap_%d_no_smooth.npz" % (_window_size, overlap), x=train_x, y=train_y)
    print('train saved ok!')
    print('test saving begin...')
    np.savez("all_data_test_0.5_window_%d_overlap_%d_no_smooth.npz" % (_window_size, overlap), x=test_x, y=test_y)
    print('test saved ok!')


if __name__ == '__main__':
    main()
