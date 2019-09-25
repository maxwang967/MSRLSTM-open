# -*- coding: utf-8 -*-
# @Time    : 2018-08-15 9:24
# @Author  : morningstarwang
# @FileName: merge_data_1_window_size.py
# @Blog    ：https://morningstarwang.github.io
import joblib
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

__author__ = 'morningstarwang'

import numpy as np
import numpy.linalg as la
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv1D, concatenate, Flatten, MaxPooling1D, Dropout, PReLU, BatchNormalization, \
    LSTM

import pandas as pd
train_data = pd.read_csv("/public/lhy/data/ict_feature/Label_Train.csv", sep=",")
# train_data = train_data[train_data['label'] != 'label']
train_data.fillna(0, inplace=True)

test_data = pd.read_csv("/public/lhy/data/ict_feature/Label_Test.csv", sep=",")
# test_data = test_data[test_data['label'] != 'label']
test_data.fillna(0, inplace=True)

X_train = train_data.iloc[:, :238].values
y_train = train_data.iloc[:, 238:].values
y_train = np.reshape(y_train, (y_train.shape[0], ))
y_train = [int(x) - 1 for x in y_train]

X_test = test_data.iloc[:, :238].values
y_test = test_data.iloc[:, 238:].values
y_test = [int(x) - 1 for x in y_test]


classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                       oob_score=False, random_state=42, verbose=0, warm_start=False)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
reversefactor = dict(zip(range(6), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
joblib.dump(classifier, 'ict_rf.pkl')
print("end")
