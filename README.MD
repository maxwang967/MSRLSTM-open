# MSRLSTM Open Source

## Notice

This repo is based on old version of Keras DL Framework, I highly recommend anyone who refers this repo to use TensorFlow 2.x and most of the APIs should go under tf.keras. I have recenly referred this repo to conduct new experiments by this way and I think it will be helpful for you too. If you really need a sample code, please feel free to contact me at max.chenxing.wang (at) gmail.com.

## Guidelines

MSRLSTM is a deep learning model for transportation mode detection. To successfully run this code, there are several works needed to be done at first:
- Download SHL Dataset from http://www.shl-dataset.org/activity-recognition-challenge/
- Sort the sampling data according to the given order file and merge data into Label_1.txt to Label_8.txt. The columns should be 'timestamp',
                    'acc_x', 'acc_y', 'acc_z',
                    'gra_x', 'gra_y', 'gra_z',
                    'gyr_x', 'gyr_y', 'gyr_z',
                    'lacc_x', 'lacc_y', 'lacc_z',
                    'mag_x', 'mag_y', 'mag_z',
                    'ori_w', 'ori_x', 'ori_y', 'ori_z',
                    'pressure', 'label'. And the 'label' column is ranged from 1 to 8, which represents 1 – Still; 2 – Walk; 3 – Run; 4 – Bike; 5 – Car; 6 – Bus; 7 – Train; 8 – Subway.
- Modify the ./utils/config.yaml configuration file.
- Run data saver by:
```shell script
python run.py --config=/public/lhy/wms/MSRLSTM/utils/config.yaml --mode=data_preprocess
```
- Run trainer by:
```shell script
python run.py --config=/public/lhy/wms/MSRLSTM/utils/config.yaml --mode=train
```
- Run tester by:
```shell script
python run.py --config=/public/lhy/wms/MSRLSTM/utils/config.yaml --mode=test
```
## Updating Status

- 2019-10-9: Release the initial version for public use.

## Application Status

The MSRLSTM model is now running on https://github.com/morningstarwang/TMDMobileNG with our cloud server. We are still testing and optimizing our MSRLSTM to behave more functionality in real world.

## Notice

This open source code is specially refracted for more human friendly study use. Other researchers may find bugs because some of the codes are untested. Our researchers are using more complex codes for our research and more codes will be released if any milestone is achieved.  

## Cite

```
@article{9078348,
 abstract = {In recent years, with the rapid development of public transportation, the ways people travel has become more diversified and complicated. Transportation mode detection, as a significant branch of human activity recognition (HAR), is of great importance in analyzing human travel patterns, traffic prediction and planning. Though many works have been devoted to transportation mode detection, there remains challenge for accurate and robust transportation pattern identification. In this paper, we propose a residual and LSTM recurrent networks-based transportation mode detection algorithm using multiple light-weight sensors integrated in commodity smartphones. Feature representation learning is adopted separately on multiple preprocessed sensor data using deep residual and LSTM network, which can enhance the identification accuracy and support one or more sensors. Residual units are introduced to accelerate the learning speed and enhance the accuracy of transportation mode detection. Furthermore, we also leverage the attention model to learn the significance of different features and different timesteps to enhance the recognition accuracy. Extensive experimental results on three datasets indicate that using our proposed model can achieve the best recognition accuracy for eight transportation modes including being stationary, walking, running, cycling, taking a car, taking a bus, taking a subway and taking a train, which outperforms other benchmark algorithms.},
 author = {C. Wang and H. Luo and F. Zhao and Y. Qin},
 doi = {10.1109/TITS.2020.2987598},
 issn = {1558-0016},
 journal = {IEEE Transactions on Intelligent Transportation Systems},
 keywords = {Context awareness;transportation pattern recognition;multimodal sensors;activity recognition.},
 month = {},
 number = {},
 pages = {1-13},
 title = {Combining Residual and LSTM Recurrent Networks for Transportation Mode Detection Using Multimodal Sensors Integrated in Smartphones},
 volume = {},
 year = {2020}
}

``` 
