# MSRLSTM Open Source

## Notice

This repo has been recently updated to support tf 2.x without comprehensive tested, please feel free to contact me at max.chenxing.wang (at) gmail.com if bugs appear.

We provide the HTC sample dataset at: https://drive.google.com/drive/folders/1pDdGNzo02vFl9Nyz4HQWd0UyvTy_tw1o?usp=sharing.

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
@ARTICLE{9078348,  author={Wang, Chenxing and Luo, Haiyong and Zhao, Fang and Qin, Yanjun},  journal={IEEE Transactions on Intelligent Transportation Systems},   title={Combining Residual and LSTM Recurrent Networks for Transportation Mode Detection Using Multimodal Sensors Integrated in Smartphones},   year={2021},  volume={22},  number={9},  pages={5473-5485},  doi={10.1109/TITS.2020.2987598}}

``` 
