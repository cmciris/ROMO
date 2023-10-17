import numpy as np
import os
import torch
import pdb
from typing import Tuple

def init_data(dataset_dir='./disjoint_n6'):
    """
    bs:
    下行信道带宽
    参考信号功率
    天线挂高
    天线方向角
    电子下倾角
    机械下倾角
    最大发射功率


    training_data x / y:
    RRC连接最大数
    同频切换出成功次数
    PDCCH信道CCE占用率
    上行PRB平均利用率
    下行PRB平均利用率
    上行调度的最大UE数目
    下行调度的最大UE数目
    上行流量(MB)
    下行流量(MB)


    training_cio_data x:
    上行调度的最大UE数目
    下行调度的最大UE数目
    上行流量(MB)
    下行流量(MB)


    training_cio_data y:
    上行PRB平均利用率
    下行PRB平均利用率
    """

    data = {}
    for i, category in enumerate(['train', 'val', 'test']):
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        if i == 0:
            data['x'] = cat_data['x']
            data['y'] = cat_data['y']
            data['adj_mx'] = cat_data['adj_mx']
            data['cio'] = cat_data['cio']
            data['bs_info'] = cat_data['bs_info']
        else:
            data['x'] = np.concatenate((data['x'], cat_data['x']), axis=0)
            data['y'] = np.concatenate((data['y'], cat_data['y']), axis=0)
            data['adj_mx'] = np.concatenate((data['adj_mx'], cat_data['adj_mx']), axis=0)
            data['cio'] = np.concatenate((data['cio'], cat_data['cio']), axis=0)
            data['bs_info'] = np.concatenate((data['bs_info'], cat_data['bs_info']), axis=0)
         
        if i == len(['train', 'val', 'test']) - 1:
            scale = data['x'].shape[0]
            assert data['y'].shape[0] == data['adj_mx'].shape[0] == data['cio'].shape[0] == data['bs_info'].shape[0]
            data['x'] = data['x'].reshape((scale, -1))
            # data['y'] = data['y'].reshape((scale, -1))
            data['adj_mx'] = data['adj_mx'].reshape((scale, -1))
            data['cio'] = data['cio'].reshape((scale, -1))
            data['bs_info'] = data['bs_info'].reshape((scale, -1))
    # x: 20766, 40 * 4
    # y: 20766, 40, 2
    # adj_mx: 20766, 40 * 40
    # cio: 20766, 40 * 40
    # bs_info: 20766, 40 * 7
    
    new_data = {}
    new_data['x'] = np.concatenate((data['cio'], data['x'], data['adj_mx'], data['bs_info']), axis=1)
    # new_data['x'] = np.concatenate((data['cio'], data['x'], data['bs_info']), axis=1)
    new_data['aux'] = data['y'].reshape((scale, -1))
    new_data['y'] = - np.sum(np.var(data['y'], axis=-2), axis=-1, keepdims=True)
    # x: 20766, 40*40 + 40*4 + 40*40 + 40*7 = 3640
    # aux: 20766, 40*2
    # y: 20766, 1

    assert new_data['x'].shape[0] == new_data['aux'].shape[0] == new_data['y'].shape[0]
    return new_data
    

def main():
    data = init_data()
    print(data['x'].shape, data['aux'].shape, data['y'].shape)
    np.save('./x.npy', data['x'])
    np.save('./aux.npy', data['aux'])
    np.save('./y.npy', data['y'])

if __name__ == "__main__":
    main()