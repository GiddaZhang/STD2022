# get_calibrate.py
# 运行后生成验证集
# 可能需要在根目录创建Calibrate文件夹


import numpy as np
import os

root_1 = 'Train'
root_2 = 'Calibrate'
vpath = os.path.join(root_1, 'vfeat')
apath = os.path.join(root_1, 'afeat')
vpath_ = os.path.join(root_2, 'vfeat')
apath_ = os.path.join(root_2, 'afeat')
for index in range(500):
    vfeat = np.load(os.path.join(vpath, '%04d.npy'%(index))).astype('float32')
    afeat = np.load(os.path.join(apath, '%04d.npy'%(index))).astype('float32')
    np.save(os.path.join(vpath_, '%04d.npy'%(index)), vfeat)
    np.save(os.path.join(apath_, '%04d.npy'%(index)), afeat)