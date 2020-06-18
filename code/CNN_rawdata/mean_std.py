import os
import numpy as np
from scipy.misc import imread

filepath = '/workspace/statProject/Data/RawData/train'  # 数据集目录
category = os.listdir(filepath)

R_channel = 0

for jj in range(len(category)):
    imdir = os.listdir(filepath + '/' +category[jj])
    for idx in range(len(imdir)):
        imgname = imdir[idx]
        img = imread(os.path.join(filepath + '/' + category[jj], imgname))
        # print(np.shape(img))
        R_channel = R_channel + np.sum(img[:, :]) / 255


num = 15 * 400 * 28 * 28  
R_mean = R_channel / num

R_channel = 0

for jj in range(len(category)):
    imdir = os.listdir(filepath + '/' +category[jj])
    for idx in range(len(imdir)):
        imgname = imdir[idx]
        img = imread(os.path.join(filepath + '/' + category[jj], imgname))
        R_channel = R_channel + np.sum((img[:, :] / 255 - R_mean) ** 2)


R_var = R_channel / num

print("R_mean is %f" % (R_mean))
print("R_std is %f" % (np.sqrt(R_var)))

