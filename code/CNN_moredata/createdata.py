from PIL import ImageChops
from PIL import Image
import os
import numpy as np
import random
import shutil


def move(root_path, img_name, off): #平移，平移尺度为off
    img = Image.open(os.path.join(root_path, img_name))
    offset = ImageChops.offset(img, off, off)
    return offset


def flip(root_path, img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def rotation(root_path, img_name, angle):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(angle) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img


def no_change(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    return img


filepath = '/workspace/statProject/Data/RawData_extra/reserved'  # 数据集目录
category = os.listdir(filepath)
for l in category:
    os.makedirs(os.path.join('/workspace/statProject/Data/MoreData/total', l), exist_ok=True)
saveDir = "/workspace/statProject/Data/MoreData/total/"   #要保存的图片的路径文件夹
#style = 'no_change'
#style = 'y_3'
#style = 'x_2'
#style = 'x1_y1'
#style = 'rotate12'
#style = 'rotatefu12'
#style = 'flip'
#style = 'yfu1'
#style = 'xfu1'
#style = 'x1_rotate8'
#style = 'y1_rotatefu8'
#style = 'rotate8_flip'
#style = 'xfu1_y1'
#style = 'x1_yfu1_rotate6'
style = 'xfu1_yfu1_flip'
for jj in range(len(category)):
    imdir = os.listdir(filepath + '/' + category[jj])
    for idx in range(len(imdir)):
        imgname = imdir[idx]
        saveName = category[jj] + str(idx) + style + '.png'
        temImage = move(filepath + '/' + category[jj] + '/', imgname, -1)
        saveImage = temImage.transpose(Image.FLIP_LEFT_RIGHT)
        #saveImage = no_change(filepath + '/' + category[jj] + '/', imgname,)
        #saveImage = move(filepath + '/' + category[jj] + '/', imgname, 1)
        #saveImage = rotation(filepath + '/' + category[jj] + '/', imgname, -12)
        #saveImage = flip(filepath + '/' + category[jj] + '/', imgname)
        saveImage.save(os.path.join(saveDir + category[jj] + '/', saveName))




