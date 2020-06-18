import os
from PIL import Image
from scipy.misc import imread, imresize
from prepare import *
import torch.nn as nn
import numpy as np
from model import *

dirct = '/workspace/statProject/Data/RawData/extra_training_data'
model_path = '/workspace/statProject/CNN_rawdata/Best_checkpoint.pth.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
category = os.listdir(dirct)
category_dict = {}
for i in range(len(category)):
    category_dict[category[i]] = i

checkpoint = torch.load(model_path, map_location='cpu')

model = checkpoint['model']
# model = model.to(device)
model.eval()
criterion = nn.CrossEntropyLoss()


for l in category:
    os.makedirs(os.path.join('/workspace/statProject/Data/RawData_extra/total', l), exist_ok=True)
    os.makedirs(os.path.join('/workspace/statProject/Data/RawData_extra/reserved', l), exist_ok=True)


def moveFile(fileDir):
    pathDir = os.listdir('/workspace/statProject/Data/RawData/extra_training_data/' + fileDir)  # 取图片的原始路径
    for name in pathDir:
        newname = name
        newname= newname.split('.')
        imgpath = '/workspace/statProject/Data/RawData/extra_training_data/' + fileDir + '/' + name

        if (newname[-1] == 'jpg') or (newname[-1] == 'JPG') or (newname[-1] == 'jpeg') or (newname[-1] == 'JPEG') or (newname[-1] == 'PNG'):
            newname[0] = newname[0] + 'p'
            newname[-1] = 'png'
            newname = '.'.join(newname)
            img = imread(imgpath)
            if (img.shape == (28, 28)) or (img.shape == (28, 28, 3)):
                im = Image.open(imgpath)
                out = im.resize((28, 28))
                out.save('/workspace/statProject/Data/RawData_extra/total/' + fileDir + '/' + newname)

        elif newname[-1] == 'png':
            img = imread(imgpath)
            if (img.shape == (28, 28)) or (img.shape == (28, 28, 3)):
                im = Image.open(imgpath)
                out = im.resize((28, 28))
                out.save('/workspace/statProject/Data/RawData_extra/total/' + fileDir + '/' + name)

    return


def Cal_Loss():
    Loss = []
    train_loader, validation_loader, training_batches, validation_batches = load_data('/workspace/statProject/Data/RawData_extra/total',
                                                                                      '/workspace/statProject/Data/RawData_with_val/validation',
                                                                                      1, get_transforms())
    # print(len(train_loader))
    for images, labels in train_loader:
        # print(images.size())
        # print(labels)
        predicts = model.forward(images)
        loss = criterion(predicts, labels)
        Loss.append(loss)

    print('Before Data clean: {}'.format(len(Loss)))
    Loss.sort()
    # print(Loss[int(len(Loss)*0.7)])
    theta = Loss[int(len(Loss) * 0.8)].item()
    return theta


def DataClean(fileDir, theta):
    img_count = 0
    pathDir = os.listdir('/workspace/statProject/Data/RawData_extra/total/' + fileDir)  # 取图片的原始路径
    label = [category_dict[fileDir]]
    label = np.array(label)
    label = torch.from_numpy(label)
    label = label.type(torch.LongTensor)
    # print(label.size())
    for name in pathDir:
        imgpath = '/workspace/statProject/Data/RawData_extra/total/' + fileDir + '/' + name
        img = imread(imgpath)
        # print(img.shape)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = img.transpose(2, 0, 1)
        L = img.tolist()
        img2 = np.array(L)
        img2 = img2 /255.0
        img2 = torch.from_numpy(img2).unsqueeze(0)
        img2 = img2.type(torch.FloatTensor)
        # print(img2.size())
        predicts = model.forward(img2)
        loss = criterion(predicts, label)
        #print(loss.item())
        if loss.item() < theta:
            im = Image.open(imgpath)
            im.save('/workspace/statProject/Data/RawData_extra/reserved/' + fileDir + '/' + name)
            img_count += 1
    return img_count


def moveimage(fileDir):
    pathDir = os.listdir('/workspace/statProject/Data/RawData/train/' + fileDir)  # 取图片的原始路径
    for name in pathDir:
        newname = name
        newname= newname.split('.')
        newname[0] = newname[0] + 'raw'
        newname = '.'.join(newname)
        imgpath = '/workspace/statProject/Data/RawData/train/' + fileDir + '/' + name
        im = Image.open(imgpath)
        im.save('/workspace/statProject/Data/RawData_extra/reserved/' + fileDir + '/' + newname)




'''
# 仅保留28*28和28*28*3的图片
for i in range(len(category)):
    moveFile(category[i])
'''

'''

# 计算loss阈值
theta = Cal_Loss()
IM_COUNT = []
# 保留loss较小的80%的图片
for i in range(len(category)):
    imgcnt = DataClean(category[i], theta)
    IM_COUNT.append(imgcnt)

print('After Data clean: {}'.format(sum(IM_COUNT)))

# 将原始数据也移入到新文件夹中
for i in range(len(category)):
    moveimage(category[i])
'''

