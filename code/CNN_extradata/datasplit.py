import os
import random
import shutil

dirct = '/workspace/statProject/Data/RawData_extra/reserved'
category = os.listdir(dirct)
category_dict = {}
for i in range(len(category)):
    category_dict[category[i]] = i

for l in category:
    os.makedirs(os.path.join('/workspace/statProject/Data/RawData_extra/train', l), exist_ok=True)
    os.makedirs(os.path.join('/workspace/statProject/Data/RawData_extra/validation', l), exist_ok=True)

def moveFile(fileDir):
    pathDir = os.listdir('/workspace/statProject/Data/RawData_extra/reserved/' + fileDir)  # 取图片的原始路径
    imagenum = len(pathDir)
    rate = 0.3  # 自定义抽取图片的比例
    picknum = int(imagenum * rate)  # 按照rate比例从文件夹中取一定数量图片
    valsample = random.sample(pathDir, picknum)  # 随机选取picknum数量的样本图片
    for i in range(len(valsample)):
        pathDir.remove(valsample[i])
    for name in valsample:
        shutil.copyfile('/workspace/statProject/Data/RawData_extra/reserved/' + fileDir + '/' + name, '/workspace/statProject/Data/RawData_extra/validation/' + fileDir + '/' + name)
    for name in pathDir:
        shutil.copyfile('/workspace/statProject/Data/RawData_extra/reserved/' + fileDir + '/' + name, '/workspace/statProject/Data/RawData_extra/train/' + fileDir + '/' + name)
    return


for i in range(len(category)):
    moveFile(category[i])
