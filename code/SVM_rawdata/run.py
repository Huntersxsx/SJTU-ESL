from sklearn import datasets
from sklearn import svm
from prepare import *
from utils import *

imagData = []
labelData = []
train_loader, validation_loader, training_batches, validation_batches = load_data('F:/课程/统计学习/project-old/resnet18val/train', 'F:/课程/统计学习/project-old/resnet18val/validation', 1, get_transforms())
# print(len(train_loader))
for images, labels in train_loader:
    img = images.view(-1)
    imagData.append(img.tolist())
    labelData.append(labels.tolist())

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(imagData, labelData)

test_folder = 'F:/课程/统计学习/project-old/project/released_test'
imgs = os.listdir(test_folder)
imgs.sort(key=lambda x: int(x.split('.')[0]))
imgNum = len(imgs)
for i in range(imgNum):
    image = Img_Norm(test_folder + '/' +imgs[i])
    img = image.view(1, -1)
    out = clf.predict(img.tolist())
    result = [imgs[i][:-4], out[0]]
    resultfile = 'Result.csv'
    save_csvfile(result, resultfile)

