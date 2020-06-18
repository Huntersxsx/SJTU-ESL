from utils import *
from model import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = '/workspace/statProject/Resnet101/Best_checkpoint.pth.tar'
test_folder = '/workspace/statProject/Data/RawData/released_test'
checkpoint = torch.load(model_path, map_location='cpu')

model = checkpoint['model']
model = model.to(device)
model.eval()

imgs = os.listdir(test_folder)
imgs.sort(key=lambda x: int(x.split('.')[0]))
imgNum = len(imgs)
for i in range(imgNum):
    image = Img_Norm(test_folder + '/' +imgs[i])
    image = image.to(device)
    out = model.forward(image)
    top_prob, top_class = out.topk(1, dim=1)
    result = [imgs[i][:-4], str(top_class.item())]
    resultfile = 'Result.csv'
    save_csvfile(result, resultfile)



