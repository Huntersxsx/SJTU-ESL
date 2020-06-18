import torch
import logging
import pandas as pd
from scipy.misc import imread, imresize
import numpy as np
from torchvision import transforms, datasets

logging.basicConfig(level=logging.INFO,
                    filename='/workspace/statProject/CNN_moredata/train_state.log',
                    filemode='a',
                    format='%(message)s'
                    )


def save_checkpoint(epoch, model):
    state = {'epoch': epoch,
             'model': model}
    filename = '/workspace/statProject/CNN_moredata/' + 'Best_checkpoint' + '.pth.tar'
    torch.save(state, filename)


def save_csvfile(my_list, name):
    df = pd.DataFrame(data=[my_list])
    df.to_csv("/workspace/statProject/CNN_moredata/" + name, encoding="utf-8-sig", mode="a", header=False, index=False)


def Img_Norm(image_path):
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (28, 28))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img)
    normalize = transforms.Normalize([0.164597, 0.164597, 0.164597], [0.324155, 0.324155, 0.324155])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 28, 28)
    image = image.unsqueeze(0)
    return image


def adjust_learning_rate(optimizer, shrink_factor):
    # print("\nDECAYING learning rate.")
    logging.info("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    # print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
    logging.info("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
