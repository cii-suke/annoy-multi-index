from keras.datasets import fashion_mnist
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import DataLoader

def extract_edge(imgs):
    _imgs = np.zeros_like(imgs)
    for i, img in enumerate(imgs):
        _imgs[i] = cv.Canny(img, 50, 110)
    return _imgs

class DataLoader(DataLoader):

    def __init__(self, train=True):
        (t_image, t_label), (v_image, v_label) = fashion_mnist.load_data()

        if train:
            self.x_ori = t_image
            self.x = extract_edge(t_image)
            self.y = t_label
        else:
            self.x_ori = v_image
            self.x = extract_edge(v_image)
            self.y = v_label        
      

    def load(self):
        return self.x, self.y, self.x_ori

    def val_load(self):
        dice = np.arange(len(self.y))
        index = np.random.choice(dice)
        return self.x[index], self.y[index], self.x_ori[index]

    def __len__(self):
        return len(self.x)

