import numpy as np
import pandas as pd
from dataset.load import DataLoader
from model.annoy import Annoy
import cv2 as cv

def main():
    d = DataLoader(train=False)
    target, label, origin = d.val_load()
    m = Annoy()
    pre_img, pre_label = m.pred(target)
    cv.imwrite('{}.jpg'.format(label), origin)
    for i, (p_im, p_l) in enumerate(zip(pre_img, pre_label)):
        cv.imwrite('pre_top{top}_{label}.jpg'.format(top=i+1, label=p_l), p_im)
    print('pred finish!')

if __name__ == '__main__':
    main()