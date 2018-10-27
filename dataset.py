import os

import numpy
from PIL import Image
import six

import numpy as np

from io import BytesIO
import os
import pickle
import json
import numpy as np

import skimage.io as io

from chainer.dataset import dataset_mixin

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='./azuren', data_range=(1,2000), size=286):
        # print("load dataset start")
        # print("    from: %s"%dataDir)
        # print("    range: [%d, %d)"%(data_range[0], data_range[1]))
        self.dataDir = dataDir
        self.dataset = []
        for i in range(data_range[0],data_range[1]):
            img_path = dataDir+"/cmp_c%04d.jpg" % i
            label_path = dataDir+"/cmp_l%04d.jpg" % i
            self.dataset.append((img_path, label_path))
        self.size = size
        #print("load dataset done")
    
    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i, crop_width=256):

        img = Image.open(self.dataset[i][0])
        label = Image.open(self.dataset[i][1])

        # # resize
        w,h = img.size
        # r = self.size / float(min(w,h))
        # img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
        # label = label.resize((int(r*w), int(r*h)), Image.NEAREST)
        
        img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
        label = label.convert("L") # グレースケール(1ch)に変更
        label = np.asarray(label).astype("f")[np.newaxis, :, :]/128.0-1.0
        
        #_,h,w = self.dataset[i][0].shape
        x_l = np.random.randint(0,w-crop_width)
        x_r = x_l+crop_width
        y_l = np.random.randint(0,h-crop_width)
        y_r = y_l+crop_width
        return label[:,y_l:y_r,x_l:x_r], img[:,y_l:y_r,x_l:x_r]
            
    
