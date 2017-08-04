import os
import pandas as pd
import numpy as np
#from medpy.io import load
from matplotlib import pyplot as plt
import glob
import scipy
import cv2
import random
from scipy.misc import imread
import matplotlib.image as mpimg
#from multiprocessing import Pool
from multiprocess import Pool
import time
import multiprocessing


rootDir = '/home/ubuntu/Yang_Sahana/256_train/'
image = []
img_arr = []
for fname in os.listdir(rootDir):
    if fname.endswith('.data'):
        image.append(fname.split('.')[0])
        img_arr.append(np.fromfile('/home/ubuntu/Yang_Sahana/256_train/' + fname, dtype='uint8').reshape(256,256,3))#, shape = (128,128,3)

image_df = pd.DataFrame({'image': image, 'img_arr': img_arr})
labels = pd.read_csv('/home/ubuntu/Yang_Sahana/trainLabels.csv')
result = pd.merge(image_df, labels, on = ['image'])
label = pd.get_dummies(result.level)
result1 = result.join(label)


def resample(num_samples, level):
    if level == 0 or level == 1 or level == 2:
        l = result1[result1['level'] == level].index
        rnd_idx = random.sample(l, num_samples)
        return result1.iloc[rnd_idx,:]
    else:
        n = len(result1[result1['level'] == level])
        l = result1[result1['level'] == level].index
        idx = np.random.choice(l, num_samples, n)
        return result1.iloc[idx,:]

ls = []
for i in range(5):
    ls.append(resample(2000, i))

resampled_df = pd.concat(ls)
resampled_x_train = np.concatenate([arr[np.newaxis] for arr in resampled_df.img_arr])
resampled_y_train = resampled_df.iloc[:,3:].values

from resnet import resnet_v1
model = resnet_v1()
model.fit(resampled_x_train, resampled_y_train, validation_split=0.1, epochs=10, batch_size=10)