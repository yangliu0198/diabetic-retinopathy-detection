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

rootDir = "/home/ubuntu/Yang_Sahana/train/"
# rootDir = "/Users/yangliu/Desktop/ga_new_cap/"
ls_file = []
for root, dirs, files in os.walk(rootDir):
    for fileName in files:
        if fileName.endswith('.jpeg'):
            ls_file.append(fileName)

            
def find_border_dynamic_threshold(image, axis):
    im_array = np.sum(np.sum(image, axis=axis), axis=1)
    threshold = im_array.max() /20
    indices = np.where(im_array > threshold)
    return indices[0][0], indices[0][-1]

def crop_img(img):
# finding min max indices
    min_x, max_x = find_border_dynamic_threshold(img, 0)
    min_y, max_y = find_border_dynamic_threshold(img, 1)
# crop
    image = img[min_y:max_y, min_x:max_x]
    return image


#reshape after crop
def reshape_img(filename, dimension): 
#     img = mpimg.imread('/Users/yangliu/Desktop/ga_new_cap/sample/' + filename)
#     img = mpimg.imread('/home/ubuntu/Yang_Sahana/train/' + filename) 
    image = cv2.resize(crop_img(mpimg.imread('../train/' + filename)),(dimension,dimension))
    return image


#create .data file for different sizes
def memap_arry(*args): #img_file,dimension
    #create memmap object
    #begin = time.time()
    #print "Started:", char
    img_file = args[0]
    dimension = args[1]
#     print dimentsion
    tmp_char_array = reshape_img('../train/' + img_file, dimension)
#     char_mmap = np.memmap(dtype='float64', filename= '/Users/yangliu/Desktop/ga_new_cap/' + str(dimension) + '_train/' + img_file.split('.')[0] +'.data', mode='w+', shape=tmp_char_array.shape)
#     char_mmap[:, :] = tmp_char_array
#     char_mmap_path = '/Users/yangliu/Desktop/ga_new_cap/'+ str(dimension) + '_train/' + img_file.split('.')[0] +'.data' #hard codes foler name
    char_mmap = np.memmap(dtype='uint8', filename= '/home/ubuntu/Yang_Sahana/' + str(dimension) + '_train/' + img_file.split('.')[0] +'.data', mode='w+', shape=tmp_char_array.shape)
    char_mmap[:, :] = tmp_char_array
    char_mmap_path = '/home/ubuntu/Yang_Sahana/'+ str(dimension) + '_train/' + img_file.split('.')[0] +'.data' #hard codes foler name

begin = time.time()
pool = Pool(multiprocessing.cpu_count())
pool.map(lambda a: memap_arry(a[0],a[1]),zip(ls_file,[512 for i in range(len(ls_file))]))#first 100
print time.time() - begin