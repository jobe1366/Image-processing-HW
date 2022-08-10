#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_histogram(original_img):
    f, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(original_img, 'gray', vmin=0, vmax=255)
    axes[1].hist(original_img.ravel() , bins=range(0,256), fc='k', ec='k')
    plt.tight_layout() 
    plt.xlabel('intensity value') 
    plt.ylabel('number of pixels') 
    plt.title('Histogram of the original image') 
    plt.show()    
    
def plotting(original_img , title):
    plt.imshow(original_img, 'gray', vmin=0, vmax=255) 
    plt.title('{}'.format(title)) 
    plt.show()
############################################################
img = cv2.imread('fig1.jpg',0)
plotting(img ,'Original Image')

##########################################
#global Thresholding


ret1,thresh1 = cv2.threshold(img,60,255,cv2.THRESH_BINARY)
# If Otsu thresholding is not used, retVal is same as the threshold value you used.

plotting(thresh1 , 'global_Threshold')
print('retVal for global Thresholding equal = {}'.format(ret1))

#########################################################
#OTSU

ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plotting(th2 , 'OTSU_Threshold')
print('retVal for OTSU equal = {}'.format(ret2))

map_hist=plot_histogram(img)

#########################################################
#Adaptive Thresholding

th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY,  11,  5)
plotting(th3 , 'ADAPTIVE_THRESH_MEAN_C')

th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY,  11,  -5)
plotting(th4 , 'ADAPTIVE_THRESH_GAUSSIAN_C')





