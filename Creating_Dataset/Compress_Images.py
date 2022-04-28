# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:29:59 2022

@author: vgari
"""
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from PIL import Image

### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#%matplotlib inline

dataDir='C:/Users/vgari/OneDrive/Documents/University/Winter 2022/IFT 6289/Project/COCOdataset2017'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# Initialize the COCO api for instance annotations
coco=COCO(annFile)

# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

#print(cats)

# Define the classes (out of the 81) which you want to see. Others will not be shown.
filterClasses = ['airplane']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses) 
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))

# initialize COCO API for caption annotations
captions_annFile = '{}/annotations/captions_train2017.json'.format(dataDir)
coco_caps = COCO(captions_annFile)


annIds = coco_caps.getAnnIds(imgIds=81)
print(annIds)
anns = coco_caps.loadAnns(annIds)

print(anns)

#Loop to save images to new folder
for i in range(2000,len(imgIds)):
  img = coco.loadImgs(imgIds[i])[0]
  foo = Image.open("C:\\Users\\vgari\\OneDrive\\Documents\\University\\Winter 2022\\IFT 6289\\Project\\New_Images_Airplanes\\content\\New_Art_Airplanes2\\{}".format(img['file_name']))
  foo.save("C:\\Users\\vgari\\OneDrive\\Documents\\University\\Winter 2022\\IFT 6289\\Project\\New_Images_Airplanes\\New_Art_Airplanes_Final\\{}".format(img['file_name']),optimize=True,quality=85)
  print(i)
