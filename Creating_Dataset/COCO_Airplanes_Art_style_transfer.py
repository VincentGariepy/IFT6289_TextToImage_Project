from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import json
import shutil

### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#%matplotlib inline

import paddle
import os
import sys

sys.path.insert(0, os.getcwd())
from ppgan.apps import LapStylePredictor
import argparse

random.seed(1)

if __name__ == "__main__":
    def append_to_json(_dict,path): 
          with open(path, 'ab+') as f:
              f.seek(0,2)                                #Go to the end of file    
              if f.tell() == 0 :                         #Check if file is empty
                  f.write(json.dumps([_dict]).encode())  #If empty, write an array
              else :
                  f.seek(-2,2)           
                  f.truncate()                           #Remove the last character, open the array
                  f.write(','.encode())                #Write the separator
                  f.write(json.dumps(_dict).encode())    #Dump the dictionary
                  f.write(']}'.encode())                  #Close the array

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_path",
                        type=str,
                        required=True,
                        help="path to content image")

    parser.add_argument("--output_path",
                        type=str,
                        default='output_dir',
                        help="path to output image dir")

    parser.add_argument("--weight_path",
                        type=str,
                        default=None,
                        help="path to model weight path")

    parser.add_argument(
        "--style",
        type=str,
        default='starrynew',
        help=
        "if weight_path is None, style can be chosen in 'starrynew', 'circuit', 'ocean' and 'stars'"
    )

    parser.add_argument("--style_image_path",
                        type=str,
                        required=True,
                        help="path to style image")

    parser.add_argument("--cpu",
                        dest="cpu",
                        action="store_true",
                        help="cpu mode.")

    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    dataDir='COCOdataset2017'
    dataType='train2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    #Styles
    artstyles=['circuit','ocean','starrynew','stars']
    # Initialize the COCO api for instance annotations
    coco=COCO(annFile)

    # Load the categories in a variable
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)

    # Define the classes (out of the 81) which you want to see. Others will not be shown.
    filterClasses = ['airplane']

    # Fetch class IDs only corresponding to the filterClasses
    catIds = coco.getCatIds(catNms=filterClasses) 
    # Get all images containing the above Category IDs
    imgIds = coco.getImgIds(catIds=catIds)

    # initialize COCO API for caption annotations
    captions_annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps = COCO(captions_annFile)

    for i in range(1):
      #Get random style
      randomstyle=random.randint(0,3)
      #Get Caption
      annIds = coco_caps.getAnnIds(imgIds=imgIds[i])
      anns = coco_caps.loadAnns(annIds)
      for j in range(len(anns)):
        temp=anns[j]['caption'][:-1]+' as '+artstyles[randomstyle]+'.' 
        annotation = {'image_id': imgIds[i],'id': annIds[j],'caption': temp}
        append_to_json(annotation,'./captions_val2017_planes2.json')

      #Get file paths
      img=coco.loadImgs(imgIds[i])[0]
      content_img_path='../New_images_Airplanes/New_Images_Airplanes/{}'.format(img['file_name'])
      output_path='../Output/'
      style=artstyles[randomstyle]
      style_image_path='../Style_Images/{}.png'.format(style)

      #Transfer Art Style
      predictor = LapStylePredictor(output=output_path,
                                  style=style,
                                  weight_path=args.weight_path)
      predictor.run(content_img_path, style_image_path)

      #Move new image
      shutil.move('../Output/LapStyle/stylized.png', '../New_Art_Airplanes/{}'.format(img['file_name']))

      print(i)
