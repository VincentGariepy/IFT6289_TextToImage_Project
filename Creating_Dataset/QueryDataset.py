from pycocotools.coco import COCO

from PIL import Image

#Path au folder COCOdataset2017
dataDir='C:/Users/vgari/OneDrive/Documents/University/Winter 2022/IFT 6289/Project/COCOdataset2017Airplanes'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

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

# initialize COCO API for caption captions
captions_annFile = '{}/annotations/captions_{}_planes.json'.format(dataDir,dataType)
coco_caps = COCO(captions_annFile)

####### IMPORTANT ########
#On peut loop à travers les images avec imgIds[i], il y a 2986 images donc i peut être de 0 à 2985.
#Notre training set est de 0 à 1999
#Notre validation set est de 2000 à 2585
#Notre test set est de 2586 à 2985
##########################

#Get Caption
annIds = coco_caps.getAnnIds(imgIds=imgIds[2985])
anns = coco_caps.loadAnns(annIds)
#Il y a 5 captions au total, pour le test on peut simplement utiliser la premiere
print(anns[0]['caption']+'\n'+anns[1]['caption']+'\n'+anns[2]['caption']+'\n'+anns[3]['caption']+'\n'+anns[4]['caption'])

#Get file paths
img=coco.loadImgs(imgIds[2985])[0]
content_img_path='{}/New_Art_Airplanes_Final/{}'.format(dataDir,img['file_name'])
#save picture
picture = Image.open(content_img_path)  
#display picture
picture.show()