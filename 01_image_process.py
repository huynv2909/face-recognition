# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 03:37:05 2019

@author: huynv
"""

# import the necessary packages
from imutils import paths
import cv2
import os

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

sourcePath = 'dataset-temp'
desPath = 'dataset'

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(sourcePath))

count = 0;

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image " + imagePath + " {}/{}:".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    id = name.split("_")[0]
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if type(faces) is tuple:
        print("----->>> Can not detect face")
    else:
        count += 1
        cv2.rectangle(img, (faces[0][0],faces[0][1]), (faces[0][0]+faces[0][2],faces[0][1]+faces[0][3]), (255,0,0), 2)
        cv2.imwrite(desPath + "/User." + str(id) + '.' + str(count) + ".jpg", gray[faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2]])
        print("----->>> Ok")
        cv2.imshow('image', img)
        
cv2.destroyAllWindows()