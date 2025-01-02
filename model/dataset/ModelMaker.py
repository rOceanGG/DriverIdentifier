import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import shutil
import os


class ModelMaker:
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier('DriverIdentifier/model/opencv/haarcascade/haarcascade_frontalface_default.xml')
        self.eyeCascade  = cv2.CascadeClassifier('DriverIdentifier/model/opencv/haarcascade/haarcascade_eye.xml')
        self.DATAPATH = "./model/dataset/"
        self.CROPPEDDATAPATH = "./model/dataset/cropped/"

    def getFaces(self):
        #This function will be used on individual images to grab the faces of the drivers
        def getCroppedImageWithTwoEyes(imagePath):
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3,5)
            for(x,y,w,h) in faces:
                regionOfInterestGray = gray[y:y+h, x:x+w]
                regionOfInterestColour = img[y:y+h, x:x+w]
                eyes = self.eyeCascade.detectMultiScale(regionOfInterestGray)
                if len(eyes) >= 2:
                    return regionOfInterestColour
        
        ImageDirectories = []
        for entry in os.scandir(self.DATAPATH):
            if entry.is_dir():
                ImageDirectories.append(entry.path)
        
        if os.path.exists(self.CROPPEDDATAPATH):
            shutil.rmtree(self.CROPPEDDATAPATH)
        
        CroppedImageDirectories = []
        DriverFileNamesDictionary = {}

        for imageDirectory in ImageDirectories:
            count = 1
            driver = imageDirectory.split('/')[-1]
            DriverFileNamesDictionary[driver] = []
            croppedFolder = self.CROPPEDDATAPATH + driver
            if not os.path.exists(croppedFolder):
                os.makedirs(croppedFolder)
            
            CroppedImageDirectories.append(croppedFolder)

            for entry in os.scandir(imageDirectory):
                try:
                    croppedImage = getCroppedImageWithTwoEyes(entry.path)
                except:
                    croppedImage = None
                
                if croppedImage:
                    newFileName = driver + str(count) + ".png"
                    newFilePath = croppedFolder + "/" + newFileName

                    cv2.imwrite(newFilePath, croppedImage)
                    DriverFileNamesDictionary[driver].append(newFileName)
                    count += 1