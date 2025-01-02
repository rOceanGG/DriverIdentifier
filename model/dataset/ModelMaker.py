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
        self.DATAPATH = "./model/dataset"
        self.CROPPEDDATAPATH = "./model/dataset/cropped"

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