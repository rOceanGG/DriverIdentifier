import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import shutil
import os



def getFaces():
    def getCroppedImageIfTwoEyes(imPath):
        img = cv2.imread(imPath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3,5)
        for(x,y,w,h) in faces:
            regionOfInterestGray = gray[y:y+h, x:x+w]
            regionOfInterestColour = img[y:y+h, x:x+w]
            eyes = eyeCascade.detectMultiScale(regionOfInterestGray)
            if len(eyes) >= 2:
                return regionOfInterestColour
            
    #This will be used to identify faces in pictures
    faceCascade = cv2.CascadeClassifier('./opencv/haarcascade/haarcascade_frontalface_default.xml')
    #This will be used to identify eyes in pictures
    eyeCascade = cv2.CascadeClassifier('./opencv/haarcascade/haarcascade_eye.xml')
    #This image was selected manually to ensure it had visible eyes and face
    img = cv2.imread('./test_images/0_Formula-1-Abu-Dhabi-test-2023-23332300449330.jpg')
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #File paths for creating new directories
    path_to_data = "./dataset/"
    path_to_cropped_data = "./dataset/cropped/"
    
    img_dirs = []
    for entry in os.scandir(path_to_data):
        if entry.is_dir():
            img_dirs.append(entry.path)
    
    if os.path.exists(path_to_cropped_data):
        shutil.rmtree(path_to_cropped_data)
    
    croppedDirectories = []
    driverFileNamesDict = {}

    for img_dir in img_dirs:
        count = 1
        driver = img_dir.split('/')[-1]
        driverFileNamesDict[driver] = []
        croppedFolder = path_to_cropped_data + driver
        if not os.path.exists(croppedFolder):
            os.makedirs(croppedFolder)
        croppedDirectories.append(croppedFolder)
        print("Generating cropped images in:", croppedFolder)
        for entry in os.scandir(img_dir):
            try:
                crp = getCroppedImageIfTwoEyes(entry.path)
            except:
                crp = None
            if crp is not None:
                newFileName = driver + str(count) + ".png"
                newFilePath = croppedFolder + "/" + newFileName

                cv2.imwrite(newFilePath, crp)
                count += 1  
                driverFileNamesDict[driver].append(newFilePath)
    

