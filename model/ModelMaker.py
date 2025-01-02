import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import shutil
import os
import pywt

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
        
        # The following 3 lines simply find which directories to scan through when processing each image
        ImageDirectories = []
        for entry in os.scandir(self.DATAPATH):
            if entry.is_dir():
                ImageDirectories.append(entry.path)
        
        # This ensures that we have an empty folder which is ready to be populated with cropped images
        if os.path.exists(self.CROPPEDDATAPATH):
            shutil.rmtree(self.CROPPEDDATAPATH)
        
        # As the names suggest, these hold both the directories with cropped images
        # and a way to check which directory holds the images for each driver
        CroppedImageDirectories = []
        DriverFileNamesDictionary = {}

        for imageDirectory in ImageDirectories:
            # When creating the new file names, we need to ensure that each one is unique.
            # The count will be used for that.
            count = 1

            # When populating my dataset, the folder that contains the images of each driver, is named after the driver itself. 
            # Hence, the name of the driver can be found by grabbing the folder name
            driver = imageDirectory.split('/')[-1]
            DriverFileNamesDictionary[driver] = []
            croppedFolder = self.CROPPEDDATAPATH + driver

            # Creates the folder to hold the cropped images
            if not os.path.exists(croppedFolder):
                os.makedirs(croppedFolder)
            
            CroppedImageDirectories.append(croppedFolder)

            for entry in os.scandir(imageDirectory):
                try:
                    croppedImage = getCroppedImageWithTwoEyes(entry.path)
                except:
                    croppedImage = None
                
                # If a face with two eyes is found, it's cropped and saved as a new image in the respective folder.
                if croppedImage:
                    # Creates a unique file name via the usage of the count variable.
                    newFileName = driver + str(count) + ".png"
                    # New file path will have to be in the directory with all new cropped images for the respective driver.
                    newFilePath = croppedFolder + "/" + newFileName

                    # Saves the cropped image in the new folder
                    cv2.imwrite(newFilePath, croppedImage)
                    # Adds the new file name to the list of file names for the current driver
                    DriverFileNamesDictionary[driver].append(newFileName)
                    count += 1

    def waveletTransform(image, mode='haar', level = 1):
        imageArray = image

        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)

        imageArray = np.float32(imageArray)
        imageArray /= 255
    
        coefficients = pywt.wavedec2(imageArray, mode, level=level)
    
        coefficientsHaar=list(coefficients)
        coefficientsHaar[0] *= 0
    
        imageArrayHaar=pywt.waverec2(coefficientsHaar, mode)
        imageArrayHaar *= 255
        imageArrayHaar = np.uint8(imageArrayHaar)
        
        return imageArrayHaar