import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import shutil
import os
import pywt
import pandas as pd
import joblib
import sklearn
import json
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

class ModelMaker:
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier('./model/opencv/haarcascade/haarcascade_frontalface_default.xml')
        self.eyeCascade  = cv2.CascadeClassifier('./model/opencv/haarcascade/haarcascade_eye.xml')
        self.DATAPATH = "./model/dataset/"
        self.CROPPEDDATAPATH = "./model/dataset/cropped/"
        self.classificationDictionary = {}
        self.CroppedImageDirectories = []
        self.DriverFileNamesDictionary = {}
        self.IMAGES = []
        self.NAMES = []
        self.IMAGESTRAIN = None
        self.IMAGESTEST = None
        self.NAMESTRAIN = None
        self.NAMESTEST = None
        self.bestEstimators = {}
        self.scores = []
        self.MODELPARAMETERS = {
            'svm' : {
                'model': svm.SVC(gamma = 'auto', probability = True),
                'params': {
                    'svc__C': [1,10,100,1000],
                    'svc__kernel': ['rbf', 'linear']
                }
            },
            'random_forest' : {
                'model' : RandomForestClassifier(),
                'params' : {
                    'randomforestclassifier__n_estimators' : [1,5,10]
                }
            },
            'logistic_regression' : {
                'model' : LogisticRegression(solver = 'liblinear', multi_class = 'auto'),
                'params' : {
                    'logisticregression__C': [1,5,10]
                }
            }
        }

    def getFaces(self):
        #This function will be used on individual images to grab the faces of the drivers
        def getCroppedImageWithTwoEyes(imagePath):
            img = cv2.imread(imagePath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, 1.3,5)
            for(x,y,w,h) in faces:
                regionOfInterestColour = img[y:y+h, x:x+w]
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

        for imageDirectory in ImageDirectories:
            # When creating the new file names, we need to ensure that each one is unique.
            # The count will be used for that.
            count = 1

            # When populating my dataset, the folder that contains the images of each driver, is named after the driver itself. 
            # Hence, the name of the driver can be found by grabbing the folder name
            driver = imageDirectory.split('/')[-1]
            self.DriverFileNamesDictionary[driver] = []
            croppedFolder = self.CROPPEDDATAPATH + driver

            # Creates the folder to hold the cropped images
            if not os.path.exists(croppedFolder):
                os.makedirs(croppedFolder)
            
            self.CroppedImageDirectories.append(croppedFolder)

            for entry in os.scandir(imageDirectory):
                try:
                    croppedImage = getCroppedImageWithTwoEyes(entry.path)
                except:
                    croppedImage = None
                
                # If a face with two eyes is found, it's cropped and saved as a new image in the respective folder.
                if croppedImage is not None:
                    # Creates a unique file name via the usage of the count variable.
                    newFileName = driver + str(count) + ".png"
                    # New file path will have to be in the directory with all new cropped images for the respective driver.
                    newFilePath = croppedFolder + "/" + newFileName

                    # Saves the cropped image in the new folder
                    cv2.imwrite(newFilePath, croppedImage)
                    # Adds the new file name to the list of file names for the current driver
                    self.DriverFileNamesDictionary[driver].append(newFileName)
                    count += 1

    def waveletTransform(self, image, mode='haar', level = 1):
        imageArray = image

        # First the image is converted to grayscale
        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)

        # Normalize the image to the range [0, 1] for further processing
        imageArray = np.float32(imageArray)
        imageArray /= 255
    
        # Compute the coefficients
        coefficients = pywt.wavedec2(imageArray, mode, level=level)
    
        # Process said coefficients 
        coefficientsHaar=list(coefficients)
        coefficientsHaar[0] *= 0
    
        # Reconstruct the image with highlighted features
        imageArrayHaar=pywt.waverec2(coefficientsHaar, mode)
        imageArrayHaar *= 255
        imageArrayHaar = np.uint8(imageArrayHaar)
        
        return imageArrayHaar
    
    def createClassificationDictionary(self):
        count = 0
        for driverName in self.DriverFileNamesDictionary.keys():
            self.classificationDictionary[driverName] = count
            count += 1
    
    # This function fills up the names and images arrays. They are parallel arrays where at a given index i,
    # images[i] will contain an image that belongs to the driver at names[i]
    def formImagesAndNames(self):
        for driverName, trainingFiles in self.DriverFileNamesDictionary.items():
            for trainingImage in trainingFiles:
                # Read the image and resize it to a standard size (32x32 pixels)
                try:
                    img = cv2.imread("./model/dataset/cropped/" + driverName + "/" + trainingImage)
                except:
                    continue
                if img is None: continue
                scaledRawImage = cv2.resize(img, (32,32))
                # Transform the image using the wavelet transformation function and then resize it
                imgHar = self.waveletTransform(img, 'db1', 5)
                scaledHarImage = cv2.resize(imgHar, (32,32))
                # Stack the raw and wavelet transformed image on top of one another and put them in the images array
                stacked = np.vstack((scaledRawImage.reshape(32*32*3, 1), scaledHarImage.reshape(32*32, 1)))
                self.IMAGES.append(stacked)
                self.NAMES.append(self.classificationDictionary[driverName])
        
        self.IMAGES = np.array(self.IMAGES).reshape(len(self.IMAGES),4096).astype(float)
    
    def trainModel(self):
        self.IMAGESTRAIN, self.IMAGESTEST, self.NAMESTRAIN, self.NAMESTEST = train_test_split(self.IMAGES, self.NAMES, random_state = 0)
        #Parameters will be changed later if needed
        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 100))])
        pipe.fit(self.IMAGESTRAIN, self.NAMESTRAIN)
        return pipe.score(self.IMAGESTEST, self.NAMESTEST)

    def testModels(self):
        for algorithm, modelParameters in self.MODELPARAMETERS.items():
            pipe = make_pipeline(StandardScaler(), modelParameters['model'])
            clf = GridSearchCV(pipe, modelParameters['params'], cv = 5, return_train_score = False)
            clf.fit(self.IMAGESTRAIN, self.NAMESTRAIN)
            self.scores.append({
                'model' : algorithm,
                'best_score' : clf.best_score_,
                'best_params' : clf.best_params_
            })
            self.bestEstimators[algorithm] = clf.best_estimator_
        
        df = pd.DataFrame(self.scores, columns=['model', 'best_score', 'best_params'])
        return df
    
    def findBestModel(self):
        bestCLM = None
        bestScore = -1

        for model in ['logistic_regression', 'svm', 'random_forest']:
            curScore = self.bestEstimators[model].score(self.IMAGESTEST, self.NAMESTEST)
            if curScore > bestScore:
                bestScore = curScore
                bestCLM = model
        
        return self.bestEstimators[bestCLM]
    
    def createModel(self):
        self.getFaces()
        self.createClassificationDictionary()
        self.formImagesAndNames()
        self.trainModel()
        print(self.testModels())
        bestModel = self.findBestModel()
        joblib.dump(bestModel, 'savedModel.pkl')
        with open("class_dictionary.json", 'w') as f:
            f.write(json.dumps(self.classificationDictionary))
    

m = ModelMaker()

m.createModel()