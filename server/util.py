import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import waveletTransform
__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def loadArtifacts():
    global __class_name_to_number
    global __class_number_to_name

    with open("./server/artifacts/class_dictionary.json") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}
    
    global __model
    if __model is None:
        with open("./server/artifacts/savedModel.pkl", 'rb') as f:
            __model = joblib.load(f)
    
def classifyImage(imageB64, filePath = None):
    # Looks for all faces in the input image
    imgs = getCroppedFaces(filePath, imageB64)
    # Setting the default result should a face not be found
    result = {"DriverName": "Unknown", "DriverProbability": 0}
    for img in imgs:
        scaledRawImage = cv2.resize(img, (32,32))
        imgHar = waveletTransform(img, 'db1', 5)
        scaledHarImage = cv2.resize(imgHar, (32,32))
        stacked = np.vstack((scaledRawImage.reshape(32*32*3, 1), scaledHarImage.reshape(32*32, 1)))

        imgArrLen = stacked.size

        finalImg = stacked.reshape(1,imgArrLen).astype(float)

        curProb = max(__model.predict_proba(finalImg)[0])

        # First condition checks if this face had a better match than any of the previous ones.
        # Second condition ensures that we have a degree of certainty we actually have a face that matches a driver
        if curProb > result["DriverProbability"] and curProb > 0.5:
            result["DriverName"] = getDriverNameFromID(__model.predict(finalImg)[0])
            result["DriverProbability"] = curProb
    
    return result

def getCv2ImageFromB64String(B64Str):
    encoded = B64Str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def getCroppedFaces(imagePath, imageB64):
    # This function is just copied from modelmaker with some tiny changes
    faceCascade = cv2.CascadeClassifier('./server/opencv/haarcascade/haarcascade_frontalface_default.xml')
    if imagePath:
        img = cv2.imread(imagePath)
    else:
        img = getCv2ImageFromB64String(imageB64)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3,5)
    croppedFaces = []
    for(x,y,w,h) in faces:
        regionOfInterestColour = img[y:y+h, x:x+w]
        croppedFaces.append(regionOfInterestColour)
    
    return croppedFaces

def getDriverNameFromID(driverID):
    return __class_number_to_name.get(driverID, "Unknown")

def getB64TestForKimi():
    with open("./server/b64.txt") as f:
        return f.read()

if __name__ == "__main__":
    loadArtifacts()
    # Tests
    # res = classifyImage(None, "./model/dataset/CarlosSainz/gettyimages-2131328781-612x612.jpg")
    # print(f"Driver: {res["DriverName"]}")
    # print(f"Probability: {res['DriverProbability']}")
    # Expect Output: Carlos Sainz
    # Actual Output: Carlos Sainz (TEST PASSED)

    # res = classifyImage(getB64TestForKimi())
    # print(f"Driver: {res["DriverName"]}")
    # print(f"Probability: {res['DriverProbability']}")
    # Expect Output: Kimi Antonelli
    # Actual Output: Kimi Antonelli (TEST PASSED)

    # res = classifyImage(None, 'model/dataset/CharlesLeclerc/gettyimages-1770581579-612x612.jpg')
    # print(f"Driver: {res["DriverName"]}")
    # print(f"Probability: {res['DriverProbability']}")
    # Expect Output: Charles Leclerc
    # Actual Output: Charles Leclerc (TEST PASSED)