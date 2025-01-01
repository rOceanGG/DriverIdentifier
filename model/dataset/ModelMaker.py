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