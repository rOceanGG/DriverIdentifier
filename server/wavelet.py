import numpy as np
import cv2
import pywt

def waveletTransform(image, mode='haar', level = 1):
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