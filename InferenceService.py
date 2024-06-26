import io
import time
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import imutils
from pathlib import Path
import locale
from PIL import Image
import random

basePath = Path(__file__).parent
filePath = (f"{basePath}")

modelpath = f"{basePath}/complex-cnn-kaggle_dataset.pb"
locale.getdefaultlocale()
model = load_model(modelpath)

def loadImage(path, imageSize):
    imageWidth, imageHeight = imageSize
    # load the image
    image = cv2.imread(path)
    # crop the brain and ignore the unnecessary rest part of the image
    image = cropBrainContour(image)
    # resize image
    image = cv2.resize(image, dsize=(imageWidth, imageHeight), interpolation=cv2.INTER_CUBIC)
    # normalize values
    image = image / 255.

    return image

def preprocess(image, imageSize):
    imageWidth, imageHeight = imageSize
    # crop the brain and ignore the unnecessary rest part of the image
    image = cropBrainContour(image)
    # resize image
    image = cv2.resize(image, dsize=(imageWidth, imageHeight), interpolation=cv2.INTER_CUBIC)
    # normalize values
    #image = image / 255.

    return image

def cropBrainContour(image):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return new_image


def predict(imageFile):
    jpg = Image.open(imageFile)
    pixelArray = np.array(jpg)

    processedImage = preprocess(pixelArray, (512, 512))

    X = np.array(processedImage)
    X = np.expand_dims(X, 0)

    pred = model.predict(X)

    return pred[0][0]

def validateImage(imageFile):
    jpg = Image.open(imageFile)
    pixelArray = np.array(jpg)

    isValidImageSize(imageFile)

    print(pixelArray)

def isValidImageSize(imageFile):
    try:
        jpg = Image.open(imageFile)
        pixelArray = np.array(jpg)
        shape = pixelArray.shape
        if shape[0] >= 50 and shape[1] >= 50 and shape[2] == 3:
            return True
    except:
        return False

def isImageGrayscale(imageFile):
    jpg = Image.open(imageFile)
    pixelArray = np.array(jpg)
    shape = pixelArray.shape
    rows = shape[0]
    cols = shape[1]

    testingPoints = min(max((rows * cols) / 100, 30), 50)

    for i in range(int(testingPoints)):
        row = random.randrange(0, rows, 1)
        col = random.randrange(0, cols, 1)

        if (pixelArray[row][col][0] != pixelArray[row][col][1] or pixelArray[row][col][1] != pixelArray[row][col][2]):
            return False

    return True