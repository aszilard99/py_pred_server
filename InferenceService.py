import time
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import imutils
from pathlib import Path

basePath = Path(__file__).parent
filePath = (f"{basePath}")


def load_image(path, image_size):
    image_width, image_height = image_size

    # load the image
    image = cv2.imread(path)

    # crop the brain and ignore the unnecessary rest part of the image
    image = crop_brain_contour(image)
    # resize image
    image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    # normalize values
    image = image / 255.

    return image


def crop_brain_contour(image):
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


def predict():
    modelpath = f"{basePath}/simplecnn-kagglebrain"
    print(f"modelpath: {modelpath}")

    model = load_model(modelpath)

    filename = 'aug_Y2_0_9816'

    path = f"{basePath}/{filename}.jpg"

    image = load_image(path, (240, 240))

    X = np.array(image)
    X = np.expand_dims(X, 0)

    t0 = time.time()
    pred = model.predict(X)
    t1 = time.time()

    print(f'time {t1 - t0}')
    print(f'pred {pred}')

    return pred


