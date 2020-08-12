from flask import Flask, request, Response
import time
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import time
from builtins import globals
import os

PATH_TO_TEST_IMAGES_DIR = './images'

app = Flask(__name__)

@app.route('/')
def index():
    return Response(open('index.html').read(), mimetype="text/html")

# save the image as a picture
@app.route('/image', methods=['POST'])
def image():
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    frame = request.files['image']  # get the image
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros((250, 300, 3), dtype="uint8")
        # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
        roi = gray[startY:startY + endY, startX:startX + endX]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
    f = ('%s.jpeg' % time.strftime("%Y%m%d-%H%M%S"))
    i.save('%s/%s' % (PATH_TO_TEST_IMAGES_DIR, f))

    return Response("%s saved" % f)
# ,ssl_context='adhoc'
if __name__ == '__main__':
    app.run()
