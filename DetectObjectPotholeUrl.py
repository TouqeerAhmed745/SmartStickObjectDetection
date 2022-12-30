
import flask
import os
import cv2
from flask import send_from_directory
import time
from geopy.geocoders import Nominatim
import os
import urllib
import numpy as np
import urllib.request
from flask import request
import math

from werkzeug.utils import secure_filename

url =""
label = "pothole"
potholes = []
potholeDistance = []
objectDetection = []
KNOWN_DISTANCE = 24.0
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0






def fileFromUrl():
    default_value = ""
    url = request.form.get('url', default_value)
    print(url)
    if url != default_value:
        ObjectDetect(url)
        detectImage(url)
        return {
            "response":
                {
                    "status": 200,
                    "massage": "Successfully get Result",
                    "pothole_detection": potholes, "objectDetection": objectDetection,
                    "potholesDistance": potholeDistance

                }
        }
    else:
        return {
            "response":
                {
                    "status": 201,
                    "massage": "Please Provide Url",
                }
        }
def ObjectDetect(url):
    # img = cv2.imread('data/picture/lena.png')
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, scores, bbox = net.detect(img, confThreshold=0.5)
    objectDetection.clear()
    for classIds, scores, box in zip(classIds, scores, bbox):
        name = classNames[classIds - 1]
        objectDetection.append(name)

        # cv2.imshow('output',img)

    # cv2.waitKey(0)



def detectImage(url):
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)

    # img = cv2.imread("data/picture/p5.jpg")  # image name
    # reading label name from obj.names file
    with open(os.path.join("project_files", 'obj.names'), 'r') as f:
        classes = f.read().splitlines()
    # importing model weights and config file
    net = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
    # detection
    potholes.clear()
    potholeDistance.clear()
    for (classId, score, box) in zip(classIds, scores, boxes):
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imwrite("im.jpg", img)

        #  marker = find_marker(img)
        t = 101
        d = 50
        height, width, channels = img.shape
        # print 'width: ', width
        # print 'Traget width: ' , w
        v = t / (w / width)  # w is how many pixels wide is the object
        # print v
        h = 0.5 * v / (math.tan(math.radians(d) / 2))
        # now we need to use basic trigo to caculate actualy distance, given that he camera is tilted
        tilt = 30  # the tilt of the camera, in degrees
        distance = h * (math.cos(math.radians(tilt)))

        potholeDistance.append(round(distance,2))
        potholes.append(label + ": " + str(round(scores[0] * 100, 2)))
