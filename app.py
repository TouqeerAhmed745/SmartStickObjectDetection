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

app = flask.Flask(__name__)

label = "pothole"
total = []
objectDetection = []


@app.route('/favicon')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/favicon.png')


@app.route('/')
@app.route('/home', methods=["GET", "POST"])
def home():
    default_value = '0'
    url = request.form.get('url', default_value)
    print(url)
    ObjectDetect(url)
    detectImage(url)
    return {"response":
                {"pothole_detection": total, "objectDetection": objectDetection}}


def ObjectDetect(url):
    #img = cv2.imread('data/picture/lena.png')
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
        print(name)
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
    total.clear()
    for (classId, score, box) in zip(classIds, scores, boxes):
        x, y, w, h = box
        print(str(round(scores[0] * 100, 2)))
        total.append(label + ": " + str(round(scores[0] * 100, 2)))


if __name__ == "__main__":
    app.secret_key = '25d9984ab936d7f1eb755fdd49e3882ae797e644c5996d56c097b40d6e72408b'
    app.debug = False
    app.run()
