import cv2
import image as imageDetect


def ObjectDetect():
    img = cv2.imread('data/picture/lena.png')
    #cap = cv2.VideoCapture('http://192.168.137.159')
    #cap.set(3,1280)
    #cap.set(4,720)
    #cap.set(10,70)
    #req = urllib.request.urlopen('http://192.168.137.159/cam-hi.jpg?w=681&h=383&crop=1&resize=681%2C383')
    #arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    #img = cv2.imdecode(arr, -1) # 'Load it as it is'

    #cv2.imshow('random_title', img)

    classNames= []
    classFile = 'coco.names'
    with open(classFile,'rt') as f:
     classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img,confThreshold=0.5)

    for classIds,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
        print(classNames[classIds-1])
        print(cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img,classNames[classIds-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('output',img)

    cv2.waitKey(0)

ObjectDetect()
#camera_video.cameraVideo()
imageDetect.imageDetection()