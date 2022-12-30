# importing necessary libraries
import cv2
import os

def imageDetection():
    # reading test image
    img = cv2.imread("data/picture/p5.jpg")  # image name

    # reading label name from obj.names file
    with open(os.path.join("project_files", 'obj.names'), 'r') as f:
        classes = f.read().splitlines()

    # importing model weights and config file
    net = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

    # detection
    for (classId, score, box) in zip(classIds, scores, boxes):
        x, y, w, h = box
        label = "pothole"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, "%" + str(round(scores[0] * 100, 2)) + " " + label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("pothole", img)
    cv2.imwrite("result1" + ".jpg", img)  # result name
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imageDetection()