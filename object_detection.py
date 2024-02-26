from ultralytics import YOLO
import cv2 as cv
import cvzone as cz
import math
# cap = cv.VideoCapture(0)

# cap.set(3, 980)  # width of capture
# cap.set(4, 720)  # hight of capture

cap = cv.VideoCapture('RoadSafety.mp4')

ClassNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',  'train',  'truck', 'boat', 'traffic light', 'fire hydrant',  'stop sign',  'parking meter',  'bench',  'bird', 'cat', 'dog',  'horse', 'sheep', 'cow',  'elephant', 'bear', 'zebra',  'giraffe',  'backpack',  'umbrella',  'handbag',  'tie',  'suitcase',  'frisbee',  'skis',  'snowboard',  'sports ball',  'kite',  'baseball bat',  'baseball glove',  'skateboard',  'surfboard',  'tennis racket',
              'bottle',  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',  'banana', 'apple', 'sandwich',  'orange', 'broccoli',  'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',  'couch',  'potted plant', 'bed',  'dining table', 'toilet',  'tv', 'laptop',  'mouse',  'remote',  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',  'toothbrush']
model = YOLO('yolov8n.pt')

# result = model('thepark.jpg', show=True)


while True:
    frames, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            # cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
            cz.cornerRect(img, (x1, y1, w, h), l=12)

            # for text
            coif = math.ceil((box.conf[0]*100))/100
            # for class
            cls = box.cls[0]
            cz.putTextRect(img, f'{ClassNames[int(cls)]} {coif}',
                           (max(0, x1), max(40, y1)), scale=1, thickness=2, offset=5)
    cv.imshow('Object Detection', img)
    cv.waitKey(1)
