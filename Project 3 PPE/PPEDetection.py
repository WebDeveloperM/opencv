import math

import cv2
import cvzone
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# cap = cv2.VideoCapture("../Videos/ppe-2-1.mp4")

# cap = cv2.VideoCapture(0)
# address = "http://192.168.1.6:8080/video"
# cap.open(address)

# cap.set(2, 1280)
# cap.set(3, 720)

# cap = cv2.VideoCapture("../Videos/ppe-1-1.mp4")

model = YOLO("../Project 3 PPE/ppe.pt")
classNames = ['Hardhat', 'No-Hardhat', 'No-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest']
myColor = (0, 0, 255)
          
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            w, h = x2-x1, y2-y1
            # cvzone.cornerRect(img, (x1,y1,w,h))


            conf = math.ceil((box.conf[0]*100))/100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                myColor = (0,255,0)
            else:
                myColor = (0,0,255)

            cvzone.putTextRect(img, f"{classNames[cls]} {conf, 1}", (max(0,x1),max(40, y1)), scale=1, thickness=1,
                               colorB = myColor, colorT=(255,255,255), colorR=myColor, offset = 5)

            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)


    cv2.imshow("Images", img)
    cv2.waitKey(1)
