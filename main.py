from ultralytics import YOLO
import cv2

model = YOLO('./Yolo-weights/yolov8l.pt')

results = model("Images/3.webp", show=True)
cv2.waitKey(0)
