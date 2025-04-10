import cv2
from ultralytics import YOLO

model = YOLO("/home/adip/workspaces/image_processing_ws/HardDrive_Segmentation/runs/segment/train/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, task="segment", verbose=False)
    result_img = results[0].plot(conf=0.8)

    cv2.imshow("YOLOv11 Segmentation", result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
