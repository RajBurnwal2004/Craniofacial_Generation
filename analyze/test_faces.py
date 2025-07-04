from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # COCO-pretrained

image_path = "eval/fake_B/002_sf_aug0_fake_B.png"
results = model(image_path)

# Visualize results
res_plotted = results[0].plot()
cv2.imshow("Detections", res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()
