from ultralytics import YOLO
import numpy
from PIL import Image

# load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  

# predict on an image
# detection_output = model.predict(source="inference/images/Bottle.JPEG", conf=0.25, save=True) 
detection_output = model.predict(source="inference/images/Cup.png", conf=0.25, save=True) 

result = detection_output[0]

box = result.boxes[0]

cords = box.xyxy[0].tolist()
cords = [round(x) for x in cords]
class_id = result.names[box.cls[0].item()]
conf = round(box.conf[0].item(), 2)
print("Object type:", class_id)
print("Coordinates:", cords)
print("Probability:", conf)

Image.fromarray(result.plot()[:, :, ::-1]).show()

# # Display tensor array
# print(detection_output)

# # Display numpy array
# print(detection_output[0].numpy())
