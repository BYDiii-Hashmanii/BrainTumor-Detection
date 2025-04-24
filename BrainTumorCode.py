# print("SAM2 Model and Yollo 11")
from ultralytics import YOLO
model= YOLO("yolo11n.pt") #Selection of Image Detection Model
print("SAM2 Model and Yollo 11")
# Training the Model on YOLOv11 
train_results= model.train(
    #data Location is given along with yaml path
    data="D:/tumor_detection/data.yaml",
#    Number of Epochs Set
    epochs=2, 
    imgsz=640,
    device='cpu',
    cache=True,
)
model = YOLO('C:/BrainTumorDetction/runs/detect/train2/weights/best.pt')

results = model('C:/BrainTumorDetction/test_images/meng1.jpg')

results[0].show()
for result in results:
    boxes = result.boxes
    print(boxes)

