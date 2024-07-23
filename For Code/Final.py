from ultralytics import YOLO
# # Load a model
# #model = YOLO("yolov8n.yaml", task="detect")  # build a new model from scratch
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# # Use the models
train_results = model.train(data="config.yaml", epochs=500, batch = 16, imgsz = 640, optimizer='SGD', rect = True)  # train the model

model = YOLO('runs/detect/train/weights/best.pt')
metrics = model.val()
results = model('/home/campus30/dnarsipu/Downloads/For Code/Regular/images/test/')


####For Enhanced ModelS
# model2 = YOLO('runs/detect/train/weights/best.pt')
# train_results = model2.train(data="config2.yaml")
model2 = YOLO('runs/detect/train/weights/best.pt')
metrics2 = model2.val(data = 'config2.yaml')
results2 = model2('/home/campus30/dnarsipu/Downloads/For Code/Enhanced/images/test/')


