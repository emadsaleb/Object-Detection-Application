import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
#load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
st.title('Object Detection Application')
upload = st.file_uploader('Upload Your Image.....' , type = ['png' , 'jpg' , 'jpeg'])
if upload is not None:
    image = Image.open(upload)
    image_array = np.array(image)
    results = model(image_array)[0]
    boxes = results.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    class_names = [model.names[c] for c in class_ids]
    unique_classes = sorted(set(class_names))
    selected_classes = st.multiselect('Image classes...', unique_classes, default= unique_classes)
    for box , cls_name , conf in zip(xyxy , class_names , confidences):
        if cls_name in selected_classes:
            x1 , x2 , y1 , y2 = box
            label = f"{class_names} {conf:.2f}"
            cv2.rectangle(image_array , (x1 ,y1), (x2 , y2) , (0 , 255 , 90) , 2)
            cv2.putText(image_array , label , (x1 , y1),
                       fontFace= cv2.FONT_HERSHEY_SIMPLEX , fontScale= 0.8 , color = (90 , 200 , 250) , thickness=2)
            st.image(image_array , caption= 'detected objects')
            
    
                          
    
