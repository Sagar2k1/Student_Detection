import os
from utils import read_videos_from_folder, extract_objects
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np
import pickle

project_location = os.getcwd()
data_location = os.path.join(project_location, 'data')
yolo_weight = os.path.join(project_location, 'yolo_v8_weight\\yolov8n.pt')
step = 1
test_file = ['IMG_0077.MOV']
frame_list = read_videos_from_folder(data_location, test_file, step)
list_info = []
for count, name, img in frame_list:
    start = time.time()
    info = extract_objects(name, count, img, yolo_weight)
    if info is not None:
        list_info.extend(info)
    t = time.time() - start
    print('time: ', t)
embeddings = []
names = []
new_shape = (960, 360)
for crop_img, name, xywh, conf in list_info:
    if conf<0.6:
        continue
    elif conf>=0.6 and conf<0.8:        
        name = 'unknown'
        reshape_img = cv2.resize(crop_img, (360, 960))
    else:
        name = name.split('_')[0]
        reshape_img = cv2.resize(crop_img, (360, 960))
    vec = reshape_img.flatten()
    embeddings.append(vec)
    names.append(name)
embeddings = np.array(embeddings)
names = np.array(names)
data = {
    'embeddings': embeddings,
    'names': names
}
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()