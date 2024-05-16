from utils import read_video
import cv2
import os
import time

path = 'data\Video'
path_saved = 'data\Frame'
step = 20

for i in os.listdir(path):
    file_vid = os.path.join(path, i)
    video_name = i.split('.')[0]
    location_saved = os.path.join(path_saved, video_name)
    if not os.path.exists(location_saved):
        try:
            os.mkdir(location_saved)
        except:
            continue
    frames = read_video(file_vid, step)
    count = 0
    for frame in frames:
        name = f'{video_name}_{count}.jpg'
        cv2.imwrite(os.path.join(location_saved,name), frame)
        print(f'created {name} in {location_saved}')
        count +=1
        time.sleep(1)
