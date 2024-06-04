import cv2
import time
import os

def show_img(img):
    import matplotlib.pyplot as plt
    import cv2
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(bgr_img)
    plt.axis('off')
    plt.show()

def read_video(file, step, duration = None):
    frame_list = []
    count = 0
    vid = cv2.VideoCapture(file)
    while True:
        _, frame = vid.read()
        count +=1
        if _== False:
            break
        elif count % step == 0:
            # h, w, _ = frame.shape
            # frame = cv2.resize(frame,(int(w/2),int(h/2)))
            frame_list.append(frame)
        if duration == None:
            pass
        elif count == duration:
            break
    return frame_list

def read_videos_from_folder(data_location = 'data', test_files = None, step_read = 1, duration_each = 1000):    
    dirs = os.listdir(data_location)
    frame_list = []
    for name in dirs:
        if test_files == None:
            pass
        elif name in test_files:
            continue
        filename = os.path.join(data_location, name)
        count = 0
        for mediafile_dirname in os.listdir(filename):
            mediafile = os.path.join(filename, mediafile_dirname)
            frames = [(count+i, name, j) for i, j in enumerate(read_video(mediafile, step_read, duration_each))]
            count += len(frames)
            frame_list.extend(frames)
    return frame_list

def extract_objects(label = 'Nhân', count = 0, img = None, yolo_weight = None):
    from ultralytics import YOLO
    model = YOLO(yolo_weight)
    result = model( img, classes = [0])
    for r in result:
        boxes = r.boxes
    if boxes.shape[0] == 0:
        return None
    elif boxes.shape[0] == 1:
        x, y, w, h = [i for i in boxes.xywh.numpy()[0]]
        confidence = boxes.conf.numpy()[0]
        crop_img = img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
        if w< 20 or h <20:
            return None
        name = label+'_'+str(count)
        location = (x,y,w,h)
        if confidence<0.5:
            return None
        else:
            return [(crop_img, name, location, confidence)]
    else:
        result_list = []
        for i, box in enumerate(boxes):        
            x, y, w, h = [i for i in box.xywh.numpy()[0]]
            confidence = box.conf.numpy()[0]
            crop_img = img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
            name = label+'_'+str(count)+'_'+str(i)
            location = (x,y,w,h)
            
            if confidence<0.5 or (w< 20 or h <20):
                continue
            result_list.append((crop_img, name, location, confidence))
        return result_list