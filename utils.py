import cv2
import time

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


