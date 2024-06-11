# import libraries
import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from mtcnn.mtcnn import MTCNN  # Make sure to install MTCNN via pip

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video file stream
video_path = "IMG_0077.mp4"
print("Opening video file...")
vs = cv2.VideoCapture(video_path)
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# initialize MTCNN face detector
detector = MTCNN()

# loop over frames from the video file stream
while True:
    # grab the frame from the video file stream
    ret, frame = vs.read()
    
    # If we reached the end of the video, break from the loop
    if not ret:
        break
    
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # detect faces in the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    # loop over the detections
    for detection in detections:
        confidence = detection['confidence']

        # filter out weak detections
        if confidence > 0.7:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detection['box']
            (startX, startY, endX, endY) = (
                box[0], box[1], box[0] + box[2], box[1] + box[3])

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated probability
            if proba > 0.5:
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.release()
