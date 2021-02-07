# import necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import cv2
import numpy as np
import time
import json
import argparse
import os

'''
Arguments
'''
# json
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
args = vars(ap.parse_args())
# load the configuration
conf = json.load(open(args["conf"]))

'''
Input video 
'''
# initialize the video stream and allow the camera
# sensor to warmup
print("[SSD] warming up camera...")
vs = VideoStream(usePiCamera=conf["use_picamera"],
                 resolution=tuple(conf["resolution"]),
                 framerate=conf["fps"]).start()

time.sleep(conf["camera_warmup_time"])
fps = FPS().start()
    
# prepare variable for writer that we will use to write processed frames
writer = None
# prepare variables for spatial dimensions of the frames
h, w = None, None
print("[SSD] starting video from camera ...")

'''
Load SSD network
'''
# initialize the list of class labels MobileNet SSD was trained to detect
labels = ["background", "aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
     "sofa", "train", "tvmonitor"]

# load prototxt and model path
prototxt_path = conf["prototxt"]
model_path = conf["model"]

# load SSD model from disk
print("[SSD] loading model ...")
start_time = time.time()
print("[SSD] loading SSD model from disk...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
end_time = time.time()
elapsed_time = end_time - start_time
print("[SSD] model loaded ... took {} seconds".format(elapsed_time))

# set minimum probability to filter weak detections
minimum_probability = conf["confidence"]

# set threshold when applying Non-Maxima Suppression
threshold = conf["threshold"]

# generate random colors for bounding boxes
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(labels), 3))

'''
Read frames in the loop
'''
# variable for counting frames
f = 0

# variable for counting total time
t = 0

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    frame = vs.read()
    
    # if the frame was not grabbed, then end of the stream
    #if not grabbed:
    #    break
    
    # get spatial dimensions of the frame (only 1st time)
    if w is None or h is None:
        h, w = frame.shape[:2]
    
    '''
    Blob from frame
    '''
    # construct a blob form the input image
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), scalefactor=0.007843, 
                                 size=(300, 300), mean=(127.5, 127.5, 127.5), crop=False)

    '''
    Forward pass
    '''
    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    end = time.time()

    # increase counters for frames and total time
    f += 1
    t += end - start
    
    # show spent time for forward pass
    #print("[INFO] objects detection took {:.6f} seconds".format(end - start))

    '''
    Get and Draw bounding boxes
    '''
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > minimum_probability:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 5)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[idx], 5)

    '''
    Showing processed frames in real time
    Write processed frame into file
    '''
    # show the output frame
    cv2.namedWindow("SSD Real Time Detections", cv2.WINDOW_NORMAL)
    cv2.imshow("SSD Real Time Detections", frame)
    
    fps.update()

    if writer is None:
        # initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*conf["video_codec"])
        writer = cv2.VideoWriter(conf["video_camera_output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)

    # write processed current frame to the file
    writer.write(frame)

    key = cv2.waitKey(1) & 0xFF
    # if the "Esc" key was pressed, break from the loop
    if key == (ord("q")) or key == 27:
        break

'''
Finish
'''
# stop the timer and display FPS information
fps.stop()
print("[SSD] elasped time: {:.2f}".format(fps.elapsed()))
print("[SSD] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
print("[SSD] cleaning up...")
# release video reader and writer
cv2.destroyAllWindows()
vs.stop()