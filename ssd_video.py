# import necessary packages
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
print("[SSD] loading video from file ...")

# load input video
vs = cv2.VideoCapture(conf["video_input"])

# prepare variable for writer that we will use to write processed frames
writer = None

# prepare variables for spatial dimensions of the frames
h, w = None, None

# try to determine the total number of frames in the video file
try:
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[SSD] {} total frames in video".format(total))
# an error occurred while trying to determine the total number of frames in the video file
except:
    print("[SSD] could not determine # of frames in video")
    print("[SSD] no approx. completion time can be provided")
    total = -1

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
    grabbed, frame = vs.read()
    
    # if the frame was not grabbed, then end of the stream
    if not grabbed:
        break
    
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
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[idx], 3)


    '''
    Write processed frame into file
    '''
    if writer is None:
        # initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*conf["video_codec"])
        writer = cv2.VideoWriter(conf["video_output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
        
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[SSD] single frame took {:.4f} seconds".format(elap))
            print("[SSD] estimated total time to finish: {:.4f}".format(elap * total))

    # write processed current frame to the file
    writer.write(frame)
    
# print final results
print()
print("[SSD] total number of frames", f)
print("[SSD] total amount of time {:.5f} seconds".format(t))
print("[SSD] fps:", round((f / t), 1))
print()
print("[SSD] cleaning up")

# release video reader and writer
vs.release()
writer.release()