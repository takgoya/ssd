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
Image
'''
print("[INFO] loading image from file ...")
# load input image
image = cv2.imread(conf["image_input"])
# get spatial dimensions from input image
h, w = image.shape[:2]

'''
Blob from image
'''
# construct a blob form the input image
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), scalefactor=0.007843, 
                             size=(300, 300), mean=(127.5, 127.5, 127.5), crop=False)

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
print("[INFO] loading SSD model from disk...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# set minimum probability to filter weak detections
minimum_probability = conf["confidence"]

# set threshold when applying Non-Maxima Suppression
threshold = conf["threshold"]

# generate random colors for bounding boxes
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(labels), 3))

'''
Forward pass
'''
# pass the blob through the network and obtain the detections and predictions
net.setInput(blob)
start = time.time()
detections = net.forward()
end = time.time()

# show spent time for forward pass
print("[INFO] objects detection took {:.6f} seconds".format(end - start))

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
        print("[INFO] {}".format(label))
        cv2.rectangle(image, (startX, startY), (endX, endY), colors[idx], 5)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[idx], 5)
        
'''
Display the image
'''
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Detections", image)
# write new image with detections
cv2.imwrite(conf["image_output"], image)
# wait for any key being pressed
cv2.waitKey(0)
# destroy opened window
cv2.destroyWindow('Detections')        