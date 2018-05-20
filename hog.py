    # USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
#args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
#imagePaths = list(paths.list_images(args["images"]))
#imagePaths = "images/person_265.bmp"
bg = cv2.createBackgroundSubtractorMOG2()

video_capture = cv2.VideoCapture("../video/TownCentre.avi")
# video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture("../video/testingImage.mp4")

no_frame = 0
while True:
    
    # Capture frame-by-frame
        #start time for read frame
    start_time = time.time()
        #---------
    
    ret, image = video_capture.read()
    image = image_resize(image, 640,480)
    no_frame += 1
        #--------
    # print("---read frame runtime %s seconds ---" % (time.time() - start_time))
        #--------

    # detect people in the image
        # start time for read frame
    # start_time = time.time()
        #---------
    
    (rects, weights) = hog.detectMultiScale(image, winStride=(2, 2),padding=(2, 2), scale=1.5)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (255, 255, 0), 2)
        print("---detect frame no ",no_frame," runtime", (time.time() - start_time) ," seconds ---" )

        #--------
    print("---detect frame no none runtime", (time.time() - start_time) ," seconds ---" )

        #--------
        
    # image = imutils.resize(image, 720)
    
    image = image_resize(image, 640,480)
    # Display the resulting frame
    cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

###
