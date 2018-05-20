# from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils

import numpy as np
import cv2 as cv
import sys
import time

# if len(sys.argv) != 2:
#     print('Input video name is missing')
#     exit()

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
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
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def isObjectSimilar(coordA,coordB, limit):
    xA = coordA[0]
    yA = coordA[1]
    xB = coordB[0]
    yB = coordB[1]
    if(abs(xA-xB) < limit) and (abs(yA-yB) < limit):
        return True 
    else :
        return False

print('Select 3 tracking targets')

#detection 
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cv.namedWindow("tracking")
camera = cv.VideoCapture("../video/TownCentre.avi")
tracker = cv.MultiTracker_create()
init_once = False

ok, image=camera.read()
if not ok:
    print('Failed to read video')
    exit()

image = image_resize(image, 640,480)

peoples = []
while camera.isOpened():
    start_time = time.time()
    ok, image=camera.read()
    image = image_resize(image, 640,480)
    if not ok:
        print 'no image to read'
        break

    (rects, weights) = hog.detectMultiScale(image, winStride=(2, 2),padding=(2, 2), scale=1.5)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # pick = rects
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:

        if init_once :
            for newbox in boxes:
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (xA,yA)
                if not isObjectSimilar((xA,yA),(xB,yB),10):
                    
                    peoples.append((xA+10, yA+10,40,40))
                    cv.rectangle(image, (xA+10, yA+10), (xB, yB), (255, 255, 0), 2)
                    break

                print("similarity = ",isObjectSimilar((xA,yA),(xB,yB),10))
        else :
            peoples.append((xA+10, yA+10,40,40))
            cv.rectangle(image, (xA+10, yA+10), (xB, yB), (255, 255, 0), 2)

        print("---detect frame no ",1," runtime", (time.time() - start_time) ," seconds ---" )


    if not init_once:
    # if True:
        for object in peoples:
            ok = tracker.add(cv.TrackerKCF_create(), image, object)
        
        peoples = []
        init_once = True

    ok, boxes = tracker.update(image)
    # print boxes

    for newbox in boxes:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv.rectangle(image, p1, p2, (200,0,0), 2, 1)
        peoples.append((newbox[0],newbox[1],40,40))

    cv.imshow('tracking', image)
        
    print tracker.getObjects()    
    # tracker.clear()
    # tracker = cv.MultiTracker_create()
    
    #- show runtime
    print("---program runtime", (time.time() - start_time) ," seconds ---" )

    k = cv.waitKey(1)
    if k == 13 : 
        bbox3 = cv.selectROI('tracking', image) # enter pressed
        objects = [bbox3]
        init_once = False
    if k == 27 : break # esc pressed