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

print('Select 3 tracking targets')

cv.namedWindow("tracking")
camera = cv.VideoCapture("../video/TownCentre.avi")
tracker = cv.MultiTracker_create()
init_once = False

ok, image=camera.read()
if not ok:
    print('Failed to read video')
    exit()

image = image_resize(image, 640,480)
bbox1 = cv.selectROI('tracking', image)
# bbox2 = cv.selectROI('tracking', image)
# bbox3 = cv.selectROI('tracking', image)

objects = [bbox1]

while camera.isOpened():
    ok, image=camera.read()
    image = image_resize(image, 640,480)
    if not ok:
        print 'no image to read'
        break

    if not init_once:
        for object in objects:
            ok = tracker.add(cv.TrackerKCF_create(), image, object)
        # ok = tracker.add(cv.TrackerMIL_create(), image, bbox1)
        # ok = tracker.add(cv.TrackerMIL_create(), image, bbox2)
        # ok = tracker.add(cv.TrackerMIL_create(), image, bbox3)
        objects = None
        init_once = True

    start_time = time.time()
    ok, boxes = tracker.update(image)
    print("---detect frame no none runtime", (time.time() - start_time) ," seconds ---" )
    print ok, boxes

    for newbox in boxes:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv.rectangle(image, p1, p2, (200,0,0), 2, 1)

    cv.imshow('tracking', image)

        #--------

    k = cv.waitKey(1)
    if k == 13 : 
        bbox3 = cv.selectROI('tracking', image) # enter pressed
        objects = [bbox3]
        init_once = False
    if k == 27 : break # esc pressed