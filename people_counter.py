import cv2
from sklearn import svm
import numpy as np
from imutils.object_detection import non_max_suppression
import time
import rekap_pengunjung

class PeopleCounter:

    def __init__(self, host, username, password, db):
        self.host = host
        self.username = username
        self.password = password
        self.db = db

    def counter(self):
        preprocessing = ImagePreprocessing()
        detection = ObjectDetection()
        tracking = ObjectTracking()
        RekapPengunjung = rekap_pengunjung.RekapPengunjung(self.host, self.username, self.password, self.db)

        camera = cv2.VideoCapture("../video/TownCentre.avi")
        fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=100, detectShadows=True)

        tracker = cv2.MultiTracker_create()
        init_once = False
        peoples = []
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


        counter = 0
        rangeBox = 8
        flagCounter = False
        while camera.isOpened():
            start_time = time.time()
            ok, image = camera.read()
            image = preprocessing.image_resize(image, 640,480)
            fgmask = fgbg.apply(image)
            if not ok:
                print('no image to read')
                break

            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(2, 2), scale=2.5)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                flag =False
                if init_once :
                    for newbox in boxes:
                        p1 = (int(newbox[0]), int(newbox[1]))
                        p2 = (xA+rangeBox,yA+rangeBox)
                        if tracking.isObjectSimilar(p1,p2,20):
                            flag = True
                            break

                    if flag == False:
                        if tracking.initialCoord(yA+rangeBox,320/2):
                            peoples.append((xA+rangeBox, yA+rangeBox,40,40))
                            cv2.rectangle(image, (xA+rangeBox, yA+rangeBox), (xB, yB), (255, 255, 0), 2)
                            init_once = False

                else :
                    if tracking.initialCoord(yA+10,160) and flagCounter == False:
                        peoples.append((xA+rangeBox, yA+rangeBox,40,40))
                        cv2.rectangle(image, (xA+rangeBox, yA+rangeBox), (xB, yB), (255, 255, 0), 2)

                print("---detect frame no ",1," runtime", (time.time() - start_time) ," seconds ---" )



            if not init_once:
            # if True:
                for object in peoples:
                    ok = tracker.add(cv2.TrackerKCF_create(), image, object)
                init_once = True
                flagCounter = False

            ok, boxes = tracker.update(image)
            print (boxes)

            peoples = []
            tempPeoples = []
            for newbox in boxes:
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                if tracking.initialCoord(p1[1],320/2) == False:
                    counter +=1
                    RekapPengunjung.save()
                    flagCounter = True
                else :
                    cv2.rectangle(image, p1, p2, (200,0,0), 2, 1)
                    tempPeoples.append((newbox[0],newbox[1],40,40))

            if flagCounter:
                del tracker
                tracker = cv2.MultiTracker_create()
                init_once = False
                peoples = tempPeoples

            cv2.putText(image, "in :" + str(counter), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.imshow('tracking', image)


            #- show runtime
            print("---program runtime", (time.time() - start_time) ," seconds ---" )

            k = cv2.waitKey(1)
            if k == 13 :
                del tracker
                tracker = cv2.MultiTracker_create()
                init_once = False
                peoples = []
            if k == 27 : break # esc pressed


class ImagePreprocessing:

    def rgb2gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray


    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
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


class ObjectDetection:
    # hog

    def __init__(self):
        # global hog
        pass

    def hog(self, image):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog.detectMultiScale(image, winStride=(2, 2),padding=(2, 2), scale=1.5)

class ImageClassification:
    pass

class ObjectTracking:

    def isObjectSimilar(self, coordA, coordB, limit):
        xA = coordA[0]
        yA = coordA[1]
        xB = coordB[0]
        yB = coordB[1]
        if(abs(xA-xB) < limit) and (abs(yA-yB) < limit):
            return True
        else:
            return False

    def initialCoord(self,y,limit):
        if y < limit:
            return True
        else :
            return False
