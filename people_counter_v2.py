import cv2
from sklearn import svm
import numpy as np
from imutils.object_detection import non_max_suppression
import time
import rekap_pengunjung
import math

class PeopleCounter:

    def __init__(self, host, username, password, db):
        self.host = host
        self.username = username
        self.password = password
        self.db = db
        # self.counter = 0
        self.rangeBox = 5
        self.flagCounter = False
        self.lineDetection= 100
        self.lineDetection2= 60
        self.lineCounter= int(320/2)
        self.lineThickness = 2
        self.widthBox = 32
        self.heightBox = 32

    def counter(self):
        preprocessing = ImagePreprocessing()
        detection = ObjectDetection()
        tracking = ObjectTracking()
        RekapPengunjung = rekap_pengunjung.RekapPengunjung(self.host, self.username, self.password, self.db)

        # camera = cv2.VideoCapture("../video/video/DatatestMKUbalun2.mp4")
        # camera = cv2.VideoCapture("../video/Datatest207v2.avi")
        camera = cv2.VideoCapture("../eksplore/video/video/WIN_20180720_14_09_23_Pro.mp4")
        # camera = cv2.VideoCapture("../video/video/Datatest207v8.avi")
        # camera = cv2.VideoCapture("../video/TownCentre.avi")
        # camera = cv2.VideoCapture(1)

        tracker = cv2.MultiTracker_create()
        init_once = False
        peoples = []
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        counter = 0
        boxes = []
        tempBoxes = []
        good_oldest =[]
        no_frame = 0

        feature_params = dict( maxCorners = 1,
                               qualityLevel = 0.1,
                               minDistance = 7,
                               blockSize = 4 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (32,32),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        ok, image = camera.read()
        image = preprocessing.image_resize(image, 640,480)
        old_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        while camera.isOpened():
            start_time = time.time()
            ok, image = camera.read()
            image = preprocessing.image_resize(image, 640,480)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            no_frame =no_frame +1
            if not ok:
                print('no image to read')
                break

            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(2, 2), scale=2.5)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            # draw the final bounding boxes
            flag_new_object = False
            for (xA, yA, xB, yB) in pick:
                flag =False
                if init_once :
                    for newbox in boxes:
                        p1 = (int(newbox[0][0]), int(newbox[0][1]))
                        p2 = (xA+self.rangeBox,yA+self.rangeBox)
                        region = image_gray[p2[1]:p2[1]+self.heightBox , p2[0]:p2[0]+self.widthBox]
                        point = cv2.goodFeaturesToTrack(region, mask = None, **feature_params)
                        if(point is not None):    
                            xPoint = point[0][0][0] + p2[0]
                            yPoint = point[0][0][1] + p2[1]
                            p2 = (xPoint,yPoint)
                        # print newbox
                        # print p2
                            if tracking.isObjectSimilar(p1,p2,(32,64)):
                                flag = True
                                break

                    if flag == False:
                        if tracking.initialCoord(yA+self.rangeBox,self.lineDetection) and (tracking.initialCoord(yA+self.rangeBox,self.lineDetection2) == False) and abs(yA-yB) < 150:
                            peoples.append((xA+self.rangeBox, yA+self.rangeBox,self.widthBox,self.heightBox))
                            cv2.rectangle(image, (xA+self.rangeBox, yA+self.rangeBox), (xB, yB), (255, 255, 0), 2)
                            flag_new_object = True

                else :
                    if tracking.initialCoord(yA+self.rangeBox,self.lineDetection) and (tracking.initialCoord(yA+self.rangeBox,self.lineDetection2) == False) and self.flagCounter == False and abs(yA-yB) < 150:
                        peoples.append((xA+self.rangeBox, yA+self.rangeBox,self.widthBox,self.heightBox))
                        cv2.rectangle(image, (xA+self.rangeBox, yA+self.rangeBox), (xB, yB), (255, 255, 0), 2)

                print("---detect frame no ",no_frame," runtime", (time.time() - start_time) ," seconds ---" )


            if init_once == False or flag_new_object:
                for object in peoples:
                    boxes = tracking.set_object(image_gray, boxes, object)
                    # print boxes
                    # ok = tracker.add(cv2.TrackerKCF_create(), image, object)
                init_once = True
                self.flagCounter = False

            # calculate optical flow
            if len(boxes) != 0 :
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, image_gray, boxes , None, **lk_params)
                good_new = p1[st==1]
                good_old = boxes[st==1]

                if(no_frame == 1):
                    good_oldest = good_new
                
                index_to_delete = []
                
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()

                    cv2.rectangle(image, (int(a)-self.widthBox,int(b)-self.heightBox), (int(a)+self.widthBox,int(b)+self.heightBox), (255, 255, 0), 2)
                    frame = cv2.circle(image,(a,b),5,color[i].tolist(),-1)
                            


                #counter & delete
                current_object = boxes
                index_to_delete = []
                for idx,val in enumerate(current_object):
                    # print(idx,val)
                    if tracking.initialCoord(val[0][1]-self.heightBox,self.lineCounter) == False:
                        counter +=1
                        RekapPengunjung.save()
                        index_to_delete.append(idx)
                good_new = np.delete(good_new, index_to_delete,0)
                
            
            cv2.putText(image, "in :" + str(counter), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.line(image, (0, self.lineCounter), (640, self.lineCounter), (0,255,0), self.lineThickness)
            cv2.line(image, (0, self.lineDetection), (640, self.lineDetection), (0,255,0), self.lineThickness)
            cv2.line(image, (0, self.lineDetection2), (640, self.lineDetection2), (0,255,0), self.lineThickness)
            cv2.imshow('tracking', image)


            #- show runtime
            print("---program runtime", (time.time() - start_time) ," seconds ---" )

            peoples = []
            old_gray = image_gray.copy()
            if len(boxes) != 0 :
                boxes = good_new.reshape(-1,1,2)
                # print("good_new reshape",boxes)
  
            k = cv2.waitKey(1)
            if k == 13 :
                bbox = cv2.selectROI(image, False)
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

    def isObjectSimilar(self, coordBox, coordPoint, limit):
        pointBox =(coordBox[0] - limit[0], coordBox[1] - limit[0] ) 
        pointBox2 = (coordBox[0] + limit[0], coordBox[1] + limit[1] )

        x = coordPoint[0]
        y = coordPoint[1]
        # dist = math.hypot(xB-xA,yB - yA)
        # print (limit, " = ",dist)
        
        if (pointBox[0] < x < pointBox2[0]) and (pointBox[1] < y < pointBox2[1]) :
            return True
        else:
            return False

    def initialCoord(self,y,limit):
        if y < limit:
            return True
        else :
            return False

    def set_object(self,image, obj, new_obj):
        
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 1,
                               qualityLevel = 0.1,
                               minDistance = 7,
                               blockSize = 4 )

        point = obj
        bbox = new_obj
        region = image[bbox[1]:bbox[1]+bbox[3] , bbox[0]:bbox[0]+bbox[2]]
         
        if len(point) == 0 :
            point = cv2.goodFeaturesToTrack(region, mask = None, **feature_params)
            if(point is not None):
                for i in xrange(len(point)):
                    point[i][0][0] = point[i][0][0] + bbox[0]
                    point[i][0][1] = point[i][0][1] + bbox[1]

        else :
            new_point = cv2.goodFeaturesToTrack(region, mask = None, **feature_params)
            if(new_point is not None):
                for i in xrange(len(new_point)):
                    new_point[i][0][0] = new_point[i][0][0] + bbox[0]
                    new_point[i][0][1] = new_point[i][0][1] + bbox[1]

                point = np.concatenate((point,new_point))

        return point
