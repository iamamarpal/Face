import os
import cv2
import numpy as np
import pandas as pd
import datetime as dt
from PIL import Image

class recognize():
    def __init__(self):
        self.fontface = cv2.FONT_HERSHEY_SIMPLEX
        self.fontscale = 1
        self.fontcolor = (0, 0, 255)
        self.userData = pd.read_pickle(r'labelData.pickle')
        self.exit = False

    def rec(self, cap):
        face_cascade = cv2.CascadeClassifier('/home/amar/Downloads/face/haarcascade_frontalface_default .xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("face-trainner.yml")
        iter = 0
        while True:
            det = False
            Id = -1
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for(x,y,w,h) in faces:
                det = True
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                Id, distance = recognizer.predict(gray[y:y+h,x:x+w])
                if distance < 50:
                    iter = 0
                    cv2.putText(img,self.userData[Id],(x,y+h),self.fontface,self.fontscale,self.fontcolor)
                    print("Hello " + self.userData[Id])
                else:
                    cv2.putText(img,"Unknown",(x,y+h),self.fontface,self.fontscale,self.fontcolor)
                    print("New Member")
                    iter += 1
            if iter>5:
                print("New Member")
            cv2.imshow('img',img)
            k = cv2.waitKey(50)
            if k == ord('q') or self.exit:
                break

cap = cv2.VideoCapture(0)
rec = recognize()
rec.rec(cap)
cap.release()
cv2.destroyAllWindows()