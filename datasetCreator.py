import os
import cv2
import numpy as np

class User:
	def __init__(self, name):
		if not os.path.exists("dataset"):
			os.mkdir("dataset")
		self.name = name

	#create dataset of user faces and save it in dataset folder
	def save(self):
		face_cascade = cv2.CascadeClassifier('/home/amar/Downloads/face/haarcascade_frontalface_default .xml')
		cap = cv2.VideoCapture(0)
		sampleNumber = 0
		while True:
			print(sampleNumber)
			ret, img = cap.read()
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray,1.3,5)
			for(x,y,w,h) in faces:
				sampleNumber += 1
				cv2.imwrite("dataset/"+name+"."+str(sampleNumber)+".jpg",gray[y:y+h,x:x+w])
				cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
				cv2.waitKey(100)
			cv2.imshow('img',img)
			cv2.waitKey(1)
			if(sampleNumber>20):
				break
		cap.release()
		cv2.destroyAllWindows()

name = input('Enter Name: ')
u1 = User(name)
u1.save()
