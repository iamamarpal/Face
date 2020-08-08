import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

images = os.listdir(image_dir)
# print(images)
id = 0
user_info = {}
for image in images:
	inDict = False
	path = os.path.join(image_dir, image)
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	label = image[:image.index(".")]
	if not user_info:
		user_info[id] = label
	for key, value in user_info.items():
		if label == value:
			id = key
			inDict = True
			break
		else:
			inDict = False
	if not inDict:
		id = id+1
		user_info[id] = label
	# faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
	# for (x,y,w,h) in faces:
		# roi = image_array[y:y+h, x:x+w]
	x_train.append(img)
	y_labels.append(id)

# print(x_train)
# print(y_labels)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("face-trainner.yml")

with open('labelData.pickle', 'wb') as handle:
	pickle.dump(user_info, handle, protocol = pickle.HIGHEST_PROTOCOL)
print(user_info)
