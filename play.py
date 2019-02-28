
from keras.utils import np_utils
from keras.models import load_model

import numpy as np
from PIL import Image
import cv2

import mss
import mss.tools

import pyautogui
import time, random

# monitor = {"top": 30, "left": 50, "width": 650, "height": 550}
monitor = {"top": 133, "left": 50, "width": 514, "height": 480}

# model = load_model('models/model1_8034_3c.h5')
model = load_model('models/model1_8104_3c_rgb_32.h5')
print("Loaded model from disk")

img_rows, img_cols = 32, 32

def straight():
	pyautogui.keyDown('a')
	pyautogui.keyUp('left')
	pyautogui.keyUp('right')

def left():
	if random.randrange(0,3) == 1:
		pyautogui.keyDown('a')
	else:
		pyautogui.keyUp('a')
	pyautogui.keyDown('left')
	pyautogui.keyUp('right')

def right():
	if random.randrange(0,3) == 1:
		pyautogui.keyDown('a')
	else:
		pyautogui.keyUp('a')
	pyautogui.keyDown('right')
	pyautogui.keyUp('a')

def forward_left():
	pyautogui.keyDown('a')
	pyautogui.keyDown('left')
	pyautogui.keyUp('right')

def forward_right():
	pyautogui.keyDown('a')
	pyautogui.keyDown('right')
	pyautogui.keyUp('left')

def no_keys():
	pyautogui.keyUp('a')
	pyautogui.keyUp('right')
	pyautogui.keyUp('left')	


i=0
print("Starting to play...")
while True:
	with mss.mss() as sct:
		sct_img = sct.grab(monitor)
		img = np.array(sct_img)[270:650, 0:]
		# cv2.imshow('op', img)

		img = Image.fromarray(img).convert('RGB')
		# area = (0, 270, 650, 550)
		# img = img.crop(area)
		img = img.resize((img_rows, img_cols))
		#.convert('L')
		cv2.imwrite("op/"+str(i)+".jpg", np.array(img))

		img_matrix = np.array([ np.array(img).flatten() ], 'f')
		image = img_matrix.reshape(1, img_rows, img_cols, 3)
		image = image.astype('float32')
		image /= 255

		pred = model.predict(image)
		prediction = np.argmax(pred)

		if prediction == 0:
			# pyautogui.press('a')
			# straight()
			print(prediction, 'a')
		# elif prediction == 1:
		# 	# pyautogui.press('left')
		# 	left()
		# 	print(prediction, 'left')
		# elif prediction == 2:
		# 	# pyautogui.press('right')
		# 	right()
		# 	print(prediction, 'right')
		elif prediction == 1:
			# pyautogui.press(['a', 'left'])
			# forward_left()
			print(prediction, 'a', 'left')
		elif prediction == 2:
			# pyautogui.press(['a', 'right'])
			# forward_right()
			print(prediction, 'a', 'right')
		else:
			print("None")

		i+=1



















# img = Image.open("pics/9933.png")
# with mss.mss() as sct:
# 	sct_img = sct.grab(monitor)
# 	mss.tools.to_png(sct_img.rgb, sct_img.size, output='pic.png')
# area = (0, 270, 650, 550)
# img = img.crop(area).resize((img_rows, img_cols)).convert('L')

# img_matrix = np.array([ np.array(img).flatten() ], 'f')
# image = img_matrix.reshape(1, img_rows, img_cols, 1)
# image = image.astype('float32')
# image /= 255

# pred = model.predict(image)
# prediction = np.argmax(pred)
# print(pred, prediction)

# if prediction == 0:
# 	print("prediction: UP")
# elif prediction == 1:
# 	print("prediction: LEFT")
# elif prediction == 2:
# 	print("prediction: RIGHT")
# elif prediction == 3:
# 	print("prediction: UP LEFT")
# elif prediction == 4:
# 	print("prediction: UP RIGHT")



















