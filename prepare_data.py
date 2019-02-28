from PIL import Image
import os
import numpy as np
import pickle

# Preparing images
img_rows, img_cols = 32, 32

data = open("training_data1.csv", 'r').read().split('\n')
path = 'pics1'
# files = os.listdir(path) 
files = []
for i in data:
	files.append(i.split(',')[0])
	
print("files:", len(files))
# num_samples = size(files)

immatrix = []

print("Processing images...")
for file in files:
	# image = Image.open(path + '//' + file)   
	image = Image.open(file)   
	area = (0, 270, 650, 550)
	cropped_img = image.crop(area)	
	img = cropped_img.resize((img_rows,img_cols))
    # for greyscale
	# gray = img.convert('L')
	immatrix.append(np.array(img).flatten())
	# gray.save(path2 +'//' +  file, "JPEG")

immatrix = np.array(immatrix)
print(immatrix.shape)

print("Saving image data...")
X = open("X_images_32_rgb.pickle","wb")
pickle.dump(immatrix, X)
X.close()
print("Image data saved!")


print("Processing labels...")

# Preparing Labels
l_data = open("training_data1.csv", 'r').read().split('\n')
print(len(l_data))

labs = []
for l in l_data:
	l = l.split(',')
	labs.append(l[1])

# One-hot encoding of labels
# [UP, UP_LEFT, UP_RIGHT]
labels = []
for l in labs:
	if l == " UP ":
		tmp = [1,0,0]
	elif "UP" in l and "LEFT" in l:
		tmp = [0,1,0]
	elif "UP" in l and "RIGHT" in l:
		tmp = [0,0,1]
	else:
		tmp = [0,0,0]
		print(l)
	
	labels.append(tmp)

labels = np.array(labels)
print("labels:", labels.shape)

print("Saving labels...")
Y = open("Y_labels1.pickle","wb")
pickle.dump(labels, Y)
Y.close()

print("Labels saved!")





