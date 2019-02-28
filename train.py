
from keras.utils import np_utils
from models import NVModel, Model1, Model2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import pickle

# Load data
file = open("X_images1.pickle", 'rb')
immatrix = pickle.load(file)
file.close()
print("X shape:", immatrix.shape)

file = open("Y_labels1.pickle", 'rb')
labels = pickle.load(file)
file.close()
print("Y shape:", labels.shape)


# SETUP training
img_rows, img_cols = 64, 64
nb_classes = 5
batch_size = 64
nb_epoch   = 10

X, y = shuffle(immatrix, labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test  = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print('X_test shape:',  X_test.shape)
print('Y_train shape:', y_train.shape)
print('Y_test shape:',  y_test.shape)

# Load Model
# model = NVModel(nb_classes, (img_rows, img_cols, 1))
model = Model2(nb_classes, (img_rows, img_cols, 1))

# train
hist = model.fit(X_train, y_train,
				batch_size=batch_size, epochs=nb_epoch,
				verbose=1, validation_data=(X_test, y_test))


# # Save Model
# model.save("model2.h5")
# print("Saved model to disk")

# # Check Predictions
# img = Image.open("pics/9922.png")
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


# # Plot model accuracy
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

plot_model_history(hist)














