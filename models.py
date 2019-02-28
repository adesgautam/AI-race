
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import adam, SGD
from keras.layers import BatchNormalization

def NVModel(classes, input_shape):
	dropout = 0.5

	model = Sequential()

	# model.add(BatchNormalization(epsilon=0.001, axis=1, input_shape=input_shape))
	model.add(Convolution2D(24,5,5, input_shape=input_shape, border_mode='valid', activation='relu', subsample=(2,2)))
	# model.add(Dropout(dropout, name='drop1'))
	model.add(Convolution2D(36,5,5, border_mode='valid', activation='relu', subsample=(2,2)))
	# model.add(Dropout(dropout, name='drop2'))
	model.add(Convolution2D(48,5,5, border_mode='valid', activation='relu', subsample=(2,2)))
	# model.add(Dropout(dropout, name='drop3'))
	model.add(Convolution2D(64,3,3, border_mode='valid', activation='relu', subsample=(1,1)))
	# model.add(Dropout(dropout, name='drop4'))
	model.add(Convolution2D(64,3,3, border_mode='valid', activation='relu', subsample=(1,1)))
	model.add(Flatten())
	model.add(Dense(1164, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(classes, activation='softmax', name='predictions'))
	opt = adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model


def Model1(classes, input_shape):
	model = Sequential()

	model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=input_shape, name='conv1'))
	# model.add(Dropout(dropout, name='drop1'))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv2'))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv3'))
	model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

	#model.add(Dropout(dropout, name='drop2'))
	model.add(Flatten(name='flatten'))
	model.add(Dense(128, activation='relu', name='fc'))
	#model.add(Dropout(dropout, name='drop3'))
	model.add(Dense(classes, activation='softmax', name='predictions'))
	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


def Model2(classes, input_shape):
	dropout = 0.1

	model = Sequential()
	model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape, name='conv1'))
	model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
	model.add(Dropout(dropout, name='drop1'))
	model.add(Convolution2D(32, 3, 3, activation='relu', name='conv2'))
	model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
	model.add(Dropout(dropout, name='drop2'))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv3'))
	model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
	# model.add(Dropout(dropout, name='drop3'))

	# model.add(Convolution2D(64, 3, 3, activation='relu', name='conv4'))
	# model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))

	model.add(Flatten(name='flatten'))
	model.add(Dense(128, activation='relu', name='fc'))
	# model.add(Dropout(dropout, name='drop'))
	model.add(Dense(classes, activation='softmax', name='predictions'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


def Model3(classes, input_shape):
	model = Sequential()
	model.add(Convolution2D(filters = 32, kernel_size = (3, 3), input_shape = input_shape, padding='same'))
	model.add(Activation('relu'))
	# model.add(BatchNormalization())
	model.add(Convolution2D(filters = 32, kernel_size = (3, 3)))
	model.add(Activation('relu'))
	# model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(filters = 64, kernel_size = (3, 3)))
	model.add(Activation('relu'))
	# model.add(BatchNormalization())
	model.add(Convolution2D(filters = 64, kernel_size = (3, 3)))
	model.add(Activation('relu'))
	# model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	# model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	# model.add(Dropout(0.25))
	model.add(Dense(classes))
	model.add(Activation('softmax'))

	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


