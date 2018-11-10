#!/usr/bin/env python

from __future__ import print_function
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Add
from keras.layers import Conv2D, Input, add
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

import os, threading
from scipy.misc import imread, imresize
import numpy as np

DATA_X_PATH = "./data/x/"
DATA_Y_PATH = "./data/y/"
TARGET_IMG_SIZE = (128, 128, 3)
BATCH_SIZE = 64
EPOCHS = 100


def tf_log10(x):
	numerator = tf.log(x)
	denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator


def load_images(directory):
	img = imread(directory)
	img = np.array(img) / 127.5 - 1.

	return img

def get_image_list(data_path):
	l = os.listdir(data_path)
	train_list = []
	for f in l:
		if f[-4:] == '.jpg':
			train_list.append(f)

	return train_list

def get_image_batch(target_list, offset):
	target = target_list[offset:offset+BATCH_SIZE]
	batch_x = []
	batch_y = []
	for t in target:
		x = imread(os.path.join(DATA_X_PATH, t))
		y = imread(os.path.join(DATA_Y_PATH, t))
		x = imresize(x, size=TARGET_IMG_SIZE[:2], interp='bicubic')
		batch_x.append(x)
		batch_y.append(y)
	batch_x = np.array(batch_x) / 127.5 - 1.
	batch_y = np.array(batch_y) / 127.5 - 1.
	return batch_x, batch_y

class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()


def image_gen(target_list):
	while True:
		for step in range(len(target_list)//BATCH_SIZE):
			offset = step*BATCH_SIZE
			batch_x, batch_y = get_image_batch(target_list, offset)
			yield (batch_x, batch_y)

# evaluate methods
def PSNR(y_true, y_pred):
	max_pixel = 1.0
	return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def SSIM(y_true, y_pred):
	max_pixel = 1.0
	return tf.image.ssim(y_pred, y_true, max_pixel)

# Get the training and testing data
img_list = get_image_list(DATA_X_PATH)
num_train_imgs = int(len(img_list)* 0.7)
train_list = img_list[:num_train_imgs]
test_list = img_list[num_train_imgs:]

input_img = Input(shape=TARGET_IMG_SIZE)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model_0 = Activation('relu')(model)

total_conv = 22  # should be even number
total_conv -= 2  # subtract first and last
residual_block_num = 5  # should be even number

for _ in range(residual_block_num):  # residual block
	model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model_0)
	model = Activation('relu')(model)
	for _ in range(int(total_conv/residual_block_num)-1):
		model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
		model = Activation('relu')(model)
		model_0 = add([model, model_0])

model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
res_img = model
output_img = add([res_img, input_img])

model = Model(input_img, output_img)

# model.load_weights('./checkpoints/weights-improvement-20-26.93.hdf5')

adam = Adam(lr=0.0001)
sgd = SGD(lr=1e-5, momentum=0.9, decay=1e-5, nesterov=False)
model.compile(adam, loss='psnr', metrics=[PSNR, "accuracy"])

model.summary()

with open('./model/vdsr_architecture.json', 'w') as f:
	f.write(model.to_json())

filepath="./checkpoints/weights-improvement-{epoch:02d}-{PSNR:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor=PSNR, verbose=1, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(image_gen(train_list), steps_per_epoch=len(train_list) // BATCH_SIZE, \
					validation_data=image_gen(test_list), validation_steps=len(test_list) // BATCH_SIZE, \
					epochs=EPOCHS, workers=8, callbacks=callbacks_list)

print("Done training!!!")

print("Saving the final model ...")

model.save('./model/vdsr_best_weights.h5')  # creates a HDF5 file
del model  # deletes the existing model
