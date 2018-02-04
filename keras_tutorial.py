#! /usr/bin/python

import numpy as np
np.random.seed(123) # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
# Neural network architecture imports

from keras.datasets import mnist
# Dataset import

from matplotlib import pyplot as plt
# plotting library

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print X_train.shape
# (60000, 28, 28)
# 60k images, with dimensions 28x28 pixels

# plt.imshow(X_train[0])
# plt.show()
# plt.imshow(X_test[0])
# plt.show()
# Sanity check for ensuring data is represented correctly

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# Reshape the data to have a depth channel of 1 (Greyscale), instead of 3 (RGB)

# print X_train.shape
# (60000, 1, 28, 28)
# 60k images, with depth of 1 and dimensions 28x28 pixels

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_train /= 255
# Normalize data values to the range of [0, 1]

# print y_train.shape
# (60000,)
# Should have 10 different classes, but only have 1-dimensional array
# print y_train[:10]
# [5 0 4 1 9 2 1 3 1 4]

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# Convert 1-dimensional class arrays to 10-dimensional class matrices

# print Y_train[:10]
# [[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
# [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
# [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
# [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
# [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
# [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
# [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
# [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]]

#print Y_train.shape
# (60000, 10)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
# Declare input layer
# 32 -> number of convolution filters to use
# 3 -> number of rows in each convolution kernel
# 3 -> number of columns in each convolution kernel

print model.output_shape

