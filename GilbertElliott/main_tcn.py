#!/usr/bin/env python
# coding: utf-8

# Description: This loads Matlab data files (train/valid/test) and
#    uses them to train and test a TCN.
#
# Requires this package to be installed -- https://github.com/philipperemy/keras-tcn

import tensorflow as tf
import numpy as np
import scipy.io as sio
from tcn import compiled_tcn

TRAIN_MODE = False                        # set to True if we need to train, set to False to load pre-trained model
TRAINDATA_FILE = 'data/train.mat'         # file containing training data set
VALIDDATA_FILE =  'data/valid.mat'        # file containing validation data set
TESTDATA_FILE_IN = 'data/test.mat'        # file containing test data set
TESTDATA_FILE_OUT = 'data/test_tcn.mat'   # file where predicted test data set values get saved
MODEL_FILENAME = 'saved_model.hdf5'       # file to load pre-trained model (if skipping training)
INPUT_LEN = 10                            # size of input vectors
FILTERS = 20                              # number of convolutional filters (or "hidden units")
KERNEL_SIZE = 4                           # size of the convolutional kernel
LAYERS = 3                                # number of layers, where 2^(LAYERS-1) is last dilation factor
STACKS = 2                                # number of stacks of residual blocks (also increases depth)
DROPOUT_RATE = 0.0                        # fraction of units to drop
LEARNING_RATE = 0.002                     # learning rate
BATCH_SIZE = 1024                         # size of each batch
EPOCHS = 25                               # number of epochs
# Receptive field is (2^LAYERS-1)*(KERNEL_SIZE-1)*STACKS+1... which should be greater than or equal to input vector length

# create model
model = compiled_tcn(return_sequences=False,
                     num_feat=2,
                     num_classes=0,
                     nb_filters=FILTERS,
                     kernel_size=KERNEL_SIZE,
                     dilations=[2 ** i for i in range(LAYERS)],
                     nb_stacks=STACKS,
                     max_len=INPUT_LEN,
                     output_len=2,
                     use_skip_connections=False,
                     regression=True,
                     dropout_rate=DROPOUT_RATE,
                     lr=LEARNING_RATE)
model.summary()

if TRAIN_MODE:
    # load training data
    matfile1 = sio.loadmat(TRAINDATA_FILE)
    x_train = matfile1['XX']
    y_train = matfile1['YY']
    
    # load validation data
    matfile2 = sio.loadmat(VALIDDATA_FILE)
    x_valid = matfile2['XX']
    y_valid = matfile2['YY']
    
    # train model, save weights if better than last best
    history = model.fit(x_train, y_train, 
                        validation_data=(x_valid, y_valid),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/weights-{epoch:04d}.hdf5', save_weights_only=True, save_best_only=True, monitor='val_loss'))
    # free up memory (maybe, relies on python's garbage collection)
    del matfile1, matfile2, x_train, y_train, x_valid, y_valid

else:  # load pre-trained model 
    model.load_weights(filepath=MODEL_FILENAME)

# load test data
matfile3 = sio.loadmat(TESTDATA_FILE_IN)
x_test = matfile3['XX']
y_test_true = matfile3['YY']

# compute outputs, save for processing in Matlab, and output test performance (summing x and y)
y_test_pred = model.predict(x_test)
sio.savemat(TESTDATA_FILE_OUT, {'YY':y_test_pred})
print(np.mean(np.square(y_test_true - y_test_pred))*2)

