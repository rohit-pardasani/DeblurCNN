#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:01:11 2018

@author: mig-imac-new
"""
# Run the network on first 10 figures and show the result
import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.initializers import RandomNormal
from keras import backend as Ks
from keras import optimizers
from keras import callbacks
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio

def scaled_mse(y_true, y_pred):
    return 1000000*keras.losses.mean_squared_error(y_true,y_pred)

class MicroscopyCNNTestSystem:
    
    def __init__(self):
        print('Constructor Called')
        self.IMAGE_WIDTH = 481
        self.IMAGE_HEIGHT = 321
        self.CHANNELS = 3
        self.N_TEST_SAMPLES = 20
        self.folderName = './MyDatasetTest/'
        self.X_TEST = np.zeros((self.N_TEST_SAMPLES,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS))
        for i in range(self.N_TEST_SAMPLES):
            pathr = self.folderName+'Qb'+str(i+1)+'.jpg'
            I = imageio.imread(pathr)
            I = I.astype(np.float32)/255
            self.X_TEST[i,:,:,:] = I
        
    def loadModelwithChangedInput(self,modelFileToLoad):
        self.savedModel = keras.models.load_model(modelFileToLoad, custom_objects={'scaled_mse': scaled_mse})
        self.savedModel.summary()
        self.N_LAYERS = 20
        self.F = 64
        self.myModel = Sequential()
        firstLayer = Convolution2D(self.F, (3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS), use_bias=True, bias_initializer='zeros')
        self.myModel.add(firstLayer)
        self.myModel.add(Activation('relu'))
        for i in range(self.N_LAYERS-2):
            Clayer = Convolution2D(self.F, (3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.F), use_bias=True, bias_initializer='zeros')
            self.myModel.add(Clayer)
            Blayer = BatchNormalization(axis=-1, epsilon=1e-3)
            self.myModel.add(Blayer)
            self.myModel.add(Activation('relu'))
        lastLayer = Convolution2D(self.CHANNELS, (3, 3), strides=(1, 1), kernel_initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None), padding='same', input_shape=(self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.F), use_bias=True, bias_initializer='zeros')
        self.myModel.add(lastLayer)    
        self.myModel.set_weights(self.savedModel.get_weights())
        print("Fresh model with changed size created")
        self.myModel.summary()
        
    def runModelAndSaveImages(self):
        #myOptimizer = optimizers.SGD(lr=0.002)
        myOptimizer = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.myModel.compile(loss='mean_squared_error', metrics=[scaled_mse],optimizer=myOptimizer)
        self.Z_TEST = self.myModel.predict(self.X_TEST,batch_size=1, verbose=1)
        print('output predicted')
        self.Y_TEST = self.X_TEST - self.Z_TEST
        self.Y_TEST = np.clip(self.Y_TEST,0.0,1.0)
        for i in range(self.N_TEST_SAMPLES):
            patht = self.folderName+'Y'+str(i+1)+'.jpg'
            I = self.Y_TEST[i,:,:,:]
            I = I*255
            I = I.astype(np.uint8)
            imageio.imsave(patht, I)
            

mctts = MicroscopyCNNTestSystem();
mctts.loadModelwithChangedInput('DeblurNetDnCNNreluV1_best_loss_1593.h5')
mctts.runModelAndSaveImages()

            
            
            
        
    
        
        
        
    
    
