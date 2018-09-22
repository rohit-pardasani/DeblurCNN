#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:14:48 2018

@author: mig128gb1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:02:52 2018

@author: mig-imac-new
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:15:32 2018

@author: mig-imac-new
"""

# Creating and training model

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
import os


def scaled_mse(y_true, y_pred):
    return 1000000*keras.losses.mean_squared_error(y_true,y_pred)

class MicroscopyCNNTrainSystem:
    def __init__(self):
        print('Constructor Called')
        self.IMAGE_WIDTH = 35
        self.IMAGE_HEIGHT = 35
        self.CHANNELS = 3
        self.N_SAMPLES = 480000
        self.N_TRAIN_SAMPLES = 400000
        self.N_EVALUATE_SAMPLES = 80000
        self.folderName = './DeblurData/'
        self.X_TRAIN = np.zeros((self.N_TRAIN_SAMPLES,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS))
        self.Z_TRAIN = np.zeros((self.N_TRAIN_SAMPLES,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS))
        self.X_EVALUATE = np.zeros((self.N_EVALUATE_SAMPLES,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS))
        self.Z_EVALUATE = np.zeros((self.N_EVALUATE_SAMPLES,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.CHANNELS))
        
        j=0
        for i in range(self.N_SAMPLES):
            if(i%1000 == 0):
                print('train data loading '+str(i))
            if((i+1)%6 != 0):
                pathr = self.folderName+'X'+str(i+1)+'.mat'
                x = sio.loadmat(pathr)
                self.X_TRAIN[j,:,:,:] = x['X']
                patht = self.folderName+'Z'+str(i+1)+'.mat'
                z = sio.loadmat(patht)
                self.Z_TRAIN[j,:,:,:] = z['Z']
                j=j+1


        j=0
        for i in range(self.N_SAMPLES):
            if(i%1000 == 0):
                print('validation data loading '+str(i))
            if((i+1)%6 == 0):
                pathr = self.folderName+'X'+str(i+1)+'.mat'
                x = sio.loadmat(pathr)
                self.X_EVALUATE[j,:,:,:] = x['X']
                patht = self.folderName+'Z'+str(i+1)+'.mat'
                z = sio.loadmat(patht)
                self.Z_EVALUATE[j,:,:,:] = z['Z']
                j=j+1
	


    def freshModelMaker(self, optim):
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
        self.myModel.compile(loss='mean_squared_error',metrics=[scaled_mse],optimizer=optim)
        print("Fresh model Created")
        self.myModel.summary()
            
    def loadPrevModel(self,modelFileToLoad):
        self.myModel = keras.models.load_model(modelFileToLoad)
        self.myModel.summary()
        
    def trainModelAndSaveBest(self, BATCH_SIZE, EPOCHS, modelFileToSave):
        myCallback = callbacks.ModelCheckpoint(modelFileToSave, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        trainHistory = self.myModel.fit(x=self.X_TRAIN, y=self.Z_TRAIN, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[myCallback], validation_data=(self.X_EVALUATE,self.Z_EVALUATE))
        return trainHistory
        
    def reCompileModel(self,optim):
        self.myModel.compile(loss='mean_squared_error', metrics=[scaled_mse],optimizer=optim)
        

os.environ["CUDA_VISIBLE_DEVICES"]="2"            
mcts = MicroscopyCNNTrainSystem();
#myOptimizer = optimizers.SGD(lr=0.002)
myOptimizer = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
mcts.freshModelMaker(myOptimizer)
#mcts.reCompileModel(myOptimizer)
#mcts.loadPrevModel('somefile')
myModelHistory = mcts.trainModelAndSaveBest(100,100,'DeblurNetDnCNNreluV1.h5')
#plt.plot(myModelHistory.history['loss'])
#plt.show()
#plt.plot(myModelHistory.history['val_loss'])
#plt.show()
