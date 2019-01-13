#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras

from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import ExpectiMaxAgent,Agent2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
NUM_CLASSES = 4
NUM_EPOCHS = 20

display1 = Display()
display2 = IPythonDisplay()
input_shape = (4, 4, 12)
#定义模型
model=Sequential()
model.add(Conv2D(filters= 128, kernel_size=(4,1),kernel_initializer='he_uniform', padding='Same', activation='relu',input_shape=input_shape))

model.add(Conv2D(filters= 128, kernel_size=(1,4),kernel_initializer='he_uniform', padding='Same', activation='relu')) 

#model.add(Dropout(0.25)) 
model.add(Conv2D(filters= 128, kernel_size=(1,1),kernel_initializer='he_uniform', padding='Same', activation='relu')) 
model.add(Conv2D(filters= 128, kernel_size=(2,2),kernel_initializer='he_uniform', padding='Same', activation='relu')) 
model.add(Conv2D(filters= 128, kernel_size=(3,3),kernel_initializer='he_uniform', padding='Same', activation='relu')) 
model.add(Conv2D(filters= 128, kernel_size=(4,4),kernel_initializer='he_uniform', padding='Same', activation='relu'))  

model.add(Flatten()) 
model.add(BatchNormalization())
model.add(Dense(256, kernel_initializer='he_uniform',activation='relu')) 
model.add(BatchNormalization())
model.add(Dense(128, kernel_initializer='he_uniform',activation='relu')) 
model.add(Dense(4, activation='softmax')) 

model.summary()
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])
model.save('model.h5')

