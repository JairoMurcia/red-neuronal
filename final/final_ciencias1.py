# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 22:03:49 2019

@author: jairo
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import pickle
from tensorflow.keras.utils import to_categorical
import cv2
X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))


X=X/255.0

model=Sequential()
model.add(    Conv2D(64,(3,3),input_shape=X.shape[1:])      )
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X,y,batch_size=1,epochs=5,validation_split=0.1)


model.save('model.h5')



