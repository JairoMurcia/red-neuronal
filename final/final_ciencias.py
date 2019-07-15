# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:42:28 2019

@author: jairo
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
IMG_SIZE=50
datadir='caras/'
categories=['feliz','sorprendido','triste']
training_data=[]

def create_training_data():

    for category in categories:
        path=os.path.join(datadir,category)
        class_num=categories.index(category)
        for img in os.listdir(path):
            try:
                imgArray=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                newArray=cv2.resize(imgArray,(IMG_SIZE,IMG_SIZE))
                training_data.append([newArray,class_num])
            except Exception as e:    
                pass
            
create_training_data() 

random.shuffle(training_data)
X=[]
y=[]
for features,labels in training_data:
   X.append(features)
   y.append(labels)
   
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1) 

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()


pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
  