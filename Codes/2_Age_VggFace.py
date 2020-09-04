#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import tensorflow as tf
import collections
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import optimizers
from os import listdir,makedirs
from os.path import isfile,join
from keras.models import Model
import json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras_vggface
from keras_vggface.vggface import VGGFace
from keras.models import model_from_json
import h5py
import keras
from keras.models import load_model
from keras.models import Input
from keras.optimizers import Nadam, Adam
import os
from pathlib import Path
import gdown
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
# from deepface import DeepFace
#from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace


# In[28]:


os.chdir('/home/sdmishra/deep_learning/input_images')
# os.chdir("C:/Sapna/Graham/Deep Learning/Projects/Face Detection/UTKFace/Input Images")
# os.chdir("C:/Sapna/Graham/Deep Learning/Projects/Codes/Sample/Final Demo Images")


# In[29]:


onlyfiles = os.listdir()
len(onlyfiles)


# In[30]:


shuffle(onlyfiles)


# In[34]:


shuffle(onlyfiles)
age = [i.split('_')[0] for i in onlyfiles]


# In[35]:


classes =[]
for i in age:
    i = int(i)
    if (i>0) and (i<=2):
        classes.append(0)
    if (i>2) and (i<=5):
        classes.append(1)
    if (i>5) and (i<=10):
        classes.append(2)
    if (i>10) and (i<=15):
        classes.append(3)
    if (i>15) and (i<=20):
        classes.append(4)
    if (i>20) and (i<=25):
        classes.append(5)
    if (i>25) and (i<=30):
        classes.append(6)
    if (i>30) and (i<=35):
        classes.append(7)
    if (i>35) and (i<=40):
        classes.append(8)
    if (i>40) and (i<=50):
        classes.append(9)
    if (i>50) and (i<=59):
        classes.append(10)
    if (i>=60):
        classes.append(11)


# In[36]:


collections.Counter(classes)


# In[37]:


X_data =[]
for file in onlyfiles:
    face = cv2.imread(file)
    face = cv2.resize(face, (224,224))
    X_data.append(face)


# In[38]:


categorical_labels = to_categorical(classes, num_classes=12)
categorical_labels[:12]


# In[55]:


(x_train, y_train), (x_test, y_test) = (X_data[:17000],categorical_labels[:17000]) , (X_data[17000:] , categorical_labels[17000:])
(x_valid , y_valid) = (x_test[:3000], y_test[:3000])
(x_test, y_test) = (x_test[3000:], y_test[3000:])

#(x_train, y_train), (x_test, y_test) = (X_data[:5],categorical_labels[:5]) , (X_data[5:] , categorical_labels[5:])
#(x_valid , y_valid) = (x_test[:3], y_test[:3])
#(x_test, y_test) = (x_test[3:], y_test[3:])


# In[56]:


x_train = np.array(x_train).reshape(-1, 224, 224, 3)/255
x_test  = np.array(x_test).reshape(-1, 224, 224, 3)/255
x_valid = np.array(x_valid).reshape(-1, 224, 224, 3)/255
print(x_train.shape)
print(x_test.shape)
print(x_valid.shape)


# In[57]:


datagen = ImageDataGenerator(horizontal_flip=True)

datagen.fit(x_train)

for X_batch, y_batch in datagen.flow(x_train, y_train,batch_size=5):
    break
        
new_train_x=np.append(x_train,X_batch,axis=0)
len(new_train_x)

new_train_y=np.append(y_train,y_batch,axis=0)
len(new_train_y)

collections.Counter(np.argmax(new_train_y, axis=1))


# In[60]:


datagen = ImageDataGenerator(brightness_range=[0.2,1.0])

datagen.fit(x_train)

for X_batch, y_batch in datagen.flow(x_train, y_train,batch_size=5000):
    break
    
new_train_x1 = np.append(new_train_x,X_batch,axis=0)
len(new_train_x1)

new_train_y1 = np.append(new_train_y,y_batch,axis=0)
len(new_train_y1)

collections.Counter(np.argmax(new_train_y1, axis=1))


# In[61]:


datagen = ImageDataGenerator(zoom_range=[0.5,1.0])

datagen.fit(x_train)

for X_batch, y_batch in datagen.flow(x_train, y_train,batch_size=5000):
    break
    
new_train_x2 = np.append(new_train_x1,X_batch,axis=0)
len(new_train_x2)

new_train_y2 = np.append(new_train_y1,y_batch,axis=0)
len(new_train_y2)

collections.Counter(np.argmax(new_train_y2, axis=1))


# In[62]:


model = VGGFace(model='resnet50',include_top=False, input_shape = (224,224, 3))
# model.summary()
# len(model.layers)
# print('Inputs: %s' % model.inputs)
# print('Outputs: %s' % model.outputs)
# len(model.layers)


# In[63]:


model.layers.pop()
# len(model.layers)
# model.summary()


# In[64]:


num_classes =12
for layer in model.layers[:160]:
    layer.trainable = False


# In[65]:


#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)


# In[66]:


# creating the final model 
model_final = Model(inputs = model.input, output = predictions)


# In[67]:


# model_final.summary()
# len(model_final.layers)


# In[68]:


model_final.compile(loss = "categorical_crossentropy",optimizer ='adagrad',metrics=["accuracy"])


# In[69]:


callbacks = [EarlyStopping(monitor='val_loss',mode='min', patience=10)]


# In[79]:


history = model_final.fit(new_train_x2, # Features
          new_train_y2, # Target vector
          epochs=100, # Number of epochs
          callbacks=callbacks, # Early stopping
          verbose=1,   # Print description after each epoch        
          batch_size=100, # Number of observations per batch
          validation_data=(x_valid, y_valid), # Data for evaluation
         )


# In[71]:

os.chdir('/home/sdmishra/deep_learning/model_weights')
#os.chdir("C:/Sapna/Graham/Deep Learning/Projects/Face Detection/UTKFace/test_tl")
# serialize model to JSON
model_json = model_final.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_final.save_weights("model.h5")
print("Saved model to disk")


# In[72]:


# load json and create model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# In[76]:


#Train Model
print("Train Loss and Accuracy",model_final.evaluate(new_train_x2, new_train_y2,verbose=0))

#Test Model
print("Test Loss and Accuracy",model_final.evaluate(x_test, y_test,verbose=0))


# In[85]:


print("Loss Details",history.history)


# In[104]:


train_classes = model_final.predict(new_train_x2, verbose=0)
train_classes1 = np.argmax(train_classes,axis=1)
#train_classes1


# In[105]:


test_classes = model_final.predict(x_test, verbose=0)
test_classes1 = np.argmax(test_classes,axis=1)
#test_classes1


# In[111]:


print("Train Accuracy:",accuracy_score(new_train_y2, train_classes))


# In[112]:


print("Test Accuracy:",accuracy_score(y_test, test_classes))

