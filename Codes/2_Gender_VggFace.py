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
import collections
import gdown
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation
# from deepface import DeepFace
#from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace


# In[2]:


np.random.seed(7)


# In[3]:


os.chdir('/home/sdmishra/deep_learning/input_images')
#os.chdir("C:/Sapna/Graham/Deep Learning/Projects/Face Detection/UTKFace/test_tl")


# In[4]:


onlyfiles = os.listdir()


# In[5]:


len(onlyfiles)


# In[6]:


shuffle(onlyfiles)
gender = [i.split('_')[1] for i in onlyfiles]


# In[7]:


classes = []
for i in gender:
    #i = int(i)
    classes.append(i)


# In[8]:


#Check for data imbalance
collections.Counter(classes)


# In[9]:


X_data =[]
for file in onlyfiles:
    face = cv2.imread(file)
    face = cv2.resize(face, (224,224) )
    X_data.append(face)


# In[11]:


categorical_labels = to_categorical(classes, num_classes=2)


# In[12]:


(x_train, y_train), (x_test, y_test) = (X_data[:17000],categorical_labels[:17000]) , (X_data[17000:] , categorical_labels[17000:])
(x_valid , y_valid) = (x_test[:3000], y_test[:3000])
(x_test, y_test) = (x_test[3000:], y_test[3000:])


# In[13]:


x_train = np.array(x_train).reshape(-1, 224, 224, 3) /255
x_valid = np.array(x_valid).reshape(-1, 224, 224, 3)/255
x_test  = np.array(x_test).reshape(-1, 224, 224, 3)/255
print(x_train.shape)
print(x_test.shape)
print(x_valid.shape)


# In[14]:


num_classes = 2
class_names = ['Female','Male']


# In[28]:


model = VGGFace(model='resnet50',include_top=False, input_shape = (224,224, 3))
model.summary()
len(model.layers)

print('Inputs: %s' % model.inputs)
print('Outputs: %s' % model.outputs)


# In[29]:


model.layers.pop()
# len(model.layers)
model.summary()


# In[30]:


num_classes =2
for layer in model.layers[:168]:
    layer.trainable = False


# In[31]:


#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)


# In[32]:


# creating the final model 
model_final = Model(inputs = model.input, output = predictions)


# In[33]:


model_final.summary()


# In[24]:


model_final.compile(loss = "binary_crossentropy",optimizer ='adagrad',metrics=["accuracy"])


# In[25]:


callbacks = [EarlyStopping(monitor='val_loss',mode='min', patience=10)]


# In[27]:


history = model_final.fit(x_train, # Features
          y_train, # Target vector
          epochs=100, # Number of epochs
          callbacks=callbacks, # Early stopping
          verbose=1,   # Print description after each epoch        
          batch_size=100, # Number of observations per batch
          validation_data=(x_valid, y_valid), # Data for evaluation
         )
#latest train loss:0.199 ; model accuracy:92 ;valloss: 0.24 ;val_accuracy:90


# In[ ]:


print(model_final.evaluate(x_test, y_test))


# In[ ]:


os.chdir('/home/sdmishra/deep_learning/model_weights')

# serialize model to JSON
model_json = model_final.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_final.save_weights("model.h5")
print("Saved model to disk")


# In[ ]:


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#tf.keras.models.model_from_json(json)
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

