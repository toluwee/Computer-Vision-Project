#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
import umap
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import optimizers
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import ops
from keras.callbacks import TensorBoard
from numpy import asarray
import os,glob
from os import listdir,makedirs
from os.path import isfile,join
import collections 
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report, auc
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


# In[8]:


np.random.seed(7)


# Convert Images into Grey Scale

# In[9]:


path = 'C:/Sapna/Graham/Deep Learning/Projects/Face Detection/UTKFace/Input Images' # Source Folder
dstpath = 'C:/Sapna/Graham/Deep Learning/Projects/Face Detection/UTKFace/Output Images' # Destination Folder
try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in same folder")
# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))] 
for image in files:
    try:
        img = cv2.imread(os.path.join(path,image))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,gray)
    except:
        print ("{} is not converted".format(image))
for fil in glob.glob("*.jpg"):
    try:
        image = cv2.imread(fil) 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to greyscale
        cv2.imwrite(os.path.join(dstpath,fil),gray_image)
    except:
        print('{} is not converted')


# In[ ]:


os.chdir('C:/Sapna/Graham/Deep Learning/Projects/Face Detection/UTKFace/')
os.chdir('Output Images')


# In[10]:


im =Image.open('1_0_0_20161219140623097.jpg.chip.jpg').resize((128,128))
im


# In[11]:


onlyfiles = os.listdir()


# In[12]:


len(onlyfiles)


# In[13]:


shuffle(onlyfiles)
race = [i.split('_')[2] for i in onlyfiles]


# 1. I would like to make clear that the image data is in its name means the first box of the second cell, the second gender, the second one, so the first step is that we are trying to separate the labels from the images so that they are stored in the classes as much as we need them
# 2. We can split the data into Gender Classes - 0 Male 1 Female
# 

# In[14]:


#Check for data imbalance
classes = []
for i in race:
    #i = int(i)
    classes.append(i)


# In[15]:


#Check for data imbalance
collections.Counter(classes)


# In[16]:


# model = load_model('C:/Sapna/Graham/Deep Learning/Projects/Codes/Transfer Learning/facenet_keras.h5')
# print('Loaded Model')


# **CONVERT IMAGES TO VECTORS**

# In[17]:


X_data =[]
for file in onlyfiles:
    face = cv2.imread(file)
    face = cv2.resize(face, (128,128) )
    X_data.append(face)


# In[18]:


classes[:10]


# In[19]:


categorical_labels = to_categorical(classes, num_classes=5)


# In[20]:


categorical_labels[:10]


# In[21]:


(x_train, y_train), (x_test, y_test) = (X_data[:17000],categorical_labels[:17000]) , (X_data[17000:] , categorical_labels[17000:])
(x_valid , y_valid) = (x_test[:3000], y_test[:3000])
(x_test, y_test) = (x_test[3000:], y_test[3000:])


# In[22]:


x_train = np.array(x_train)/255


# In[23]:


x_valid = np.array(x_valid)/255


# In[24]:


x_test  = np.array(x_test)/255


# In[25]:


print(x_train.shape)
print(x_test.shape)
print(x_valid.shape)


# In[26]:


# model.predict(x_train)


# In[27]:


# len(x_train)+len(x_test) + len(x_valid) == len(X)


# In[28]:


# ZOOM AUGMENT
datagen = ImageDataGenerator(horizontal_flip=True)
# fit parameters from data
datagen.fit(x_train)


# In[29]:


# configure batch size and retrieve one batch of images
#os.chdir('C:/Sapna/Graham/Deep Learning/Projects/Face Detection/UTKFace/')
#os.makedirs('images')
#for i in range(300):
for X_batch, y_batch in datagen.flow(x_train, y_train,batch_size=6000):
    break


# In[30]:


len(X_batch)


# In[31]:


len(x_train)


# In[32]:


new_train_x=np.append(x_train,X_batch,axis=0)
len(new_train_x)


# In[34]:


new_train_y=np.append(y_train,y_batch,axis=0)
len(new_train_y)


# In[35]:


collections.Counter(np.argmax(new_train_y, axis=1))


# In[56]:


# # BRIGTNESS AUGMENT
# datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# # fit parameters from data
# datagen.fit(x_train)

# for X_batch, y_batch in datagen.flow(x_train, y_train,batch_size=6000):
#     break


# In[57]:


# new_train_x1 = np.append(new_train_x,X_batch,axis=0)
# len(new_train_x1)


# In[58]:


# new_train_y1 = np.append(new_train_y,y_batch,axis=0)
# len(new_train_y1)


# In[59]:


# collections.Counter(np.argmax(new_train_y1, axis=1))


# In[60]:


model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=4, padding='same', activation='relu', input_shape=(128,128,3))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(5, activation='softmax'))

# Take a look at the model summary
model.summary()


# In[61]:


model.compile(loss='categorical_crossentropy',
             optimizer='adagrad',
             metrics=['accuracy'])
#Try different optimizers - SGD/Momentum/NAG/Adagrad/Adadelta/Rmsprop/adam


# In[62]:


callbacks = [EarlyStopping(monitor='val_loss',mode='min', patience=10)]


# In[1]:


history = model.fit(new_train_x, # Features
          new_train_y, # Target vector
          epochs=100, # Number of epochs
          callbacks=callbacks, # Early stopping
          verbose=1,   # Print description after each epoch        
          batch_size=100, # Number of observations per batch
          validation_data=(x_valid, y_valid), # Data for evaluation
         )


# In[ ]:


history = model.history.history


# In[ ]:


#history['loss']
history.keys()


# In[ ]:


#print(history.history.keys())
# summarize history for accuracy
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[44]:


#Horizontal data augmentation -horizontal flip
model.save('C:/Sapna/Graham/Deep Learning/Projects/Codes/race_model.h5')


# In[ ]:


# Evaluate the model on test set
#score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
#print('\n', 'Test accuracy:', score[1])


# In[ ]:


labels =["White",  # index 0
        "Black",      # index 1
         "Asian",
         "Indian",
         "Hispanic-Latino"
        ]


# In[ ]:


train_preds = model.predict(new_train_x1)
test_preds  = model.predict(x_test)


# In[ ]:


train_acc = model.evaluate(new_train_x, new_train_y, verbose=0)


# In[ ]:


train_acc


# In[ ]:


len(x_test)
len(y_test)


# In[ ]:


test_acc  = model.evaluate(x_test, y_test, verbose=0)


# In[ ]:


test_acc


# In[ ]:


labels =["White",  # index 0
        "Black",      # index 1
         "Asian",
         "Indian",
         "Hispanic-Latino"
        ]


# In[ ]:


# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(test_preds[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(labels[predict_index], 
                                  labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
plt.show()

