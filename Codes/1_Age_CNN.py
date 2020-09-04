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


# In[2]:


os.chdir('C:/Sapna/Graham/Deep Learning/Projects/Face Detection/UTKFace/')
os.chdir('Output Images')


# In[3]:


im =Image.open('1_0_0_20161219140623097.jpg.chip.jpg').resize((128,128))
im


# In[4]:


onlyfiles = os.listdir()


# In[5]:


len(onlyfiles)


# In[6]:


shuffle(onlyfiles)
age = [i.split('_')[0] for i in onlyfiles]


# We can split the data into Classes
# * Children (1-14) CLASS 0
# * Youth (14-25) CLASS 1
# *  ADULTS (25-40) CLASS 2
# *  Middle age (40-60) CLASS 3
# *  Very Old (>60) CLASS 4

# In[7]:


classes = []
for i in age:
    i = int(i)
    if i <= 14:
        classes.append(0)
    if (i>14) and (i<=25):
        classes.append(1)
    if (i>25) and (i<40):
        classes.append(2)
    if (i>=40) and (i<60):
        classes.append(3)
    if i>=60:
        classes.append(4)


# In[8]:


#Check for data imbalance
collections.Counter(classes)


# **CONVERT IMAGES TO VECTORS**

# In[9]:


X_data =[]
for file in onlyfiles:
    face = cv2.imread(file)
    face =cv2.resize(face, (128, 128) )
    X_data.append(face)


# In[10]:


classes[:10]


# In[11]:


categorical_labels = to_categorical(classes, num_classes=5)


# In[12]:


categorical_labels[:10]


# In[13]:


(x_train, y_train), (x_test, y_test) = (X_data[:17000],categorical_labels[:17000]) , (X_data[17000:] , categorical_labels[17000:])
(x_valid , y_valid) = (x_test[:3000], y_test[:3000])
(x_test, y_test) = (x_test[3000:], y_test[3000:])


# In[14]:


len(x_train)+len(x_test) + len(x_valid) == len(X_data)


# In[15]:


x_train = np.array(x_train)/255


# In[16]:


x_valid = np.array(x_valid)/255


# In[ ]:


x_test  = np.array(x_test)/255
# print(x_train.shape)
# print(x_test.shape)
# print(x_valid.shape)


# In[ ]:


# HORIZONTAL AUGMENT
datagen = ImageDataGenerator(horizontal_flip=True)
# fit parameters from data
datagen.fit(x_train)


# In[17]:


# configure batch size and retrieve one batch of images
#os.chdir('C:/Sapna/Graham/Deep Learning/Projects/Face Detection/UTKFace/')
#os.makedirs('images')
#for i in range(300):
for X_batch, y_batch in datagen.flow(x_train, y_train,batch_size=6000):
    break


# In[18]:


new_train_x=np.append(x_train,X_batch,axis=0)
len(new_train_x)


# In[19]:


new_train_y=np.append(y_train,y_batch,axis=0)
len(new_train_y)


# In[20]:


collections.Counter(np.argmax(new_train_y, axis=1))


# In[ ]:


# # BRIGTNESS AUGMENT
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# fit parameters from data
datagen.fit(x_train)

for X_batch, y_batch in datagen.flow(x_train, y_train,batch_size=6000):
    break


# In[1]:


new_train_x1 = np.append(new_train_x,X_batch,axis=0)
len(new_train_x1)


# In[ ]:


new_train_y1 = np.append(new_train_y,y_batch,axis=0)
len(new_train_y1)


# In[ ]:


collections.Counter(np.argmax(new_train_y1, axis=1))


# In[21]:


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


# In[22]:


model.compile(loss='categorical_crossentropy',
             optimizer='adagrad',
             metrics=['accuracy'])


# In[23]:


callbacks = [EarlyStopping(monitor='val_loss',mode='min', patience=10)]


# In[ ]:


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


# In[ ]:


#Horizontal data augmentation -horizontal flip
model.save('C:/Sapna/Graham/Deep Learning/Projects/Codes/age_class_model.h5')


# In[ ]:


# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])


# In[ ]:


score = model.evaluate(x_train, y_train, verbose=0)

# Print test accuracy
print('\n', 'Train accuracy:', score[1])


# In[ ]:


labels =["CHILD",  # index 0
        "YOUTH",      # index 1
        "ADULT",     # index 2 
        "MIDDLEAGE",        # index 3 
        "OLD",         # index 4
        ]


# In[2]:


y_hat = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(labels[predict_index], 
                                  labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
plt.show()

