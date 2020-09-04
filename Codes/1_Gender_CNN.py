#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from PIL import Image
from random import shuffle
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from keras.wrappers.scikit_learn import KerasClassifier
from IPython.display import SVG
from keras.optimizers import Adam,SGD,Adagrad,RMSprop,Nadam,Adadelta,Adamax
from keras.layers import Dense, Dropout, Flatten,BatchNormalization


# In[ ]:


from google.colab import drive
drive.mount('/gdrive')
get_ipython().run_line_magic('cd', '/gdrive')


# In[ ]:


#load file and check size 23708,4
file=r'/gdrive/My Drive/Colab Notebooks/data_image.npy'
data=np.load(file,allow_pickle=True )
data.shape


# # Import the dataset

# In[ ]:


shuffle(data)


# In[ ]:


# Create train, validation and test set

X_train = []
y_train = []
X_test = []
y_test = []
X_valid = []
y_valid = []

train = 0
test = 0
valid = 0

for features, age, gender, race in data:
    if(train <= 15000):
        X_train.append(features)
        y_train.append(to_categorical(gender, num_classes=2))
        train = train + 1
    elif((train > 15000) and (valid <= 7000)):
        X_valid.append(features)
        y_valid.append(to_categorical(gender, num_classes=2))
        valid = valid + 1
    else:
        X_test.append(features)
        y_test.append(to_categorical(gender, num_classes=2))
        test = test + 1


# In[ ]:


# Convert list to arrays

X_train = np.array(X_train).reshape(-1, 50,50, 3)
X_valid = np.array(X_valid).reshape(-1, 50,50, 3)
X_test = np.array(X_test).reshape(-1, 50,50, 3)
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)

# Normalize images

X_train = X_train/255.0
X_valid = X_valid/255.0
X_test = X_test/255.0


# In[ ]:


X_train.shape


# # Running Basic Cnn model

# In[ ]:


model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(50,50,3)))
model1.add(BatchNormalization())
model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(2, activation='softmax'))
model1.summary()
model1.compile(loss='binary_crossentropy',
             optimizer=Adagrad(lr=0.008),
             metrics=['accuracy'])


# In[ ]:


m2=model1.fit(X_train,
         y_train,
         batch_size=10,
         epochs=20,
         verbose=2,
         validation_data=(X_valid, y_valid))

valid_loss1 = m2.history["val_loss"]
plt.plot(valid_loss1, linewidth=3, label="The baseline")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.xlim(0, 20)
plt.yscale("log")
plt.show()


# In[ ]:


model1.evaluate(X_test, y_test)


# In[ ]:





# # Creating a deeper CNN
# 

# In[ ]:


model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(50,50,3)))
model2.add(BatchNormalization())
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(2, activation='softmax'))
model2.summary()
model2.compile(loss='binary_crossentropy',
             optimizer=Adagrad(lr=0.001),
             metrics=['accuracy'])


# In[ ]:


m3=model2.fit(X_train,
         y_train,
         batch_size=10,
         epochs=30,
         verbose=2,
         validation_data=(X_valid, y_valid))


# In[ ]:



valid_loss1 = m3.history["val_loss"]
train_loss1 = m3.history["loss"]
plt.plot(valid_loss1, linewidth=3, label="Validation loss")
plt.plot(train_loss1, linewidth=3, label="Train loss")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.xlim(0, 35)
plt.yscale("log")
plt.show()

model2.evaluate(X_test, y_test)


# In[ ]:


m3.history


# In[ ]:


#No need to run more than 20 epochs


# # Testing Adam

# In[ ]:


model3 = Sequential()
model3.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(50,50,3)))
model3.add(BatchNormalization())
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))
model3.add(Conv2D(128, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))
model3.add(Flatten())
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(2, activation='softmax'))
model3.summary()
model3.compile(loss='binary_crossentropy',
             optimizer=Adam(lr=0.01),
             metrics=['accuracy'])


# In[ ]:


m4=model3.fit(X_train,
         y_train,
         batch_size=10,
         epochs=20,
         verbose=2,
         validation_data=(X_valid, y_valid))

valid_loss1 = m4.history["val_loss"]
plt.plot(valid_loss1, linewidth=3, label="The baseline")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.xlim(0, 45)
plt.yscale("log")
plt.show()

model3.evaluate(X_test, y_test)


# In[ ]:





# In[ ]:





# # Creating a Grid Search

# In[ ]:





# ## Testing for batch and epoch sizes

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


def create_model():
	# create model
	
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(50,50,3)))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))
  model.summary()
  model.compile(loss='binary_crossentropy',
             optimizer=Adagrad(lr=0.001),
             metrics=['accuracy'])
  return model


# In[ ]:


model = KerasClassifier(build_fn=create_model, verbose=2)


# In[ ]:


batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 20, 30]


# In[ ]:


param_grid = dict(batch_size=batch_size, epochs=epochs)


# In[ ]:


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,cv=2, verbose=3)
grid_result = grid.fit(X_train,y_train,validation_data=(X_valid, y_valid))


# In[ ]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


#Batch size 10 and Epoch 30 best parameters


# ## Tune the Training Optimization Algorithm

# In[ ]:


def create_model(optimizer='adam'):
	# create model
	
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(50,50,3)))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))
  model.summary()
  model.compile(loss='binary_crossentropy',
             optimizer=optimizer,
             metrics=['accuracy'])
  return model


# In[ ]:


#based on previous iteration select best batch size and epoch number
model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=10, verbose=2)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop','Adagrad', 'Adadelta','Adam', 'Adamax', 'Nadam']
#, 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2,verbose=2)
grid_result = grid.fit(X_train,y_train,validation_data=(X_valid, y_valid))


# In[ ]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


#Seems SGD and AdaDelta seem to perform the best


# ## Tune for Learning Rate and Momentum

# In[ ]:


learn_rate1 = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate1, momentum=momentum)
param_grid


# In[ ]:


def create_model(learn_rate=0.01, momentum=0):
	# create model
	
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(50,50,3)))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))
  model.summary()
  optimizer = SGD(lr=learn_rate, momentum=momentum)
  model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
  return model


# In[1]:


grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train,y_train,validation_data=(X_valid, y_valid))
# summarize results


# In[ ]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

