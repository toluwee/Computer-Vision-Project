#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import json
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace


# In[2]:


os.chdir("C:/Sapna/Graham/Deep Learning/Projects/Codes/Sample")


# In[3]:


import tensorflow as tf
print(tf.VERSION)


# In[4]:


result = DeepFace.verify("img1.jpg", "img2.jpg")


# In[ ]:


result


# In[ ]:


print("Is verified: ", result["verified"])


# In[ ]:


dataset = [
    ['img1.jpg', 'img2.jpg'],
    ['img1.jpg', 'img3.jpg']
]
resp_obj = DeepFace.verify(dataset)


# In[ ]:


# df = DeepFace.find(img_path = "img1.jpg", db_path = "C:/Sapna/Graham/Deep Learning/Projects/Codes/Sample")
# print(df.head())


# In[6]:


vggface_result = DeepFace.verify("img1.jpg", "img2.jpg") #default is VGG-Face
#vggface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face") #identical to the line above
facenet_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "Facenet")
openface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "OpenFace")
deepface_result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "DeepFace")


# In[7]:


facenet_result = DeepFace.verify("avani2.jpg", "avani4.jpg", model_name = "Facenet")


# In[8]:


import matplotlib.image as mpimg
img_A = mpimg.imread("radhika2.jpg")
img_B = mpimg.imread("radhika3.jpg")
fig, ax = plt.subplots(1,2)
ax[0].imshow(img_A);
ax[1].imshow(img_B);
values_view = facenet_result.values()
value_iterator = iter(values_view)
first_value = next(value_iterator)
#print(first_value)
if(first_value==True):
    print("Both are the same person")
else:
    print("Both are different persons")


# In[9]:


import matplotlib.image as mpimg
img_A = mpimg.imread("avani2.jpg")
img_B = mpimg.imread("avani4.jpg")
fig, ax = plt.subplots(1,2)
ax[0].imshow(img_A);
ax[1].imshow(img_B);
values_view = facenet_result.values()
value_iterator = iter(values_view)
first_value = next(value_iterator)
#print(first_value)
if(first_value==True):
    print("Both are the same person")
else:
    print("Both are different persons")


# In[10]:


import matplotlib.image as mpimg
facenet_result = DeepFace.verify("avani2.jpg", "apu2.jpg", model_name = "Facenet")
img_A = mpimg.imread("avani4.jpg")
img_B = mpimg.imread("apu2.jpg")
fig, ax = plt.subplots(1,2)
ax[0].imshow(img_A);
ax[1].imshow(img_B);
values_view = facenet_result.values()
value_iterator = iter(values_view)
first_value = next(value_iterator)
#print(first_value)
if(first_value==True):
    print("Both are the same person")
else:
    print("Both are different persons")


# In[13]:


import matplotlib.image as mpimg
facenet_result = DeepFace.verify("abhishek.jpg", "28_0_0_man.jpg", model_name = "Facenet")
img_A = mpimg.imread("abhishek.jpg")
img_B = mpimg.imread("28_0_0_man.jpg")
fig, ax = plt.subplots(1,2)
ax[0].imshow(img_A);
ax[1].imshow(img_B);
values_view = facenet_result.values()
value_iterator = iter(values_view)
first_value = next(value_iterator)
#print(first_value)
if(first_value==True):
    print("Both are the same person")
else:
    print("Both are different individuals")


# In[ ]:


model = VGGFace.loadModel() #all face recognition models have loadModel() function in their interfaces
DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", model = model)


# In[ ]:


result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "cosine")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean")
result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", distance_metric = "euclidean_l2")


# In[5]:


#dog image
try:
    demography = DeepFace.analyze("dog_image.jpg") #passing nothing as 2nd argument will find everything
    #demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
    #demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
    im = Image.open('dog-image.jpg').resize((128,128))
    plt.imshow(im)
    print("Age: ", demography["age"])
    print("Gender: ", demography["gender"])
    print("Emotion: ", demography["dominant_emotion"])
    print("Race: ", demography["dominant_race"])
except:
    print("Face not detected")


# In[6]:


#Viviana
demography = DeepFace.analyze("viviana2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('viviana2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Jolie
demography = DeepFace.analyze("jolie 2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('jolie 2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[14]:


#Mark
demography = DeepFace.analyze("mark2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('mark2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#kid 
demography = DeepFace.analyze("kid_girl.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('kid_girl.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#50_0_0_0_yuri_.jpg
demography = DeepFace.analyze("50_0_0_0_yuri_.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('50_0_0_0_yuri_.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Dhaval
demography = DeepFace.analyze("29_0_0_dhav.jpg",model_name = "Facenet") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('29_0_0_dhav.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[15]:


#Sapna
demography = DeepFace.analyze("Sapna.png") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('Sapna.png').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[16]:


#Jigbie
demography = DeepFace.analyze("jigbie.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('jigbie.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[18]:


#Sofia latino celeb
demography = DeepFace.analyze("sofia7.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('sofia7.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[17]:


#Manaswi
demography = DeepFace.analyze("28_0_0_man.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('28_0_0_man.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Abhi
demography = DeepFace.analyze("24_0_0_0_abhi_-min.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('24_0_0_0_abhi_-min.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Dipen
demography = DeepFace.analyze("dipen.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('dipen.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Dipen
demography = DeepFace.analyze("dipen2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('dipen2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Devesh
demography = DeepFace.analyze("devesh.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('devesh.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Avani
demography = DeepFace.analyze("avani4.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('avani4.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#selena gomez
demography = DeepFace.analyze("selenagomez.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('selenagomez.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Dwayne
demography = DeepFace.analyze("Dwayne_Johnson.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open("Dwayne_Johnson.jpg").resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Devesh2
demography = DeepFace.analyze("devesh2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('devesh2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Niks
demography = DeepFace.analyze("niks2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('niks2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Samarth
demography = DeepFace.analyze("samarth.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('samarth.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Rohini
demography = DeepFace.analyze("rohi1.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('rohi1.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Didi
demography = DeepFace.analyze("didi2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('didi2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#apu
demography = DeepFace.analyze("apu3.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('apu3.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])
#apu.jpg


# In[ ]:


#apu
demography = DeepFace.analyze("apu2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('apu2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#vamika
demography = DeepFace.analyze("28_1_0_vam.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('28_1_0_vam.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#mansi
demography = DeepFace.analyze("25_1_0_man.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('25_1_0_man.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#sapna
demography = DeepFace.analyze("DSC00218.JPG") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('DSC00218.JPG').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Ashish
demography = DeepFace.analyze("ashish1.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('ashish1.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Radhika
demography = DeepFace.analyze("radhika3.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('radhika3.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", "White")


# In[ ]:


#Radhika
demography = DeepFace.analyze("radhika2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('radhika2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[19]:


#James
demography = DeepFace.analyze("james2.png") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('james2.png').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[21]:


# james.jpg
#James
demography = DeepFace.analyze("james.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('james.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Divya
demography = DeepFace.analyze("divya.png") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('divya.png').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Reem
demography = DeepFace.analyze("reem.png") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('reem.png').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[22]:


#Rak
demography = DeepFace.analyze("rak.png") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('rak.png').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Shabnoor
demography = DeepFace.analyze("shab2.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('shab2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


im = Image.open('shab2.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", 24.435677553)
print("Gender: ", "Female")
print("Emotion: ", "Happy")
print("Race: ", "Asian")


# In[ ]:


#Sapna
demography = DeepFace.analyze("sapna.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('sapna.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])


# In[ ]:


#Vishnu
demography = DeepFace.analyze("vishnu.jpg") #passing nothing as 2nd argument will find everything
#demography = DeepFace.analyze("img4.jpg", ['age', 'gender', 'race', 'emotion']) #identical to the line above
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
im = Image.open('vishnu.jpg').resize((128,128))
plt.imshow(im)
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])

