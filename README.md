# Computer-Vision-Project on Face Recognition & Facial Attribute Analysis from Facial Images : <br/>


# Problem Statement: <br/>

Detect the Gender, Age and Race from facial images using Convolutionsl Neural Networks and VGGFace Transfer Learning Models <br/>

# Project Motivation: <br/>
> Implement different model architectures <br/>
> Target Specific Audience for advertising <br/>
> Understand Customer behaviour & campaign feedback <br/>
> Surveillance system to follow COVID-19 face mask guidelines <br/>

# Data Procurement & Specifications <br/>

UTKFace:- ~23K cropped face images with age,gender and race <br/>

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of ~23K face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc. <br/>

https://susanqq.github.io/UTKFace/ <br/>

![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/UTKFace%20Images.JPG)<br/>

Insights on the Response variables:
> There is balance between Male and Female facial images<br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Gender.png)<br/>
> The Age is right skewed<br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Age.png)<br/>
> For Race, White is dominant in the data set<br/><br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Race.png)<br/>
> Images are alligned and only contains cropped faces<br/>
> Images show variation in pose, facial expression, illumination and resolution.<br/>

# Convolutional Model Implementation:<br/>

Given that we have to detect the Gender, Age and Race. We will have to build three different CNN models for each of the Response variables.<br/>

### Initial Consideration for building the Model:<br/>
> Gender - Female/Male <br/>
> Age -    Child,Youth,Adult,Middle Age, Very Old (>60). Age buckets were created and the model was built to detect each of these age buckets <br/>
> Race -   White,Black,Asian,Indian and Hispanic-Latino. <br/>


## Gender Model: <br/>
### Steps: <br/>
#### Data Pre-Processing: <br/>
>> Reshape- shaping the image size (50x50 was the final one) <br/>
>> Gray Scale <br/>
>> Normalizing <br/>
>> Data Augmentation:  <br/>
    >>> Zooming images <br/>
    >>> Changing the image brightness <br/>
    >>> Flipping the images horizontally <br/>
    
#### Model Performance: <br/>
>> Train Accuracy :- 97% <br/>
>> Test Accuracy:- 95% <br/>

![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Gender%201.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Gender%202.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Gender%20Predictions.png) <br/>

## Age Model:  <br/>
### Steps: <br/>
#### Data Pre-Processing: <br/>
>> Reshape- shaping the image size (128x128 was the final one) <br/>
>> Gray Scale <br/>
>> Normalizing <br/>
>> Data Augmentation:  <br/>
    >>> Zooming images <br/>
    >>> Changing the image brightness <br/>
    >>> Flipping the images horizontally <br/>
#### Model Performance:    <br/> 
>> Train Accuracy :- 85% <br/>
>> Test Accuracy:-  80% <br/>

![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Age%201.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Age%202.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Age%203.png) <br/>

## Race Model <br/>
### Steps: <br/>
#### Data Pre-Processing: <br/>
>> Reshape- shaping the image size (128x128 was the final one) <br/>
>> Gray Scale <br/>
>> Normalizing <br/>
>> Data Augmentation:  <br/>
    >>> Zooming images <br/>
    >>> Changing the image brightness <br/>
    >>> Flipping the images horizontally <br/>
#### Model Performance:     <br/>
>> Train Accuracy :- 90% <br/>
>> Test Accuracy:-  83% <br/>

![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Race1.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Race2.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Race%203.png) <br/>


# Building Model using Pre-trained Models 
Why use Transfer learning? <br/>
Pre-Trained models are built on millions of datasets. To improve the accuracy of our models we decided to use pre-trained models. <br/>

## Keras VGGFace2 - Resnet50 <br/>
Face recognition is a computer vision task of identifying and verifying a person based on a photograph of their face. <br/>
VGGFace and VGGFace2 model developed by researchers at the Visual Geometry Group at Oxford. <br/>
Although the model can be challenging to implement and resource intensive to train, it can be easily used in standard deep learning libraries such as Keras through the use of freely available pre-trained models and third-party open source libraries. <br/>

As we saw above that the Gender CNN model that was built from scratch did a good job in detecting both the Female and Male classes accurately. <br/>
Race CNN model did a descent job but the Age model has lot of room for improvement. Hence, we decieded upon using VGGFace pre-trained model wherein we tunned the last few layers and trained it to detect the below 12 age buckets: <br/>
12 classes = ["0-2","3-5","6-10","11-15","16-20","21-25","26-30","31-35","36-40","41-50","51-59",">=60"]<br/>
#### Specifications: Input Image size (224x224)<br/>
> Train Accuracy :  60% <br/>
> Test Accuracy:  58% <br/>

# Facebook DeepFace<br/>
Deepface is a lightweight facial analysis framework including face recognition and demography detection. <br/>
It has 4 extended models that is used to detect the demographic details of the given image, they are Gender, Race, Age (exact age) and Emotions.<br/>
Emotions - Angry, Fear, Neutral, Sad, Disgust, Happy and Surprise<br/>

# Future Work<br/>
Improve existing model performance by fine tuning the pretrained model layers.<br/>
Try other Data Augmentation techniques to increase data size and to handle data imbalance<br/>
Check for other alternative Face recognition models like Facenet and Googlenet<br/>

#### Team members: This was a team effort including Abhishek Yadav and Manaswi Mishra

#### https://susanqq.github.io/UTKFace/
#### https://modelzoo.co/
#### https://pypi.org/project/deepface/
#### http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
#### https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
