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

Initial Consideration for building the Model:<br/>
> Gender - Female/Male <br/>
> Age -    Child,Youth,Adult,Middle Age, Very Old (>60). Age buckets were created and the model was built to detect each of these age buckets <br/>
> Race -   White,Black,Asian,Indian and Hispanic-Latino. <br/>


## Gender Model: <br/>
### Steps: <br/>
Data Pre-Processing: <br/>
>> Reshape- shaping the image size (50x50 was the final one) <br/>
>> Gray Scale <br/>
>> Normalizing <br/>
>> Data Augmentation:  <br/>
    >>> Zooming images <br/>
    >>> Changing the image brightness <br/>
    >>> Flipping the images horizontally <br/>
    
Model Performance: <br/>
>> Train Accuracy :- 97% <br/>
>> Test Accuracy:- 95% <br/>

![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Gender%201.png)
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Gender%202.png)
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Gender%20Predictions.png)

## Age Model:  <br/>
### Steps: <br/>
Data Pre-Processing: <br/>
>> Reshape- shaping the image size (128x128 was the final one) <br/>
>> Gray Scale <br/>
>> Normalizing <br/>
>> Data Augmentation:  <br/>
    >>> Zooming images <br/>
    >>> Changing the image brightness <br/>
    >>> Flipping the images horizontally <br/>
Model Performance:    <br/> 
>> Train Accuracy :- 85% <br/>
>> Test Accuracy:-  80% <br/>

![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Age%201.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Age%202.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Age%203.png) <br/>

## Race Model <br/>
### Steps: <br/>
Data Pre-Processing: <br/>
>> Reshape- shaping the image size (128x128 was the final one) <br/>
>> Gray Scale <br/>
>> Normalizing <br/>
>> Data Augmentation:  <br/>
    >>> Zooming images <br/>
    >>> Changing the image brightness <br/>
    >>> Flipping the images horizontally <br/>
Model Performance:     <br/>
>> Train Accuracy :- 90% <br/>
>> Test Accuracy:-  83% <br/>

![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Race1.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Race2.png) <br/>
![alt text](https://github.com/sdmishra123/Computer-Vision-Project/blob/master/Race%203.png) <br/>








