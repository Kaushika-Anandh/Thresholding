Exp.No : 09 
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
Date : 26.04.2023 
<br>
# Thresholding of Images
## Aim
To segment the image using global thresholding, adaptive thresholding and Otsu's thresholding using python and OpenCV.

## Software Required
1. Anaconda - Python 3.7
2. OpenCV

## Algorithm
- **Step1:** Load the necessary packages.
- **Step2:** Read the Image and convert to grayscale.
- **Step3:** Use Global thresholding to segment the image.
- **Step4:** Use Adaptive thresholding to segment the image.
- **Step5:** Use Otsu's method to segment the image.
- **Step6:** Display the results.
## Program

> program by: Kaushika A <br>
> reg no: 21221230048

**Load the necessary packages**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
**Read the Image and convert to grayscale**
```python
in_img=cv2.imread('suzume.PNG')
in_img2=cv2.imread('suzume2.PNG')

in_img= cv2.resize(in_img, (461,250))
in_img2= cv2.resize(in_img2, (463,284))

gray_img = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(in_img2,cv2.COLOR_BGR2GRAY)
```
**Use Global thresholding to segment the image**
```python
# cv2.threshold(image, threshold_value, max_val, thresholding_technique)
ret,thresh_img1=cv2.threshold(gray_img,86,255,cv2.THRESH_BINARY)
ret,thresh_img2=cv2.threshold(gray_img,86,255,cv2.THRESH_BINARY_INV)
ret,thresh_img3=cv2.threshold(gray_img,86,255,cv2.THRESH_TOZERO)
ret,thresh_img4=cv2.threshold(gray_img,86,255,cv2.THRESH_TOZERO_INV)
ret,thresh_img5=cv2.threshold(gray_img,100,255,cv2.THRESH_TRUNC)
```
**Use Adaptive thresholding to segment the image**
```python
# cv2.adaptiveThreshold(source, max_val, adaptive_method, 
                        threshold_type, blocksize, constant)
thresh_img6=cv2.adaptiveThreshold(gray_img2,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,11,2)
thresh_img7=cv2.adaptiveThreshold(gray_img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,11,2)
```
**Use Otsu's method to segment the image**
```python
# cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh_img8=cv2.threshold(gray_img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```
**Display the results**
```python
cv2.imshow('original image',in_img)
cv2.imshow('original image(second)',in_img2)

cv2.imshow('original image(gray)',gray_img)
cv2.imshow('original image(gray)(second)',gray_img2)

cv2.imshow('binary threshold',thresh_img1)
cv2.imshow('binary-inverse threshold',thresh_img2)
cv2.imshow('to-zero threshold',thresh_img3)
cv2.imshow('to-zero-inverse threshold',thresh_img4)
cv2.imshow('truncate threshold',thresh_img5)

cv2.imshow('mean adaptive threshold',thresh_img6)
cv2.imshow('gaussian adaptive threshold',thresh_img7)

cv2.imshow('otsu\'s threshold',thresh_img8)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

<br>

## Output
**Original Image**

<img src="https://github.com/Kaushika-Anandh/Thresholding/blob/main/9.PNG" width="600" height="400">

**Global Thresholding**

<img src="https://github.com/Kaushika-Anandh/Thresholding/blob/main/1.PNG" width="600" height="200">

<img src="https://github.com/Kaushika-Anandh/Thresholding/blob/main/2.PNG" width="600" height="200">

<img src="https://github.com/Kaushika-Anandh/Thresholding/blob/main/3.PNG" width="600" height="200">

<img src="https://github.com/Kaushika-Anandh/Thresholding/blob/main/4.PNG" width="600" height="200">

<img src="https://github.com/Kaushika-Anandh/Thresholding/blob/main/5.PNG" width="600" height="200">


**Adaptive Thresholding**

<img src="https://github.com/Kaushika-Anandh/Thresholding/blob/main/6.PNG" width="600" height="200">

<img src="https://github.com/Kaushika-Anandh/Thresholding/blob/main/7.PNG" width="600" height="200">

**Optimum Global Thresholding using Otsu's Method**

<img src="https://github.com/Kaushika-Anandh/Thresholding/blob/main/8.PNG" width="600" height="200">

## Result
Thus the images are segmented using global thresholding, adaptive thresholding and optimum global thresholding using python and OpenCV.

