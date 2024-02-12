<span style="color:#292934"> __Object detection with yolov3 \- __ </span>  <span style="color:#292934"> __colab__ </span>

# Slides Content

* About project
  * Purpose
  * Dataset
* Log in and start jupyter notebook
* Part 1: Download project materials
* Part 2: Download and convert yolov3 pretrained weight
* Part 3: Download and preprocess experimental dataset
* Part 4: Import libraries and pre\-defined functions
* Part 5: Link Google drive
* Part 6: Define variables and callbacks
* Part 7: Finetune yolov3 model using raccoon dataset
* Part 8: Test the model\.

# 

# Purpose

Help student get familiar with object detection using yolov3 model \[1\] that was pre\-trained on a large COCO dataset \[2\]\.

Students are able to finetune the yolov3 model to recognize object with a custom dataset \(e\.g\.\, raccoon dataset \[3\]\) using Google Colab\.

<span style="color:#222222">\[1\] Redmon J\, Farhadi A\. Yolov3: An incremental improvement\. </span>  <span style="color:#222222">arXiv</span>  <span style="color:#222222"> preprint arXiv:1804\.02767\. 2018 Apr 8\.</span>

<span style="color:#222222">\[2\] Lin TY\, Maire M\, </span>  <span style="color:#222222">Belongie</span>  <span style="color:#222222"> S\, Hays J\, </span>  <span style="color:#222222">Perona</span>  <span style="color:#222222"> P\, Ramanan D\, </span>  <span style="color:#222222">Dollár</span>  <span style="color:#222222"> P\, </span>  <span style="color:#222222">Zitnick</span>  <span style="color:#222222"> CL\. Microsoft coco: Common objects in context\. </span>  <span style="color:#222222">InEuropean</span>  <span style="color:#222222"> conference on computer vision 2014 Sep 6 \(pp\. 740\-755\)\. Springer\, Cham\.</span>

<span style="color:#222222">\[3\] https://github\.com/datitran/raccoon\_dataset</span>

# Yolov3 Architecture

<span style="color:#FF0000">Darknet\-53</span>

<span style="color:#FF0000">Trained on ImageNet</span>

<span style="color:#FF0000">53 more convolutional layers</span>

![](img%5CObject_Detection_with_Yolov30.png)

# Darknet-53

Darknet\-53 is used as a feature extractor\.

Darknet\-53 mainly composed of 3 x 3 and 1 x 1 filters with skip connections like the residual network in ResNet\.

![](img%5CObject_Detection_with_Yolov31.png)

# Dataset

* In this project\, raccoon dataset \[1\] is used to finetune the yolov3 model\.
* About raccoon dataset:
  * Number of classes: 1
  * Number of images: 200
  * Annotation file format: Pascal VOC XML \(\.xml\)

![](img%5CObject_Detection_with_Yolov32.png)

<span style="color:#222222">\[1\] </span>  <span style="color:#222222">https://github\.com/datitran/raccoon\_dataset</span>

<span style="color:#292934"> __Log in and start __ </span>  <span style="color:#292934"> __jupyter__ </span>  <span style="color:#292934"> __ notebook__ </span>

# 

# Login Information

ID: guest

Password:

# 

![](img%5CObject_Detection_with_Yolov33.png)

<span style="color:#FF0000">Right click \-> Open in Terminal \-> </span>  <span style="color:#FF0000">jupyter</span>  <span style="color:#FF0000"> notebook</span>

<span style="color:#292934"> __Part 1: Clone project materials__ </span>

# Download project materials

![](img%5CObject_Detection_with_Yolov34.png)

<span style="color:#292934"> __Part 2: Download and convert yolov3 pretrained weight__ </span>

# 

# Download and convert yolov3 pretrained weight

The Yolov3 model in this project is trained using Darknet library\, thus the pre\-trained weights is in Darknet format \(\.weights\)

We need to covert the pre\-trained weight into \.h5 file that can be used by TensorFlow Keras\.

![](img%5CObject_Detection_with_Yolov35.png)

![](img%5CObject_Detection_with_Yolov36.png)

<span style="color:#292934"> __Part 3:  download and preprocess experimental dataset__ </span>

# 

# Download experimental dataset

![](img%5CObject_Detection_with_Yolov37.png)

# Preprocess dataset

![](img%5CObject_Detection_with_Yolov38.png)

# Example of a Pascal VOC XML file

![](img%5CObject_Detection_with_Yolov39.png)

<span style="color:#FF0000">Information we need</span>

# Example of Yolo format file

![](img%5CObject_Detection_with_Yolov310.png)

<span style="color:#FF0000">xmin</span>  <span style="color:#FF0000">\, </span>  <span style="color:#FF0000">ymin</span>  <span style="color:#FF0000">\, </span>  <span style="color:#FF0000">xmax</span>  <span style="color:#FF0000">\, </span>  <span style="color:#FF0000">ymax</span>  <span style="color:#FF0000">\, class</span>

<span style="color:#FF0000">Images folder path \+ filename</span>

<span style="color:#FF0000">\./images/ \+ raccoon\-1\.jpg</span>

# data_classes.txt

![](img%5CObject_Detection_with_Yolov311.png)

<span style="color:#292934"> __Part 4:  Import libraries and pre\-defined functions__ </span>

# 

# Import libraries and pre-defined functions

![](img%5CObject_Detection_with_Yolov312.png)

<span style="color:#292934"> __Part 5:  Define variables and Callbacks__ </span>

# 

# Variables

![](img%5CObject_Detection_with_Yolov313.png)

# Callbacks

![](img%5CObject_Detection_with_Yolov314.png)

<span style="color:#292934"> __Part 6:  Finetune yolov3 model using raccoon dataset__ </span>

# 

# Finetune stage 1

<span style="color:#212121">Train with frozen layers first\, to get a stable loss</span>

# Split data

![](img%5CObject_Detection_with_Yolov315.png)

# Create and Compile model

![](img%5CObject_Detection_with_Yolov316.png)

![](img%5CObject_Detection_with_Yolov317.png)

# Train model

![](img%5CObject_Detection_with_Yolov318.png)

# Finetune stage 2

<span style="color:#212121">Unfreeze and continue training\, to fine\-tune\.</span>

<span style="color:#212121">Train longer if the result is unsatisfactory\.</span>

# Re-compile and continue to train model

![](img%5CObject_Detection_with_Yolov319.png)

<span style="color:#292934"> __Part 8: Test the model__ </span>

# 

# Load the trained model

![](img%5CObject_Detection_with_Yolov320.png)

# Testing function

![](img%5CObject_Detection_with_Yolov321.png)

# Test the model

![](img%5CObject_Detection_with_Yolov322.png)

<span style="color:#292934"> __Create Custom dataset Tutorial__ </span>

# 

# Download a batch of images from google images search

<span style="color:#FF0000">Download all images extension</span>

![](img%5CObject_Detection_with_Yolov323.png)

[https://download\-all\-images\.mobilefirst\.me](https://download-all-images.mobilefirst.me/)

# Annotate images

![](img%5CObject_Detection_with_Yolov324.png)

    * [https://www\.makesense\.ai/](https://www.makesense.ai/)

# Upload images

![](img%5CObject_Detection_with_Yolov325.png)

![](img%5CObject_Detection_with_Yolov326.png)

# Start project

![](img%5CObject_Detection_with_Yolov327.png)

# Add label

![](img%5CObject_Detection_with_Yolov328.png)

<span style="color:#FF0000">Actions > Edit Labels</span>

![](img%5CObject_Detection_with_Yolov329.png)

# Annotate all images

![](img%5CObject_Detection_with_Yolov330.png)

# Export Annotations

![](img%5CObject_Detection_with_Yolov331.png)

<span style="color:#FF0000">Actions > Export Annotations</span>

![](img%5CObject_Detection_with_Yolov332.png)

