## Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[![IMAGE ALT TEXT HERE](./readme_files/german_traffic_signs.png)](https://www.youtube.com/)

Overview
---
This program uses deep neural networks and convolutional neural networks to classify traffic signs. The model is able to classify traffic sign images and was trained and validated using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) with 43 classes and more than 40,000 images in total.
---

The steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset
The pickled dataset contains a training, validation and test set. The images are resized to 32x32.

## Model 

### Architecture

I decided to use a deep neural network classifier as a model, which was aforementioned in [Pierre Sermanet's / Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It has __ layers: **__ convolutional layers** for feature extraction and **__ fully connected layer** as a classifier.

<p align="center">
  <img src="model_architecture.png" alt="Model architecture"/>
</p>
