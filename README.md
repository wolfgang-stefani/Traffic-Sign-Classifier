# Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

A deep learning model that learns to recognize traffic signs.

[![IMAGE ALT TEXT HERE](./readme_files/german_traffic_signs.png)](https://www.youtube.com/)

## Overview

This program uses deep neural networks and convolutional neural networks to classify traffic signs. The model is able to classify traffic sign images and was trained and validated using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) with 43 classes and more than 40,000 images in total.
The pickled dataset contains a training, validation and test set. The images are resized to 32x32.

Here is a link to the [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

The steps are the following:
* Step 0: Load The Data
* Step 1: Dataset Summary & Visualization
  * Summary of the original dataset
  * Visualization of the original dataset
  * Data Augmentation
  * Basic Summary of the augmented dataset
  * Data preprocessing
* Step 2: Design a Model Architecture (Deep Learning model)
* Step 3: Train and Evaluate the Deep Learning model
  * Tuning hyperparameters
  * Features and Labels
  * Training pipeline
  * Evaluation pipeline
  * Train and validate the model
* Step 4: Use the model to make predictions on new images found on the web
  * Load and Output the Images
  * Predict the Sign Type for Each Image
  * Analyze Performance
  * Analyze the softmax probabilities (output Top 5 Softmax Probabilities) for each image found on the web
---

### Data Set Summary & Visualization

#### 1. Summary of the original dataset

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of the test set is 12,630.
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the dataset is 43.

#### 2. Visualization of the original dataset

Here is an exploratory visualization of the German Traffic Signs dataset. 

First, a bar chart showing how the training samples are distributed across the classes:

![alt text][image1]

Some classes are highly **underrepresented**. Some only have 200 samples or even less, which is not enough for most of the models to perform well. The training samples are also fairly unbalanced which means some classes are represented to significantly lower extent than others. This will be fixed with **Data augmentation** (next rubric).

Additionally to the bar char chart above, for each class (type of traffic sign) 10 random images are plotted. As an example, here you can see two classes:

![alt text][image9]

#### 3. Data augmentation

Augmenting the training set (=generating additional data) helps improving the model. It makes the model more robust to slight variations, and hence prevents the model from overfitting.
Augmenting techniques are "cheap tricks" because no additional data needs to be collected and only a small mount on additional computing resources are needed but performance can significantly be improved.

Augmentation techniques can be Flipping, translation, scaling (zoom), rotation and many more. To add more data to the the dataset, I used the rotation technique because it is quite simple to implements but triples the amount of data in my solution. 

##### Flipping

Flipping is another augmentation technique, which was not implemented but illustrates well the general method of augmenting. Signs like "Ahead Only" are horizontally and/or vertically symmetrical. These can be simply flipped, which would allow us to get twice as much data for these classes.

Other signs like "Turn right ahead" and "Turn left ahead" are some kind of interchageable pairs. These can in a first step be flipped and then be assigned to the corresponding, pairing class. In this case the number of samples could be increased by a factor of around 4.

##### Translation, scaling (zoom) and rotation

CNNs have built-in invariance to small translations, scaling and rotations. The training doesn't contain such mutations, so we will add those. In this project we implement rotation. To demonstrate visually what rotation means, here is an example of an random original image and its augmented counterparts:

![alt text][image10]

Each image from the training set has been rotated like this resulting in a 3x wider training set:

* The size of the new, augmented training set is 104,397.

#### 4. Data Preprocessing

Preprocessing refers to techniques such as converting to grayscale, normalization, etc.

##### Normalizing
As a first step, the images are being normalized so that the data has mean zero and equal variance. Normalizing helps the network to converge faster. It makes it a lot easier for the optimizer to proceed numerically:

![alt text][image11]

For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data. It doesn't change the content of the images:

![alt text][image12]

#### Grayscaling (single-channel images)
As a second step, I grayscaled the image data because using color channels didn't seem to improve things a lot as Pierre Sermanet and Yann LeCun mentioned in [their paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

---

### Model Architecture (Deep Learning model)

As a starting point, I decided to use a convolutional neural network architecture known as LeNet-5 and similarly implemented in [Pierre Sermanet's / Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). LeNet-5 consists of 6 layers (C1, S2, C3, S4, C5, F6) if you do not count input and output as a layer:

![alt text][image13]

A problem with this initial architecture is overfitting - a high accuracy on the training set but low accuracy on the validation set.

After an iterative approach consisting of several steps of testing different layers and hyperparameters, my convolutional neural network classifier consists of 8 layers: **3 convolutional layers**, **3 subsampling/pooling layers** and **2 fully connected layers**:

![alt text][image14]

In comparison to LeNet, the **learning rate** is reduced from 0,001 to 0,0005. A higher learning rate does not mean to learn more and faster. In fact you get to a better model with low loss faster with a low learning rate.
**Epochs** are changed from 10 to 50. An epoch is a single pass of the whole dataset through the model used to increase the accuracy of the model without requiring more data. It is importan to chose the right number of epochs.

For reducing overfitting, the regularization technique dropout is implemented with `keep_prob` set to 0,7. 

The code of the model can be found in the [project's IPython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) in the first cell right under the headline "Step 2: Design a Model Architecture (Deep Learning model)".

The results of the final model are:
* training set accuracy of 98,1%
* validation set accuracy of 99,9%
* test set accuracy of 95,6%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/two_classes.jpg "Two Classes"
[image10]: ./examples/rotation.jpg "Rotation"
[image11]: ./examples/Normalized_problem.jpg "Normalized Inputs"
[image12]: ./examples/Normalizing_images.jpg "Normalizing Images"
[image13]: ./examples/LeNet-5.png "LeNet-5.png"
[image14]: ./examples/convnet.jpg "Convnet"
