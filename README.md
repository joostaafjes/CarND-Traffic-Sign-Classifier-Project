# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/distr_training_set.png "Distribution training set"
[image2]: ./examples/distr_validation_set.png "Distribution validation set"
[image3]: ./examples/distr_test_set.png "Distribution test set"
[image4]: ./examples/overview_data.jpg "Overview dataset"
[image5]: ./examples/random_noise.jpg "Random Noise"
[image6]: ./examples/30kmh.png "30 km/h"
[image7]: ./examples/stop.png "Stop"
[image8]: ./examples/70kmh.png "70 km/h"
[image9]: ./examples/oneway-noentry.png "Oneway no entry"
[image10]: ./examples/oneway-go.png "Oneway go"
[image11]: ./examples/rightturn.png "Right turn"
[image12]: ./examples/workinprogress.png "Work in progress"
[image13]: ./examples/trucks-forbidden.png "Trucks forbidden"
[image14]: ./examples/majorroad-ahead.png "Yield"
[image15]: ./examples/majorroad.png "Priority road"
[image16]: ./examples/100kmh.png "100 km/h"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Below the distribution of the training, validation and test sets per label/category are shown.

It can be noticed that:
1. The distribution of the categories for all sets are appr. the same.
2. The dataset is very unbalanced. Min. occurances are 180, max. 2010 (factor 11x)

![distribution training set][image1]
![distribution validation set][image2]
![distribution test set][image3]

Below 20 random samples per category.

![overview dataset][image4]

It can be noticed that:
- All trafic sign are aligned in the middle and appr. same size
- Contrast is very diverse
- Lightness/darkness of images is very diverse
- Sharpness is diverse
- Backgrounds are diverse

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Normalization

The following experiments are done to find the optimal normalization strategy (with 10 epocs):
1. Raw data, without normalization -> 0.866 test accuracy
2. Normalization to 0.1,0.9 -> 0.850 test accuracy
3. Normalization to -1.0,1.0 -> 0.904 test accuracy
4. Normalization to -0.5,0.5 -> 0.903 test accuracy
5. Normalization to 0.0,1.0 -> 0.902 test accuracy

Conclusion: normalisation is effective. Decided is to scale to 0.0 is 1.0 because this is also very handy with displaying images.

#### Grayscale

The following experiments are done to find the optimal normalization strategy (with 10 epocs):
1. Normalization to 0.1,0.9 -> 0.859 test accuracy
2. Normalization to -1.0,1.0 -> 0.899 test accuracy

Conclusion: converting to grayscale does not improve the performance. This also makes sense: When converting to grayscale some color information is lost, while color holds information about the type of sign.

#### Augmenting

For augmenting, I have the following techniques (using imgaug python library, https://github.com/aleju/imgaug):
1. Gaussian blur
2. Contrast normalisation
3. Additive Gaussian Noise
4. Multiply
5. Scaling
6. Translating
7. Rotating
8. Shearing

See below for the examples of augmentation:

![way of augmenting][image5]

The following augmenting strategies have used:
1. Different combinations of types of augmenting (e.g. scaling and translating)
2. Sequential version One of
3. Different number of augmentation between 500 and 2010 (=max)

The following mix resulted in the best test accuracy:
1. Multiplying, scaling, translating, rotating and shearing
2. Randomly use one of each augmentation (e.g. scaling or rotating)
3. Extend each training category to 2010 sample if needed

This lead to test accuracy of appr. 0.95


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started with the standard Lenet configuration that was also proposed in the previous lesson. Because of the good results, I haven't changed much. The only thing that I have added are dropout layers. After some experiments the following worked best:
1. Dropout layers after the relu of the fully connected layers.
2. Dropout probability of 0.5

This resulted in the following layers:
- input layers of 32x32x3, RGB input
- Convolution 5x5, output 28x28x6
- RELU
- Max pooling, stride 2x2, output 14x14x6
- Convoltion 5x5, output 10x10x6
- RELU
- Max pooling 2x2, output 5x5x16
- Flattening to 400
- Fully connected: 400 input to 120 output
- RELU
- dropout layer with 0.5
- Fully connected: 120 input to 84 output
- RELU
- dropout layer with 0.5
- Fully connected: 84 input to 43 output
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimization algorithm because of it proven good results.

Other hyperparameter that have used and not changed because of good results are:
- batch size 128
- learning rate 0.001

During training, I have mostly used 10 EPOCs to compare the results. When the model was stable, I have tried the optimal number between 50 and 100. After 50 EPOCs the results did almost not improve anymore.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.996
* test set accuracy of 0.951

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The first traffic signs that I have found, were classified correctly. So I search till I have found a mismatch.

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) | Speed limit (30km/h) | 
| Stop | Stop |
| Speed limit (70km/h) | Speed limit (70km/h) | 
| No entry | No entry |
| Ahead only | Ahead only |
| Turn right ahead | Turn right ahead |
| Road work | Road work |
|  |  |
| Yield | Yield |
| Priority road | Priority road |
|  |  |
|  |  |


The model was able to correctly guess 10 of the 11 traffic signs, which gives an accuracy of 91%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the xxth cell of the Ipython notebook.

For the first 10 images, the model is 100% sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

For the last(11th) image, the model is completly wrong. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| | 
| .20     				| |
| .05					| |
| .04	      			| |
| .01				    | |
