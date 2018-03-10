# **Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: ./examples/distr_training_set.png "Distribution training set"
[image2]: ./examples/distr_validation_set.png "Distribution validation set"
[image3]: ./examples/distr_test_set.png "Distribution test set"
[image4]: ./examples/overview_data.png "Overview dataset"
[image5]: ./examples/augmenting.png "Overview augmenting"
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

### Data Set Summary & Exploration

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


#### Final model architecture

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
 


#### Training

To train the model, I used the Adam optimization algorithm because of it proven good results.

Other hyperparameter that have used and not changed because of good results are:
- batch size 128
- learning rate 0.001

During training, I have mostly used 10 EPOCs to compare the results. When the model was stable, I have tried the optimal number between 50 and 100. After 50 EPOCs the results did almost not improve anymore.

#### Results

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.996
* test set accuracy of 0.951

### Test a Model on New Images

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


#### Prediction

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
| Speed limit (100km/h) | Speed limit (30km/h) | 
|  |  |
|  |  |


The model was able to correctly guess 10 of the 11 traffic signs, which gives an accuracy of 91%.

#### Probabilities

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
