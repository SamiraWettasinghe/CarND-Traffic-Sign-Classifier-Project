@@ -1,185 +1 @@
# **Traffic Sign Recognition** 

## Writeup

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/SamiraWettasinghe/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used Python to calculate statistics about my data set.

* The size of training set is 29409.
* The size of the validation set is 9800.
* The size of the test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.


#### 2. Include an exploratory visualization of the dataset.

Shown below is a histogram indicating the number of images that are in the training set for each class.
![alt_text][image2]
Shown below is a histogram indicating the number of images that are in the validation set for each class.
![alt_text][image3]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Initially, the LeNet-5 neural network from the previous lab was run with grayscale images. However, it was later found while calculating the SoftMax probabilities of various selected test images, that the model is 100% certain of an incorrect classification. In addition, this was for an image with 1200 training examples and a model that attained 94.1% test accuracy. While diving deeper into the training data, it was found that a greyscale operation only works when the lighting conditions are near constant. Grayscale also assumes that the classifications are independent of color and therefore the model relies heavily on shape. Since traffic signs are designed to be attractive to the human user using contrasting colors, it is beneficial for the model to use color.

Training with the same architecture but with color, removed the problem. No color corrections were done to the images for fear of making some images better while others worse.

As a last step, I normalized the image data because this made convergence occur much faster. This does not have an overall effect on what the images looked like.

The training/validation set is modified such that a better distribution of data is present. I forced each class in the training set to have at most 1200 images with at least 100 images of each class in the validation set. If these conditions can't be made, then 90% of the images in the class must be in the training set. This way, I can ensure that each class will have the best possible representation during training.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 12x12x54	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x54					|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 4x4x162	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x162					|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 1x1x486	|
| RELU					|												|
| Fully connected		| Input = 486, Output = 380						|
| RELU					|												|
| Dropout				| Keep_Prob = 0.7								|
| Fully connected		| Input = 380, Output = 120						|
| RELU					|												|
| Dropout				| Keep_Prob = 0.7								|
| Fully connected		| Input = 120, Output = 50						|
| RELU					|												|
| Dropout				| Keep_Prob = 0.7								|
| Fully connected		| Input = 50, Output = 43						|

 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with the following hyperparameters:
 * Epochs = 15
 * Batch Size = 192
 * Learning Rate = 0.0006

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 99.8%
* Validation set accuracy of 99.0%
* Test set accuracy of 94.4%

As an initial trial, the exact LeNet-5 architecture used for the MNIST library in the previous lab was used. Using a copy of this architecture and nothing else yielded in a maximum validation set accuracy of 72%.

A trial was then done to observe the effects of changing filter size and depth. Increasing the initial filter size and depth helped with the validation accuracy. With a smaller filter size, the validation set accuracy increased up to 80%. Changing the maximum depth from 12 to 32 layers caused the validation set accuracy to go from a maximum of 82.7%. To increase the accuracy, more layers are needed.

The next trail was run with an additional fully connected layer followed by a RELU activation layer, and a convolution layer followed by a RELU activation layer and a max pooling layer. This caused the validation accuracy to increase and approach 90%. However, once dropout was added in between the fully connected layers, the accuracy increased up to 99%. This is because dropout prevents the model from overtraining on certain features on images.

It is recommended to run the model for close to 20 EPOCHS to get close to a validation set accuracy of 99.5%. However, this was not done for fear of overfitting and time constraints.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first two images are like each other. In fact, the first image (no entry) was misclassified as the second image (no passing) when the images were trained as grayscale. This is because the model had a difficult time differentiating since both images look like a circle with a line going across it. This can most likely be fixed from a better model with more convolution/fully connected layers or with more training set images, however it was deemed that the fastest method is to simply change the model to handle 3 color layers.

Also, the second image has sunlight behind the image. This was chosen to trick the max pooling layers. Since the max pooling layers look for the most prominent pixels only, the model should focus on the area where the sunlight is showing which is not where the traffic sign is located.

Image 3 should be simple to detect. This image was chosen due to the triangular shape of the sign.

Image 4 is well defined however it has 621 training images. This image is chosen to test how the model handles cases where it has relatively little training experience with. This is compared to image 5 which is also well defined however has more than double the training images.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image							        | Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| No entry      						| No entry   									| 
| No passing   							| No passing									|
| Right-of-way at the next intersection	| Right-of-way at the next intersection			|
| Turn right ahead						| Turn right ahead				 				|
| Speed limit (70km/h)					| Speed limit (70km/h) 							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.4%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is not very sure about its classification (probability of 72%), however the model has predicted the correct traffic sign. It seems as if more training is required for all top five SoftMax probabilities. All these signs lack in the number of training images, therefore, adding more diverse training images should fix this issue. The top five SoftMax probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.21e-01     			| No entry   									| 
| 1.13e-01 				| Slippery Road									|
| 5.53e-02				| Keep right									|
| 4.20e-02				| Dangerous curve to the right	 				|
| 2.02e-02				| End of no passing								|

For the second image note that the model is sure that this is a no passing sign. The model seems to do well in differentiating varying lighting conditions since this image is complex with many features in the background. The top five SoftMax probability are as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No passing   									| 
| 2.12e-16 				| No passing for vehicles over 3.5 metric tons	|
| 8.61e-20				| No vehicles									|
| 1.24e-21				| Vehicles over 3.5 metric tons prohibited		|
| 5.69e-22				| Speed limit (60km/h)							|

For the third image, the model is certain this is a right of way sign. The model can differentiate differences in the shape of the sign. The top five SoftMax probabilities are as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right-of-way at the next intersection			| 
| 5.69e-13 				| Beware of ice/snow							|
| 2.18e-17				| Double curve									|
| 7.50e-24				| End of speed limit (80km/h)					|
| 2.10e-24				| Pedestrians									|

For the fourth image, the model is certain this is a turn right ahead sign. The model does well in finding images that it has low training experience in. The top five SoftMax probabilities are as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn right ahead								| 
| 7.46e-12 				| Turn left ahead								|
| 1.30e-12				| Ahead only									|
| 7.48e-13				| Keep left						 				|
| 5.15e-13				| Dangerous curve to the left					|

For the fifth image, the model is certain that this is a 70km/h speed limit sign. This should be easy for the model to predict since there are 1200 training images and the sign is well defined. The top five SoftMax probabilities are as follows:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99e-01				| Speed limit (70km/h)							| 
| 1.37e-03 				| Speed limit (30km/h)							|
| 2.59e-05				| Speed limit (20km/h)							|
| 1.37e-12				| Speed limit (50km/h)					 		|
| 1.72e-14				| Bicycles crossing								|
# CarND-Traffic-Sign-Classifier-Project