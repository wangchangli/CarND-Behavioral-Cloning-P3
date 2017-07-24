#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the nvidia pipeline which was mentioned in the lesson, refer [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) for the architecture details.
####2. Attempts to reduce overfitting in the model
The nvidia model doesn't contain a dropout layer.
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 19). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used both left/center/right camera images, for left camera images, add a steering correction of 0.3, for right camera images, add a steering correction of -0.3(model.py line 30-43)

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My starting model config was: nvidia pipeline + left/right images with steering correction of 0.2 + 32 batch size + 7 epochs; 
The training result shows that loss and val_loss are both ok, but the car still goes off the track at the first turn right after the bridge.  

The conner at which the car goes off the track is a big corner, So I increase the steering correction to 0.3/-0.3, then it works!

I also try a steering correction of 0.2/-0.2, but with a lower speed, the car goes well. I guess we need to train speed and steering together to get a general solution.

####2. Final Model Architecture

nvidia pipeline + left/right images with steering correction of 0.3 + 32 batch size + 7 epochs.

####3. Creation of the Training Set & Training Process
I find it is a little difficult to get a ideal data with just keyboard input control...So I use the provided training data directly.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by that MSE doesn't reduce after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.