
# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


---
### Relevant files

#### 1. This repo includes all required files and can be used to run the simulator in autonomous mode

 This repo includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Model script

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model.




### Model Architecture and Performance

#### 1. Incremental model improvement:

Minimal training, car gets stuck at a dirt road exit:

<img src="/images/run2-select.gif" video>

Regularization, dropout, additional training. Car takes the dirt road exits:

<img src="/images/run4.gif" video>

After augmenting the data by recording the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer away from edges when necessary, car learned to drive around the track  without a hitch:

<img src="/images/video.gif" video>

#### 2. Final Model Architecture

The final model architecture (model.py lines 82-111) consisted of a convolution neural network with the following layers and layer sizes:

![model_architecture](images/model.png)







```python

```
