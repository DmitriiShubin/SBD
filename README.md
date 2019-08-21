# Summary

This repository demonstrates the ability to improve the target metric of the Deep Neural Network through the **Sub-Band Decomposition** - the method of the time-series pre-processing.

Applications:

* Audio processing, voice recognition, GAN-based de-noising, personalized audio Deep Networks
* Time-series forecasting (Energy consumption, stock prices, real-estate prices)
* Time-series segmentation (seq2seq, 1D-Unet, 1D Variational autoencoder, etc.)
* Time-series regression and classification (biological, RF signals)

# Theory of the operation

## Algorithm basics

Any Time-series can be represented in time and frequency domains:

![FFT](/pictures/fft.png)

_The theory behind the operation of the **Convolution Neural Network (CNN)** (LSTM/RNN) applied to Time-series, literally, is equal to applying the set of ![FIR](https://en.wikipedia.org/wiki/Finite_impulse_response) (![IIR](https://en.wikipedia.org/wiki/Infinite_impulse_response) for recurrent NNs) filters (i.e. kernels), passed through the non-linear function:_

![cnn](/pictures/cnnPNG.PNG)

That means, CNN/LSTM is trying to find some optimal frequency bands useful for the loss function. Let's try to put the model "on rails": the following method considers applying the bank of FIR filters on the data in order to identify what frequency domains are "useless" for the model:

![sbd](/pictures/SBD.PNG)

## Algorithm description

SBD is working as a pre-trained input layer of the DN. After the initial decomposition of the time-series, all components are combined into the vector and considered as features for the future Backward Recurrent Feature Elimination to find out the optimal bandwidth for fitting the model.

**Algorithm:**

Considering there is the input data: 
Data.shape = [n_samples,n_channels,Length] (Pytorch-like)

1. Generate the batch of filters (filer bank) with N filters. The overlap between filters is preferable. The order of filters is defined with respect to the accuracy/computational time trade-off. 

2. Apply all filters on the data, combine the filtered data into channels:

Data.shape = [n_samples,n_channels * n_filters,Length]

4. Feed the data into the network, estimate the target score

5. Considering each channel as a separate feature, apply Backward Recursive Feature Elimination:
  
  5.1 Remove one channel
  5.2 Estimate the score without one channel
  5.3 If the score is not improved - return the channel back; overwise continue
  5.4 Check all channels one by one using procedure described above


# Example and benchmarks

The following example demonstrates the performance of the approach, which is about 1-3% improvement of the quality.


**Objective**

- Binary classification

- Target metric is ROC-AUC score


**Data**

The dataset contains samples of Abnormal and Normal Electrocardiography (ECG) heartbeats.

It is split into test and train set; 4046 data samples in each. 

Labels are ideally balanced to make the ROC-AUC less robust for performance evaluation.

There were 3 experiments for the training:
1. training the model of the original dataset
2. Training the model with added Gaussian White Noise with STD = 0.05
3. Training the model with the White noise and applying SBD approach

**All scores were estimated in the Test set only.**

**Model**

The model has 3 channels, with different sizes of kernels, which means this model is optimized to analyse both "quick" and "slow" changes in the input sequence.

![sbd](/pictures/modelPNG.PNG)

**Results**


| Condition             | ROC-AUC             |
| --------------------- | ------------------- |
| Without the noise     | 0.86280705338223    |
| With noise, no SBD    | 0.8165197212035437  |
| With Noise, with SBD  | 0.8466614603288788  |
| Improvement           | 0.03                | 


# How to launch the code

1. follow the ./dependences folder
2. launch the install.py script
3. wait for installation
3. get back to the main folder
4. Run the Main.py script

In order to change the model parameters, refere to the config.py script
