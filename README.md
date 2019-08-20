# Summary

This repository demonstrates the ability to improve the target metric of the Deep Neural Network through the **Sub-Band Decomposition** - the method of the time-series pre-processing.

Applications:

* Audio processing, voice recognition, GAN-based de-noising, personalized audio Deep Networks
* Time-series forecasting (Energy consumption, stock prices, real-estate prices)
* Time-series segmentation (seq2seq, 1D-Unet, 1D-VGG, etc.)
* Time-series regression and classification (biological, RF signals)

# Theory of the operation

## Algorithm basics

Any Time-series can be represented in time and frequency domains:



_The theory behind the operation of the **Convolution Neural Network (CNN)** (LSTM/RNN) applied on Time-series, literally, is equal to applying the set of ![FIR](https://en.wikipedia.org/wiki/Finite_impulse_response) (![IIR](https://en.wikipedia.org/wiki/Infinite_impulse_response) for recurrent NNs) filters (i.e. kernels), passed through the non-linear function:_



That means, for the lots of the shapes in the dataset, there are 'useful' frequency bands for the target loss function and 'useless' frequency bands which insert bias, variance, and the noise into the model:



## Algorithm description

Considering there is the input data: 
Data.shape = [n_samples,n_channels,Length] (Pytorch-like)

1. Generate the batch of filters (filer bank) with N filters (overlap between filters is preferable. The order of filters is defined with respect to the accuracy/computational time trade-off)

2. Apply all filters on the data, combine the filtered data into channels:

Data.shape = [n_samples,n_channels * n_filters,Length]

4. Feed the data into the network, estimate the target score

5. Considering each channel as a separate feature, apply Recurrent Feature Elimination:
  
  5.1 Remove one channel
  5.2 Estimate the score without one channel
  5.3 If the score is not improved - return the channel back; overwise continue
  5.4 Check all channels one by one using procedure described above


# Example and benchmarks

Typically, the following approach allows to fine tune the model by 1-3%.

The following example demonstrates perfomance of the approach.

Dataset contains samples of Abnormal and Normal Electrocardiography (ECG) heart beats.

The objective: binary classification; target metric - ROC-AUC score.

The dataest is splited into test and train set; 4046 data samples in each. 

Labales are ideally balanced to make the ROC-AUC less robust for perfomance evaluation.

Results of the run are represented below:

| Condition             | ROC-AUC             |
| --------------------- | ------------------- |
| Without the noise     | 0.86280705338223    |
| With noise, no SBD    | 0.8165197212035437  |
| With Noise, with SBD  | 0.8466614603288788  |
| Improvement           | 0.03                | 

# How to launch the code
