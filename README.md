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



That means, for the rest of the shapes in the dataset, there are 'useful' frequency bands for the target loss function and 'useless' frequency bands which insert bias, variance, and the noise into the model:



## Algorithm description

input:
Data.shape = [n_samples,n_channels,Length] (Pytorch-like)

1. Generate the batch of filters (filer bank) with N filters (overlap between filters is preferable. The order of filters if defined with respect to the accuracy/computational time trade-off)

2. Apply all filters in the data, combine the filtered data into channels:

Data.shape = [n_samples,n_channels * n_filters,Length]

4. Feed the data into the network, estimate the target score

5. Considering each channel as a separate feature, apply Recurrent Feature Elimination:
  
  5.1 Remove one channel
  5.2 Estimate the score without one channel
  5.3 If the score is not improved - return channel back; overwise continue
  5.4 Check all channels one by one using algorithm represented above


# Example and benchmarks



# How to launch the code
