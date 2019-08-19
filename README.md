## Summary

This repository demonstrates ability to improve the target metring of the Deep Neural Network through the **Sub-Band Decomposition** - the method of the time-series pre-processing.

Applications:

* Audio processing, voice recognition, GAN-based de-noising, personalized audio Deep Networks
* Time-series forcasting (Energy consumption, stock prices, real-estate prices)
* Time-series segmentation (seq2seq, 1D-Unet, 1D-Vgg, etc.)
* Time-series regression and classification (biological, RF signals)

## Theory of the operation

# Algorithm basics

Any Time-series can be represented in time and frequency domains:



_The theory behind the operation of the **Convolution Neural Network (CNN)** (LSTM/RNN) applied on Time-series, literally, is equal to applying the set of ![FIR](https://en.wikipedia.org/wiki/Finite_impulse_response) (![IIR](https://en.wikipedia.org/wiki/Infinite_impulse_response) for reccurent NNs) filters (i.e. kernels), passed through the non-linear function._

That means, for the rest of the shapes in the dataset, there are 'usefull' frequency bands for the target loss function and 'useless' frequency bands which insert bias, varience and the noise into the model:



# Algorithm description


## Comparison to the wavelet processing



## Example and benchmarks



## How to launch
