#imports

#data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.signal import convolve

#deep learning
#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import EarlyStopping
from pytorchtools import EarlyStopping

#visualization
import matplotlib.pyplot as plt

#system libs
import os
import gc

#fix random seed
np.random.seed(42)
torch.manual_seed(42)

#names:
DATA_PATH = './data/'
TRAIN_NAME = 'train.csv'
TEST_NAME = 'test.csv'
MODEL_PATH = './model/'
MODEL_NAME = MODEL_PATH + 'model.pt'

#model configs:

#training params
N_EPOCH = 100
LR = 0.1
LR_CUCLES = 5

# early stopping settings
DELTA = 0.001 # thresold of improvement
PATIENCE = 10 # wait for 10 epoches for emprovement
BATCH_SIZE = 512
N_FOLD = 2 #number of folds for cross-validation
VERBOSE = 500 # print score every n batches

NOISE_STD = 0.05 #standard deviation of the noise

#input and output sizes of the model
INPUT_SIZE = 128
OUT_SIZE = 1

#dictionary of hyperparameters
HYPERPARAM = dict()

#global dropout rate
HYPERPARAM['Drop_rate'] = 0.25

#number of filers for the model
HYPERPARAM['n_filt_1'] = 128
HYPERPARAM['n_filt_2'] = 128
HYPERPARAM['n_filt_3'] = 128

#size of kernel of input channels
HYPERPARAM['kern_size_1'] = 4
HYPERPARAM['kern_size_2'] = 8
HYPERPARAM['kern_size_3'] = 16

HYPERPARAM['dense_size'] = 300