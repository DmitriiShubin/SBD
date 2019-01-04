
# coding: utf-8

# In[ ]:


#import keras 
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

#define architecture for EEG recognizer
def ECG_rec(input_size,output_size):
    inputs = Input(input_size)
    conv1_1 = Conv1D(64, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1_1 = MaxPooling1D(pool_size=2)(conv1_1)
    flat_1 = Flatten()(pool1_1)
    dense1_1 = Dense(10,activation='relu')(flat_1)
    dense1_1 = Dropout(0.5)(dense1_1)
    final_1 = Dense(output_size,activation='sigmoid')(dense1_1)
                         
    conv1_2 = Conv1D(64, 8, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1_2 = MaxPooling1D(pool_size=2)(conv1_2)
    flat_2 = Flatten()(pool1_2)
    dense1_2 = Dense(10,activation='relu')(flat_2)
    dense1_2 = Dropout(0.5)(dense1_2)
    final_2 = Dense(output_size,activation='sigmoid')(dense1_2)
    
    conv1_3 = Conv1D(64, 16, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1_3 = MaxPooling1D(pool_size=2)(conv1_3)
    flat_3 = Flatten()(pool1_3)
    dense1_3 = Dense(10,activation='relu')(flat_3)
    dense1_3 = Dropout(0.5)(dense1_3)
    final_3 = Dense(output_size,activation='sigmoid')(dense1_3)
    
    final = Average()([final_1,final_2,final_3])
    model = Model(input = inputs, output = final)
                           
    #model.summary()

    return model

