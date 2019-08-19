#import modules
from DataGenerator import *
from model.model_run import *
from Decompose.SBD import *


#create datasets
GetData = DataGenerator(decomp=False,
                 noise=False,
                 data_path=DATA_PATH,
                 test_name=TEST_NAME,
                 train_name=TRAIN_NAME)


print('Training without the noise')

wo_auc = CV_loop(GetData, noise=False)
print('________________________________________')
print('\n AUC_ROC score on test set without noise: ',wo_auc,'\n')
print('________________________________________')


print('Training with the noise')

#create datasets
GetData = DataGenerator(decomp=False,
                 noise=True,
                 data_path=DATA_PATH,
                 test_name=TEST_NAME,
                 train_name=TRAIN_NAME)

features = list(np.arange(102))
w_auc = CV_loop(GetData,noise=True)
print('________________________________________')
print('\n AUC_ROC score on test set with noise: ',w_auc,'\n')
print('________________________________________')




print('Training with the noise + SBD')

#create datasets
GetData = DataGenerator(decomp=True,
                 noise=False,
                 data_path=DATA_PATH,
                 test_name=TEST_NAME,
                 train_name=TRAIN_NAME)

features = np.arange(102)
features = list(np.flip(features))


score  = CV_loop(GetData,noise=True) #find the initial score of the model

for i in features:

    X_train = GetData.X_train[:,i,:].copy()
    X_test = GetData.X_test[:, i, :].copy()

    X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    GetData.X_train = np.delete(GetData.X_train, i, 1)  #remove one of channels
    GetData.X_test = np.delete(GetData.X_test, i, 1)  # remove one of channels

    # save result in the array
    score_new = CV_loop(GetData,noise=True)  #count the score again

    # check the score to remove/keep the channel
    if score <= score_new:
        print('Score improved')
        print('Current score:', score)
        print('New score:', score_new)
        score = score_new
        gc.collect()
    else:
        GetData.X_train = np.append(GetData.X_train, X_train, 1)
        GetData.X_test = np.append(GetData.X_test, X_test, 1)
        print('Score did not improved')
        print('Current score:', score)
        print('New score:', score_new)
        gc.collect()


print('________________________________________')
print('\n AUC_ROC score on test set with noise and SBD: ',score,'\n')
print('________________________________________')