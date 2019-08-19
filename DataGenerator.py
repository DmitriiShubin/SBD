#import
from config import *
from Decompose.SBD import *


class DataGenerator:

    def __init__(self,
                 decomp,
                 noise,
                 data_path,
                 test_name,
                 train_name
                 ):



        # load test and train
        df_test = pd.read_csv(data_path+test_name, index_col=0, header=0).values.astype('float32')
        df_train = pd.read_csv(data_path + train_name, index_col=0, header=0).values.astype('float32')

        gc.collect()

        self.X_test = df_test[:, :128]
        self.X_train = df_train[:, :128]

        self.y_test = df_test[:,-1]
        #self.y_test[self.y_test > 0] = 1 #replace all abnormal samples with 1s

        self.y_train = df_train[:, -1]
        #self.y_train[self.y_train > 0] = 1  # replace all abnormal samples with 1s

        # add the white noise
        if noise:

            plt.plot(self.X_train[0, :])
            plt.savefig('./pictures/Clean_signal.png')

            for i in range(self.X_train.shape[0]):
                self.X_train[i, :] += np.random.normal(0, NOISE_STD, (self.X_train.shape[1]))

            for i in range(self.X_test.shape[0]):
                self.X_test[i, :] += np.random.normal(0, NOISE_STD, (self.X_test.shape[1]))

            plt.plot(self.X_train[0, :])
            plt.savefig('./pictures/Noisy_signal.png')

        #reshape to feed into the CNN
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))


        # apply subband decomposition
        if decomp:

            self.X_train = SBD(self.X_train)
            self.X_test = SBD(self.X_test)

    def get_train_val(self,train_ind,val_ind):

        #get trian samples
        X_train = self.X_train[train_ind,:,:]
        y_train = self.y_train[train_ind]

        # get validation samples
        X_val = self.X_train[val_ind, :,:]
        y_val = self.y_train[val_ind]

        X_train, y_train, X_val, y_val = self.preprocessing(X_train,y_train,X_val,y_val)


        return X_train, y_train, X_val, y_val


    def preprocessing(self,X_train,y_train,X_val,y_val):

        #reshape target
        y_train = np.reshape(y_train,(y_train.shape[0],1))
        y_val = np.reshape(y_val, (y_val.shape[0], 1))

        return X_train,y_train,X_val,y_val





