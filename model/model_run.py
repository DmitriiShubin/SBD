#import
from model.model_ECG import *
from config import *
from Decompose.SBD import *
from DataGenerator import *

class Model:

    def __init__(self,
                 hyperparams,
                 input_size,
                 out_size,
                 n_channels,
                 lr,
                 LR_cucles,
                 n_epoch,
                 delta,
                 patience,
                 batch_size,
                 verbose
                 ):

        #set model parameters:
        self.n_epoch = n_epoch
        self.lr = lr
        self.LR_cucles = LR_cucles  # the number of cycles for decreassing of the learning rate
        self.delta = delta
        self.patience = patience
        self.batch_size = batch_size
        self.verbose = verbose


        #define the model
        self.net = ECG_Net(hyperparams=hyperparams,
                    input_size=input_size,
                    out_size=out_size,
                    n_channels=n_channels)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.cuda()
        summary(self.net, (n_channels, input_size))

        # loss function; oprimizer
        self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.CrossEntropyLoss()



    def fit(self,X_train,y_train,X_val,y_val):


        lr = self.lr
        delta = self.delta

        train_loss_monitor = np.array([])
        test_loss_monitor = np.array([])

        for cycle in range(self.LR_cucles+1):

            # define optimizer
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

            # define early stopping
            early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=delta)


            for epoch in range(self.n_epoch):


                self.net.train()

                # calculate the number of butches and the length of the tail
                n_batch = X_train.shape[0] // self.batch_size  # number of batches
                tailLength = int(X_train.shape[0] - self.batch_size * n_batch)

                #clear running loss every epoch

                running_loss = 0.0

                for batch in range(n_batch):

                    X_batch = X_train[0 + batch * self.batch_size:(batch + 1) * self.batch_size, :,:]
                    y_batch = y_train[0 + batch * self.batch_size:(batch + 1) * self.batch_size,:]

                    #X_batch = np.reshape(X_batch, (self.batch_size, 1, X_batch.shape[1]))

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    y_batch = torch.from_numpy(y_batch).float()
                    X_batch = torch.from_numpy(X_batch).float()

                    y_batch = y_batch.to(self.device)
                    X_batch = X_batch.to(self.device)

                    outputs = self.net(X_batch)
                    train_loss = self.criterion(outputs, y_batch)
                    train_loss.backward()
                    self.optimizer.step()

                    # print statistics
                    running_loss += train_loss.item()
                    if batch % self.verbose == self.verbose - 1:  # print every 2000 mini-batches
                        print('| Epoch: ',epoch + 1,'| Batch :',batch + 1,'| Loss: ',train_loss.item() / (self.verbose*self.batch_size))

                    # clear variables
                    X_batch = X_batch.cpu().detach()
                    y_batch = y_batch.cpu().detach()

                    del X_batch, y_batch, outputs, train_loss
                    gc.collect()

                if (tailLength != 0):
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    X_batch = X_train[-tailLength:-1, :]
                    y_batch = y_train[-tailLength:-1,:]

                    #X_batch = np.reshape(X_batch, (tailLength - 1, X_batch.shape[1], X_batch.shape[1]))

                    X_batch = torch.tensor(X_batch).float()
                    y_batch = torch.tensor(y_batch).float()

                    y_batch = y_batch.to(self.device)
                    X_batch = X_batch.to(self.device)

                    # forward + backward + optimize
                    outputs = self.net(X_batch)
                    train_loss = self.criterion(outputs, y_batch)
                    train_loss.backward()
                    self.optimizer.step()

                    running_loss += train_loss.item()

                    # clear variables
                    X_batch = X_batch.cpu().detach()
                    y_batch = y_batch.cpu().detach()

                    del X_batch, y_batch,outputs,train_loss
                    gc.collect()

                running_loss /= X_train.shape[0]
                train_loss_monitor = np.append(train_loss_monitor, running_loss)



                #evaluate the model

                self.net.eval()  # prep model for evaluation
                torch.no_grad()

                # forward pass: compute predicted outputs by passing inputs to the model

                print('Evaluation of the model')

                n_batch = X_val.shape[0] // self.batch_size  # number of batches
                tailLength = int(X_val.shape[0] - self.batch_size * n_batch)

                y_pred = np.zeros((y_val.shape[0],1))

                val_loss = 0.0

                for batch_val in range(n_batch):

                    X_batch = X_val[0 + batch_val * self.batch_size:(batch_val + 1) * self.batch_size, :,:]
                    #X_batch = np.reshape(X_batch, (self.batch_size, 1, X_batch.shape[1]))

                    y_batch = y_train[0 + batch_val * self.batch_size:(batch_val + 1) * self.batch_size, :]

                    # forward
                    X_batch = torch.from_numpy(X_batch).float()
                    y_batch = torch.from_numpy(y_batch).float()

                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.net(X_batch)
                    y_pred[0 + batch_val * self.batch_size:(batch_val + 1) * self.batch_size] = outputs.cpu().detach().numpy()

                    #get validation loss
                    val_loss += self.criterion(outputs, y_batch).item()

                    # clear variables
                    X_batch = X_batch.cpu().detach()
                    y_batch = y_batch.cpu().detach()

                    del X_batch, y_batch, outputs
                    gc.collect()

                if (tailLength != 0):

                    X_batch = X_train[-tailLength:-1, :,:]
                    y_batch = y_train[-tailLength:-1,:]

                    #X_batch = np.reshape(X_batch, (tailLength - 1, 1, X_batch.shape[1]))

                    X_batch = torch.tensor(X_batch).float()
                    y_batch = torch.tensor(y_batch).float()

                    y_batch = y_batch.to(self.device)
                    X_batch = X_batch.to(self.device)

                    # forward + backward + optimize
                    outputs = self.net(X_batch)
                    y_pred[-tailLength:-1] = outputs.cpu().detach().numpy()

                    # get validation loss
                    val_loss += self.criterion(outputs, y_batch).item()

                    # clear variables
                    X_batch = X_batch.cpu()
                    y_batch = y_batch.cpu()

                    del X_batch, y_batch, outputs
                    gc.collect()

                val_loss /= y_val.shape[0]

                test_loss_monitor = np.append(test_loss_monitor, val_loss)

                #stimate the tatget metric
                metric = target_metric(y_val, y_pred)

                print('Eval ROC_AUC score: ',metric)

                early_stopping(metric, self.net)

                #scheduler.step(metric)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                # load the last checkpoint with the best model
                self.net.load_state_dict(torch.load('checkpoint.pt'))



            #reducing the learning rate
            lr /= 3
            delta /= 3

            # define early stopping
            early_stopping.delta = delta


        return train_loss_monitor,test_loss_monitor


    #prediction function
    def predict(self,X_test,batch_size):

        n_batch = X_test.shape[0] // batch_size  # number of batches
        tailLength = int(X_test.shape[0] - batch_size * n_batch)

        y_pred = np.zeros((X_test.shape[0], 1))

        val_loss = 0.0

        for batch in range(n_batch):

            X_batch = X_test[0 + batch * self.batch_size:(batch + 1) * self.batch_size, :,:]
            #X_batch = np.reshape(X_batch, (self.batch_size, 1, X_batch.shape[1]))

            # forward
            X_batch = torch.from_numpy(X_batch).float()
            X_batch = X_batch.to(self.device)

            outputs = self.net(X_batch)
            y_pred[0 + batch * self.batch_size:(batch + 1) * self.batch_size] = outputs.cpu().detach().numpy()

        # clear variables
        X_batch = X_batch.cpu()

        del X_batch, outputs
        gc.collect()

        return y_pred

    def model_save(self,model_path):

        torch.save(self.net.state_dict(), model_path)


def CV_loop(GetData,noise,mode=False):

    # kfold cross-validation
    kf = KFold(N_FOLD, shuffle=True, random_state=42)

    for fold, (train_ind, val_ind) in enumerate(kf.split(GetData.X_train)):

        # load dataset
        X_train, y_train, X_val, y_val = GetData.get_train_val(train_ind, val_ind)

        # define model
        Peak_classifier = Model(hyperparams=HYPERPARAM,
                                input_size=INPUT_SIZE,
                                out_size=OUT_SIZE,
                                n_channels=X_train.shape[1],
                                lr=LR,
                                LR_cucles=LR_CUCLES,
                                n_epoch=N_EPOCH,
                                delta=DELTA,
                                patience=PATIENCE,
                                batch_size=BATCH_SIZE,
                                verbose=VERBOSE
                                )

        # fit the model
        train_loss, val_loss = Peak_classifier.fit(X_train, y_train, X_val, y_val)

        Peak_classifier.model_save(MODEL_NAME)

        if mode == False:
            break

    # plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['Train loss', 'Test loss'])
    plt.title('Training curves')

    if noise:
        plt.savefig('./pictures/Learning_curve_with_noise.png')
    else:
        plt.savefig('./pictures/Learning_curve_without_noise.png')
    # make predictions on test

    # get data
    X_test = GetData.X_test
    y_test = GetData.y_test

    # make prediciton
    y_pred = Peak_classifier.predict(X_test, BATCH_SIZE)

    #estimate the metric
    auc_score = target_metric(y_test, y_pred)

    return auc_score
