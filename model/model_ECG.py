from config import *

class ECG_Net(nn.Module):

    def __init__(self, hyperparams, input_size, out_size,n_channels):
        super(ECG_Net, self).__init__()

        # model parameters
        self.hyperparams = hyperparams
        self.input_size = input_size
        self.dropout_rate = hyperparams['Drop_rate']

        # define model layers
        # channel 1
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=hyperparams['n_filt_1'],
                               kernel_size=hyperparams['kern_size_1'], stride=1,
                               padding=int(hyperparams['kern_size_1'] / 2))
        self.bn1 = nn.BatchNorm1d(self.hyperparams['n_filt_1'])

        self.fc1 = nn.Linear(in_features=int(hyperparams['n_filt_1'] * input_size / 2),
                             out_features=hyperparams['dense_size'])
        self.fcbn1 = nn.BatchNorm1d(hyperparams['dense_size'])

        # channel 2
        self.conv2 = nn.Conv1d(in_channels=n_channels, out_channels=hyperparams['n_filt_2'],
                               kernel_size=hyperparams['kern_size_2'], stride=1,
                               padding=int(hyperparams['kern_size_2'] / 2))
        self.bn2 = nn.BatchNorm1d(self.hyperparams['n_filt_2'])

        self.fc2 = nn.Linear(in_features=int(hyperparams['n_filt_2'] * input_size / 2),
                             out_features=hyperparams['dense_size'])
        self.fcbn2 = nn.BatchNorm1d(hyperparams['dense_size'])

        # channel 3
        self.conv3 = nn.Conv1d(in_channels=n_channels, out_channels=hyperparams['n_filt_3'],
                               kernel_size=hyperparams['kern_size_3'], stride=1,
                               padding=int(hyperparams['kern_size_3'] / 2))
        self.bn3 = nn.BatchNorm1d(self.hyperparams['n_filt_3'])

        self.fc3 = nn.Linear(in_features=int(hyperparams['n_filt_3'] * input_size / 2),
                             out_features=hyperparams['dense_size'])
        self.fcbn3 = nn.BatchNorm1d(hyperparams['dense_size'])

        # output layer
        self.fc_out = nn.Linear(in_features=hyperparams['dense_size'], out_features=out_size)

    def forward(self, x):
        # we apply the convolution layers, followed by batch normalisation,

        # channel 1
        ch1 = self.bn1(self.conv1(x))  # batch_size x 64 x 64 x 1
        ch1 = F.relu(F.max_pool1d(ch1, 2))  # batch_size x 64 x 32 x 1

        ch1 = ch1.view(-1, int(self.input_size / 2 * self.hyperparams['n_filt_1']))  # batch_size x 8*256
        ch1 = F.dropout(F.relu(self.fcbn1(self.fc1(ch1))),
                        p=self.dropout_rate, training=self.training)
        ch1 = F.sigmoid(self.fc_out(ch1))

        # channel 2
        ch2 = self.bn2(self.conv1(x))  # batch_size x 64 x 64 x 1
        ch2 = F.relu(F.max_pool1d(ch2, 2))  # batch_size x 64 x 32 x 1

        ch2 = ch2.view(-1, int(self.input_size / 2 * self.hyperparams['n_filt_2']))  # batch_size x 8*256
        ch2 = F.dropout(F.relu(self.fcbn2(self.fc2(ch2))),
                        p=self.dropout_rate, training=self.training)
        ch2 = F.sigmoid(self.fc_out(ch2))

        # channel 3
        ch3 = self.bn3(self.conv1(x))  # batch_size x 64 x 64 x 1
        ch3 = F.relu(F.max_pool1d(ch3, 2))  # batch_size x 64 x 32 x 1

        ch3 = ch3.view(-1, int(self.input_size / 2 * self.hyperparams['n_filt_3']))  # batch_size x 8*256
        ch3 = F.dropout(F.relu(self.fcbn3(self.fc3(ch3))),
                        p=self.dropout_rate, training=self.training)
        ch3 = F.sigmoid(self.fc_out(ch3))

        # avaraging of layers
        averaged_preds = (ch1 + ch2 + ch3)/3 #torch.mean(torch.stack((ch1, ch2, ch3)))

        return averaged_preds

def target_metric(y_true,y_pred):

    return metrics.roc_auc_score(y_true,y_pred)#,average=None)


