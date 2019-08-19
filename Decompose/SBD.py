from config import *

def SBD(X):

    filter_bank = pd.read_csv('./Decompose/filters.csv', header=None, index_col=None).values

    # filter fuction
    def filter_data(data, filt):

        data_filtered = np.convolve(filt, data, mode='full')
        data_filtered = data_filtered[int((filt.shape[0] - 1) / 2):-int((filt.shape[0] - 1) / 2)]

        return data_filtered

    # apply sub-bands decomposition
    def subband(data, filter_grid):

        data = data[0, :]

        data_subband = np.zeros((filter_grid.shape[1],data.shape[0]))

        for i in range(filter_grid.shape[1]):
            data_subband[i, :] = filter_data(data, filter_grid[:, i])

        return data_subband

    # novel data
    X1 = np.zeros((X.shape[0], filter_bank.shape[1], X.shape[2]),dtype='float32')

    # novel data filling
    for i in range(X.shape[0]):
        buf = subband(X[i,:,:], filter_bank)
        X1[i, :, :] = buf

    gc.collect()

    return X1