import scipy.sparse as sps
import pandas as pd


class DataTransformer(object):
    """ Tranform data before processing """

    def __init__(self, dataframe):

        super(DataTransformer, self).__init__()
        self.dataframe = dataframe
        self.filler = None
        self.data = None

    def __str__(self):
        return "DataTransformer"

    def transform(self, columns, filler=0, clean=True, vectorize=True):
        self.filler = filler

        # cleaning data
        if clean:
            self.clean_data(columns)

        # vectorize if specified
        if vectorize:
            self.vectorization(columns)
        else:
            self.data = self.dataframe

        return self.data

    def clean_data(self, columns):

        for col in columns:
            self.dataframe[col] = self.dataframe[col].fillna(self.filler)
            self.dataframe[col] = self.dataframe[col].astype(str)

    def vectorization(self, columns):

        shape = (self.dataframe.shape[0], 1)
        self.data = sps.csc_matrix(shape)   # initialization
        for col in columns:

            sparse_matrix = sps.csr_matrix(pd.get_dummies(self.dataframe[col]))
            self.data = sps.hstack([self.data, sparse_matrix])

        self.data = self.data.tocsc()[:, 1:] # removing the first column (initialization)

