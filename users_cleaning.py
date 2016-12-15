import pandas as pd
import scipy.sparse as sps


def read(path, sep=','):

    return pd.read_csv(path, sep)


def clean_data(dataframe, columns, filler=0):

    for col in columns:

        dataframe[col] = dataframe[col].fillna(filler)
        dataframe[col] = dataframe[col].astype(str)

    return dataframe


def vectorize_data(dataframe, columns):

    sparse_matrix = sps.csr_matrix()
    for col in columns:


