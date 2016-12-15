import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def read_dataset(path,
                 header=None,
                 columns=None,
                 key='user_id',
                 sep='\t',
                 series=False):

    # loading the dataframe
    dataframe = pd.read_csv(path, sep=sep)
    logger.info('Columns: {}'.format(dataframe.columns.values))

    if series:
        instance_to_idx = build_series(dataframe[key])
        dataframe['instance_idx'] = instance_to_idx[dataframe[key].values].values
        idx_to_instance = pd.Series(index=instance_to_idx.data, data=instance_to_idx.index)

        return dataframe, idx_to_instance, instance_to_idx

    else:
        return dataframe


def build_series(series):

    instances = series.unique()  # may be users or items
    # let's build two useful series
    instance_to_idx = pd.Series(data=range(len(instances)), index=instances)
    # idx_to_instance = pd.Series(index=instance_to_idx, data=instance_to_idx.index)

    return instance_to_idx


def process_data(data, attribute):
    # transform column
    attr_list, attr_series = retrieve_attribute(data, attribute)
    print(attr_list)
    # vectorize users' jobroles
    user_attr_matrix = build_bow(attr_list).tocsc()

    return user_attr_matrix


def recommendable_items(dataframe, attribute, key):

    series = dataframe[dataframe[attribute] == 1][key]
    return series


def retrieve_attribute(dataframe, attribute):

    # let's keep only the 'tags' column
    tags = dataframe[attribute]
    tag_list = np.array(tags.tolist())

    # transforming tag's format
    n = len(tag_list)
    for i in np.arange(0, n):
        tag_list[i] = ' '.join(str(x) for x in tag_list[i].split(','))

    tag_series = pd.Series(index=np.arange(0, n), data=tag_list)

    return tag_list, tag_series


def build_bow(l):

    vectorizer = TfidfVectorizer(min_df=2)

    return vectorizer.fit_transform(l)
