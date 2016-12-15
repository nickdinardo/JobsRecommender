import argparse
import logging
from collections import OrderedDict
from datetime import datetime as dt
import pandas as pd
import numpy as np

from recpy.recommeders.item_knn import ItemKNNRecommender
from recpy.recommeders.user_knn import UserKNNRecommender
from recpy.utils.data_utils import read_dataset, df_to_csr


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('item_knn', ItemKNNRecommender),
    ('user_knn', UserKNNRecommender),
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--is_implicit', action='store_true', default=True)
parser.add_argument('--make_implicit', action='store_true', default=True)
parser.add_argument('--implicit_th', type=float, default=1.0)
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default='\t')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--rating_key', type=str, default='interaction_type')
parser.add_argument('--rnd_seed', type=int, default=1234)
parser.add_argument('--recommender', type=str, default='user_knn')
parser.add_argument('--item_file', type=str, default='idx_item.csv')
parser.add_argument('--user_file', type=str, default='idx_user.csv')
parser.add_argument('--params', type=str, default=None)
parser.add_argument('--rec_length', type=int, default=5)
args = parser.parse_args()

# get the recommender class
assert args.recommender in available_recommenders, 'Unknown recommender: {}'.format(args.recommender)
RecommenderClass = available_recommenders[args.recommender]

# parse recommender parameters
init_args = OrderedDict()
if args.params:
    for p_str in args.params.split(','):
        key, value = p_str.split('=')
        try:
            init_args[key] = eval(value)
        except:
            init_args[key] = value

# convert the column argument to list
if args.columns is not None:
    args.columns = args.columns.split(',')

# let's load indices of all possible users (previously computed)
item_path = 'output/models/' + args.item_file
user_path = 'output/models/' + args.user_file
item_idx = pd.read_csv(item_path, index_col=1, squeeze=True, sep=',')
user_idx = pd.read_csv(user_path, index_col=1, squeeze=True, sep=',')

# read the dataset
logger.info('Reading {}'.format(args.dataset))
dataset, idx_to_user, idx_to_item = read_dataset(
    args.dataset,
    header=args.header,
    sep=args.sep,
    columns=args.columns,
    make_implicit=args.make_implicit,
    implicit_th=args.implicit_th,
    item_key=args.item_key,
    user_key=args.user_key,
    rating_key=args.rating_key,
    item_to_idx=item_idx,
    user_to_idx=user_idx)

nusers, nitems = len(idx_to_user), len(idx_to_item)
logger.info('The dataset has {} users and {} items'.format(nusers, nitems))

# let's construct the training set
train = df_to_csr(dataset, is_implicit=args.is_implicit, nrows=nusers, ncols=nitems)
logger.info('The train set is a sparse matrix of shape: {}'.format(train.shape))

# train the recommender
recommender = RecommenderClass(**init_args)
logger.info('Recommender: {}'.format(recommender))
logger.info('Parameters: {}'.format(init_args if args.params else 'default'))
tic = dt.now()
logger.info('Training started')
recommender.fit(train)
logger.info('Training completed in {}'.format(dt.now() - tic))
