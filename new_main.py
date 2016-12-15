import argparse
import logging
from collections import OrderedDict
from datetime import datetime as dt
import pandas as pd
import numpy as np

from recpy.item_knn import ItemKNNRecommender
from recpy.user_knn import UserKNNRecommender
# from recpy.recommenders.non_personalized import TopPop, GlobalEffects
from recpy.utils.data_utils import read_without_idx, df_to_csr
#from recpy.metrics import roc_auc, precision, recall, map, ndcg, rr

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_recommenders = OrderedDict([
    #('top_pop', TopPop),
    #('global_effects', GlobalEffects),
    ('item_knn', ItemKNNRecommender),
    ('user_knn', UserKNNRecommender),
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset_train')
parser.add_argument('dataset_test')
parser.add_argument('active_items')
parser.add_argument('--is_implicit', action='store_true', default=False)
parser.add_argument('--make_implicit', action='store_true', default=False)
parser.add_argument('--implicit_th', type=float, default=4.0)
parser.add_argument('--holdout_perc', type=float, default=0.8)
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default=',')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--rnd_seed', type=int, default=1234)
parser.add_argument('--recommender', type=str, default='top_pop')
parser.add_argument('--params', type=str, default=None)
parser.add_argument('--prediction_file', type=str, default=None)
parser.add_argument('--rec_length', type=int, default=10)
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

# read the dataset
logger.info('Reading {}'.format(args.dataset_train))
dataset = read_without_idx(
    args.dataset_train,
    sep=',',
    make_implicit=args.make_implicit,
    implicit_th=args.implicit_th,
    rating_key=args.rating_key)


# Let retrieve the target users
logger.info('Reading {}'.format(args.dataset_test))
targets = pd.read_csv(args.dataset_test)

# computing number of users and items
all_users = dataset.merge(targets, on='user_id', how='outer')['user_id'].unique()
all_items = dataset['item_id'].unique()
n_users, n_items = len(all_users), len(all_items)

# computing indexes for items and users
item_to_idx = pd.Series(data=np.arange(n_items), index=all_items)
user_to_idx = pd.Series(data=np.arange(n_users), index=all_users)
idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
idx_to_user = pd.Series(index=user_to_idx.data, data=user_to_idx.index)

dataset['user_idx'] = user_to_idx[dataset['user_id'].values].values   # new user indexing
dataset['item_idx'] = item_to_idx[dataset['item_id'].values].values   # new item indexing
targets['user_idx'] = user_to_idx[targets['user_id'].values].values   # new target indexing
targets['new_idx'] = np.arange(targets.shape[0])

# transforming into csr
train = df_to_csr(dataset, is_implicit=args.is_implicit, nrows=n_users, ncols=n_items)


# train the recommender
recommender = RecommenderClass(**init_args)
logger.info('Recommender: {}'.format(recommender))
logger.info('Parameters: {}'.format(init_args if args.params else 'default'))
tic = dt.now()
logger.info('Training started')
recommender.fit(train)
logger.info('Training completed in {}'.format(dt.now() - tic))

# open the prediction file
if args.prediction_file:
    pfile = open(args.prediction_file, 'w')
    n = args.rec_length if args.rec_length is not None else n_items
    header = 'user_id,'
    header += ','.join(['rec_item{}'.format(i+1) for i in range(args.rec_length)]) + '\n'
    pfile.write(header)

active_items = pd.read_csv(args.active_items, sep='\t')
active_items = active_items.rename(columns={'id': 'item_id'})
active_items = active_items.merge(dataset, on='item_id', how='inner')
active_items = active_items[['item_id', 'item_idx']]
active_items_idx = pd.DataFrame(data=active_items['item_idx'].unique(),
                                index=active_items['item_id'].unique(),
                                columns=['item_idx'])
active_items_idx['new_idx'] = np.arange(active_items_idx.shape[0])

item_index_series = pd.Series(index=np.arange(active_items_idx.shape[0]), data=active_items['item_id'].unique())

rec_items = active_items_idx['item_idx'].tolist()
target_users = targets['user_idx'].tolist()

logger.info("Computing recommendations ...")
recommender.compute_recommendations(rec_items, target_users, n_users)
logger.info("Done!")

new_idx_targets = targets['new_idx'].tolist()
logger.info("Writing predictions")
for test_user in new_idx_targets:

    #user_profile = train[test_user]
    # this will rank **all** items
    recommended_items = recommender.recommend(user_id=test_user, n=5, exclude_seen=False)

    if args.prediction_file:
        # write the recommendation list to file, one user per line
        user_id = targets[targets['new_idx'] == test_user]['user_id'].values[0]
        s = str(user_id) + ','
        if recommended_items == ['p', 'p', 'p', 'p', 'p']:
            s += '2778525 1244196 1386412 657183 2791339' + '\n'
        else:
            rec_list = item_index_series[recommended_items].values[:args.rec_length]
            s += ' '.join(str(x) for x in rec_list) + '\n'
        pfile.write(s)
logger.info('Done!')

# close the prediction file
if args.prediction_file:
    pfile.close()
    logger.info('Recommendations written to {}'.format(args.prediction_file))
