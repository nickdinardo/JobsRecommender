from recpy.recommeders.user_cbf import CBFUsersRecommender
from recpy.utils.cbf_utils import read_dataset, recommendable_items, build_series
from recpy.utils.data_utils import df_to_csr
from recpy.utils.data_utils import read_dataset as read_interactions

import logging
import argparse
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('item_profile')
parser.add_argument('target_users')
parser.add_argument('interactions')
parser.add_argument('prediction_file')
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default='\t')
parser.add_argument('--attr', type=str, default='active_during_test')
parser.add_argument('--item_key', type=str, default='id')
args = parser.parse_args()

# let's read the dataset of items
logger.info('Reading {}'.format(args.item_profile))
items = read_dataset(args.item_profile, header=args.header, columns=args.columns, sep=args.sep)
items = recommendable_items(items, args.attr, args.item_key)  # series of recommendable items

interactions = pd.read_csv('data/interactions.csv', sep='\t')  # TODO it sucks
items = interactions.merge(pd.DataFrame(items), left_on='item_id', right_on='id', how='inner')
items = items['id'].unique()
idx_item = pd.Series(index=range(len(items)), data=items)
#item_idx = build_series(items)
item_idx = pd.Series(index=idx_item.data, data=idx_item.index)

# now we need users indexes
idx_user = pd.read_csv('output/models/user_idx.csv', index_col=1, squeeze=True, sep=',')
print(idx_user.head())
user_idx = pd.Series(pd.read_csv('output/models/idx_user.csv', index_col=1, squeeze=True, sep=','))
print(user_idx.head())

profiles = pd.read_csv("data/user_profile.csv", sep='\t')

# now the dataset of target users
logger.info('Reading {}'.format(args.target_users))
targets = read_dataset(args.target_users, sep=',')

# print(set(targets_all['user_id'].values) <= set(idx_user.data))
#targets = targets_all.merge(profiles, how='inner', on='user_id')
targets['user_idx'] = user_idx[targets['user_id'].values].values

# finally interactions
logger.info('Reading {}'.format(args.interactions))
interactions, n1, n2 = read_interactions(args.interactions, sep=args.sep, user_to_idx=user_idx, item_to_idx=item_idx)
interactions = interactions[interactions['item_idx'] >= 0.0]

urm = df_to_csr(interactions, user_idx.shape[0], len(interactions['item_idx'].unique()), is_implicit=True)

recommender = CBFUsersRecommender()
recommender.load_user_weights('output/models/sparse_cbf.npz')
recs = recommender.make_prediction(targets['user_idx'].values, urm)
print(recs.shape)

# open the prediction file and write the header
if args.prediction_file:
    pfile = open(args.prediction_file, 'w')
    header = 'user_id,recommended_items' + '\n'
    pfile.write(header)

new_user_idx = targets['user_idx'].values
for target in range(recs.shape[0]):
    user_id = idx_user[new_user_idx[target]]
    rec_list = idx_item[recs[target]].values
    s = str(user_id) + ','
    s += ' '.join([str(x) for x in rec_list]) + '\n'
    pfile.write(s)
    print('user ' + str(user_id) + ' done!')


# close the prediction file
if args.prediction_file:
    pfile.close()
    logger.info('Recommendations written to {}'.format(args.prediction_file))
