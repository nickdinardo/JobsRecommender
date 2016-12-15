from recpy.recommeders.user_knn import UserKNNRecommender
from recpy.recommeders.item_knn import ItemKNNRecommender
from recpy.utils.cbf_utils import read_dataset, recommendable_items
from recpy.utils.data_utils import df_to_csr
from recpy.utils.data_utils import read_dataset as read_interactions

import logging
from collections import OrderedDict
import argparse
import pandas as pd

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
parser.add_argument('item_profile')
parser.add_argument('target_users')
parser.add_argument('interactions')
parser.add_argument('prediction_file')
parser.add_argument('model_file')
parser.add_argument('--recommender', type=str, default='user_knn')
parser.add_argument('--idx_item', type=str, default='idx_item.csv')
parser.add_argument('--idx_user', type=str, default='idx_user.csv')
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default='\t')
parser.add_argument('--attr', type=str, default='active_during_test')
parser.add_argument('--item_key', type=str, default='id')
args = parser.parse_args()

# get the recommender class
assert args.recommender in available_recommenders, 'Unknown recommender: {}'.format(args.recommender)
RecommenderClass = available_recommenders[args.recommender]

# let's read the dataset of items
logger.info('Reading {}'.format(args.item_profile))
items = read_dataset(args.item_profile, header=args.header, columns=args.columns, sep=args.sep)
items = recommendable_items(items, args.attr, args.item_key)  # series of recommendable items

interactions = pd.read_csv('data/interactions.csv', sep='\t')  # TODO it sucks
items = interactions.merge(pd.DataFrame(items), left_on='item_id', right_on='id', how='inner')
item_path = 'output/models/' + args.idx_item
idx_item = pd.read_csv(item_path, index_col=0, squeeze=True, sep=',')
item_idx = pd.Series(index=idx_item.data, data=idx_item.index)
# items = items['id'].unique()
# idx_item = pd.Series(index=range(len(items)), data=items)

items['item_idx'] = item_idx[items['item_id'].values].values
recomendable_items = items['item_idx'].unique()

# now we need users indexes
user_path = 'output/models/' + args.idx_user
idx_user = pd.read_csv(user_path, index_col=0, squeeze=True, sep=',')
user_idx = pd.Series(index=idx_user.data, data=idx_user.index)

# now the dataset of target users
logger.info('Reading {}'.format(args.target_users))
targets = read_dataset(args.target_users, sep=',')
# we need to merge with the new indices
targets['user_idx'] = user_idx[targets['user_id'].values].values

# finally interactions
logger.info('Reading {}'.format(args.interactions))
interactions, n, n1 = read_interactions(args.interactions, sep=args.sep, user_to_idx=user_idx, item_to_idx=item_idx)
interactions = interactions[interactions['item_idx'] >= 0.0]

urm = df_to_csr(interactions, user_idx.shape[0], item_idx.shape[0], is_implicit=True)

recommender = RecommenderClass()
model_path = 'output/models/' + args.model_file  # from where to load the already computed model (similarity matrix)
recommender.load_weights(model_path)
recs = recommender.make_prediction(targets['user_idx'].values, urm, recomendable_items, num=5)

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

# close the prediction file
if args.prediction_file:
    pfile.close()
    logger.info('Recommendations written to {}'.format(args.prediction_file))
