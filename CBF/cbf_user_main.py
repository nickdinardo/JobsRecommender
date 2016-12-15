import logging
import argparse
from collections import OrderedDict
from datetime import datetime as dt
import scipy.sparse as sps


from recpy.utils.cbf_utils import process_data
from recpy.utils.cbf_utils import read_dataset
from recpy.utils.preprocessing import DataTransformer
from recpy.recommeders.user_cbf import CBFUsersRecommender
from recpy.recommeders.item_cbf import CBFItemsRecommender


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(l-evelname)s: %(message)s")

available_cbf_recommenders = OrderedDict([
    ('cbf_users', CBFUsersRecommender),
    ('cbf_items', CBFItemsRecommender)
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--holdout_perc', type=float, default=0.8)
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default='\t')
parser.add_argument('--key', type=str, default='user_id')
parser.add_argument('--attr', type=str, default='jobroles')
parser.add_argument('--rnd_seed', type=int, default=1234)
parser.add_argument('--recommender', type=str, default='cbf_users')
parser.add_argument('--params', type=str, default=None)
parser.add_argument('--prediction_file', type=str, default=None)
parser.add_argument('--rec_length', type=int, default=5)
args = parser.parse_args()

# get the recommender class
assert args.recommender in available_cbf_recommenders, 'Unknown recommender: {}'.format(args.recommender)
RecommenderClass = available_cbf_recommenders[args.recommender]

# get the parameters
init_args = OrderedDict()
if args.params:
    for p_str in args.params.split(','):
        key, value = p_str.split('=')
        try:
            init_args[key] = eval(value)
        except:
            init_args[key] = value

# let's read the dataset
logger.info('Reading {}'.format(args.dataset))
dataset, idx_to_instances, instance_to_index = read_dataset(
    args.dataset,
    header=args.header,
    sep=args.sep,
    columns=args.columns,
    key=args.key,
    series=True)

#idx_to_instances.to_csv('data/output/idx_instance.csv', header=True)
#instance_to_index.to_csv('data/output/instance_idx.csv', header=True)

num_instances = len(idx_to_instances)
logger.info('The dataset has {} instances'.format(num_instances))

# transformer = DataTransformer(dataset)
# to_clean = ['career_level']
# not_to_clean = []
# clean_vectorize = transformer.transform(to_clean)
# only_vectorize = transformer.transform(not_to_clean, clean=False)

# merging the two sparse matrices
# first_part = sps.hstack([clean_vectorize, only_vectorize]).tocsc()
# first_part = clean_vectorize


# now we need to construct the second part on features that are list of objects
# we treat them as words, building a Bag of Words model
train_set = process_data(dataset, args.attr)
print(train_set.shape)

# dataset['edu_fieldofstudies'] = dataset['edu_fieldofstudies'].fillna(0)
# edu_bow = process_data(dataset, 'edu_fieldofstudies')

# merging the two bag of words
# second_part = sps.hstack([jobroles_bow, edu_bow])


# finally the entire training set
# train_set = sps.hstack([first_part, jobroles_bow]).tocsc()

# train the recommender
recommender = RecommenderClass(**init_args)
logger.info('Recommender: {}'.format(recommender))
logger.info('Parameters: {}'.format(init_args if args.params else 'default'))
tic = dt.now()
logger.info('Training started')
recommender.fit(train_set)
logger.info('Training completed in {}'.format(dt.now() - tic))

# idx_to_instances.to_csv('output/models/idx_user.csv')
# instance_to_index.to_csv('output/models/user_idx.csv')


