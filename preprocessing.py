import pandas as pd
import numpy as np
import logging
from recpy.utils.data_utils import df_to_csr

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

# Reading the two files interactions.csv and targets_users.csv
logger.info('Reading the files ...')
interactions = pd.read_csv('data/interactions.csv', sep="\t")

# removing useless items
counts = interactions['item_id'].value_counts()
useful_items = counts[counts.values > 1].index.tolist()
useful_items = pd.DataFrame(useful_items, columns=['item_id'])
useful_interactions = interactions.merge(useful_items, on='item_id', how='right')

print("Shape of the interactions dataframe: {}".format(useful_interactions.shape))

# let keep users and items
users = useful_interactions["user_id"].unique()
items = useful_interactions["item_id"].unique()

num_users, num_items = len(users), len(items)
print("There are %d users and %d items" % (num_users, num_items))

# indexing of users and items
user_idx = pd.Series(index=users, data=np.arange(num_users))
item_idx = pd.Series(index=items, data=np.arange(num_items))

# building the final dataframe adding "user's index" and "item's index"
useful_interactions["user_idx"] = user_idx[useful_interactions["user_id"].values].values
useful_interactions["item_idx"] = item_idx[useful_interactions["item_id"].values].values
data_csr = df_to_csr(useful_interactions, num_items, num_users, user_key='item_idx', item_key='user_idx')

useful_items = []
for i in np.arange(num_items):
    if (data_csr[i].nnz > 1):
        useful_items.append(i)

useful_items = pd.DataFrame(useful_items, columns=['item_idx'])
useful_interactions = useful_interactions.merge(useful_items, on='item_idx', how='right')
useful_interactions = useful_interactions.drop(['user_idx', 'item_idx'], axis=1)

print("Shape of the interactions dataframe: {}".format(useful_interactions.shape))

useful_interactions.to_csv('data/useful_interactions.csv', index=False)


