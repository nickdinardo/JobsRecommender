import pandas as pd
from recpy.utils.cbf_utils import process_data

path = 'data/item_profile.csv'
items = pd.read_csv(path, sep='\t')
print(items.info())

items['title'] = items['title'].fillna(0)
items['tags'] = items['tags'].fillna(0)
print(items.info())

item_sim = process_data(items, 'title')
print(item_sim.shape)

prova = item_sim.dot(item_sim.transpose())


