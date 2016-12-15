import pandas as pd


output = pd.read_csv("output/prova.csv", sep=',')

rec_items = []
jobs = ['rec_item1', 'rec_item2', 'rec_item3', 'rec_item4', 'rec_item5', ]
for i in output.index:
    rec_items.append(" ".join(str(x) for x in output.ix[i][jobs].values))


rec = pd.DataFrame(output['user_id'], columns=['user_id'])
rec['recommended_items'] = rec_items

rec.to_csv('output/output.csv', index=False)
