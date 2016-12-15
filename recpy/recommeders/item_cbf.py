from recpy.recommeders.base import Recommender
from recpy.recommeders.similarity import Cosine, AdjustedCosine, Pearson
from recpy.utils.data_utils import save_sparse, load_sparse

import numpy as np


class CBFItemsRecommender(Recommender):

    """ Content Based Filtering - User approach """

    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False):
        super(CBFItemsRecommender, self).__init__()
        self.k = k
        self.shrinkage = shrinkage
        self.item_weights = None
        self.normalize = normalize
        self.dataset = None
        self.similarity_name = similarity

        if similarity == 'cosine':
            self.similarity = Cosine(shrinkage=self.shrinkage)
        elif similarity == 'pearson':
            self.similarity = Pearson(shrinkage=self.shrinkage)
        elif similarity == 'adj-cosine':
            self.similarity = AdjustedCosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not implemented'.format(similarity))

    def __str__(self):
        return "CBFItemKNN(similarity={})".format(self.similarity_name)

    def fit(self, X):
        self.dataset = X
        self.item_weights = self.similarity.compute(X.transpose())

        # for each column, keep only the top-k scored items
        for i in np.arange(0, self.item_weights.shape[1]):
            col = self.item_weights.getcol(i)   # get the column i
            idx_sorted = np.argsort(col, axis=0)    # sort indexes by column
            not_top_k = idx_sorted[:-self.k]     # keep indexes of the less relevant
            self.item_weights[not_top_k, i] = 0.0   # set them to zero
            print(i)

        save_sparse('output/models/sparse1', self.user_weights)
