from recpy.recommeders.base import Recommender
from recpy.recommeders.similarity import Cosine, AdjustedCosine, Pearson
from recpy.utils.data_utils import save_sparse, load_sparse

import numpy as np


class CBFUsersRecommender(Recommender):

    """ Content Based Filtering - User approach """

    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False):
        super(CBFUsersRecommender, self).__init__()
        self.k = k
        self.shrinkage = shrinkage
        self.user_weights = None
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
        return "CBFUserKNN(similarity={})".format(self.similarity_name)

    def fit(self, X):
        self.dataset = X
        self.user_weights = self.similarity.compute(X.transpose())

        # for each column, keep only the top-k scored items
        for i in np.arange(0, self.user_weights.shape[1]):
            col = self.user_weights.getcol(i)   # get the column i
            idx_sorted = np.argsort(col, axis=0)    # sort indexes by column
            not_top_k = idx_sorted[:-self.k]     # keep indexes of the less relevant
            self.user_weights[not_top_k, i] = 0.0   # set them to zero
            print(i)

        save_sparse('output/models/user_cbf', self.user_weights)

    def make_prediction(self, targets, urm, num=5, exclude_seen=False):

        scores = -(self.user_weights.transpose()[targets, :].dot(urm))
        # recommendations = scores.argsort(axis=1)[:, scores.shape[1]-num:]
        recommendations = np.zeros((len(targets), num))

        for i in np.arange(len(targets)):
            recommendations[i, :] = np.array(scores[i, :].todense()).argsort(axis=1)[0][:num]
            # np.array(recommendations.getrow(i).todense())[0][::-1]
            print(i)

        return recommendations

    def load_user_weights(self, file):
        self.user_weights = load_sparse(file)






        #idx_sorted = np.argsort(self.user_weights, axis=0)  # sort by column
        # index of the items that DON'T BELONG
        # to the top-k similar items
        #not_top_k = idx_sorted[:-self.k, :]
        # zero-out the not top-k items for each column
        #self.user_weights[not_top_k, np.arange(self.user_weights.shape[1])] = 0.0
        #print(self.user_weights.shape)

