import numpy as np

from recpy.recommeders.base import Recommender
from recpy.recommeders.similarity import Cosine, Pearson, AdjustedCosine
from recpy.utils.data_utils import save_sparse, load_sparse


class ItemKNNRecommender(Recommender):
    """ ItemKNN recommender"""

    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False):
        super(ItemKNNRecommender, self).__init__()
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
        return "ItemKNN(similarity={})".format(self.similarity_name)

    def fit(self, X):
        self.dataset = X
        self.item_weights = self.similarity.compute(X)

        # for each column, keep only the top-k scored items
        for i in np.arange(0, self.item_weights.shape[1]):
            col = self.item_weights.getcol(i)  # get the column i
            idx_sorted = np.argsort(col, axis=0)  # sort indexes by column
            not_top_k = idx_sorted[:-self.k]  # keep indexes of the less relevant
            self.item_weights[not_top_k, i] = 0.0  # set them to zero
            print(i)

        save_sparse('output/models/collaborative_item', self.item_weights)

    def load_weights(self, file):
        self.item_weights = load_sparse(file)

    def make_prediction(self, targets, urm, rec_items, num=5, exclude_seen=False):
        #scores = -(self.user_weights[targets, :].dot(urm))
        scores = -(urm[targets, :].dot(self.item_weights[:, rec_items]))
        # recommendations = scores.argsort(axis=1)[:, scores.shape[1]-num:]
        recommendations = np.zeros((len(targets), num))

        for i in np.arange(len(targets)):
            recommendations[i, :] = np.array(scores[i, :].todense()).argsort(axis=1)[0][:num]
            # np.array(recommendations.getrow(i).todense())[0][::-1]
            print(i)

        return recommendations









        # self.dataset = X
        # self.item_weights = self.similarity.compute(X)
        # for each column, keep only the top-k scored items
        # idx_sorted = np.argsort(self.item_weights, axis=0)  # sort by column
        # index of the items that DON'T BELONG 
        # to the top-k similar items
        # not_top_k = idx_sorted[:-self.k, :]
        # zero-out the not top-k items for each column
        # self.item_weights[not_top_k, np.arange(self.item_weights.shape[1])] = 0.0

    def recommend(self, user_id, n=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.item_weights).ravel()

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = user_profile.copy()
            rated.data = np.ones_like(user_profile.data)
            den = rated.dot(self.item_weights).ravel()
            den[np.abs(den) < 1e-6] = 1.0 # to avoid NaNs
            scores /= den
        # rank items
        ranking = scores.argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]