import numpy as np
# from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sps

from recpy.recommeders.base import Recommender
from recpy.recommeders.similarity import Cosine, Pearson, AdjustedCosine
from recpy.utils.data_utils import save_sparse, load_sparse

class UserKNNRecommender(Recommender):
    """User K-Nearest Neighbors Recommender"""

    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False):
        super(UserKNNRecommender, self).__init__()
        self.k = k
        self.shrinkage = shrinkage
        self.user_weights = None
        # self.idx_sorted = None
        self.normalize = normalize
        self.dataset = None
        # self.recommendations = None
        # self.nbrs = None
        # self.new_users = []
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
        return "UserKNN(similarity={})".format(self.similarity_name)

    def fit(self, X):
        self.dataset = X
        self.user_weights = self.similarity.compute(X.transpose())

        # for each column, keep only the top-k scored items
        for i in np.arange(0, self.user_weights.shape[1]):
            col = self.user_weights.getcol(i)  # get the column i
            idx_sorted = np.argsort(col, axis=0)  # sort indexes by column
            not_top_k = idx_sorted[:-self.k]  # keep indexes of the less relevant
            self.user_weights[not_top_k, i] = 0.0  # set them to zero
            print(i)

        save_sparse('output/models/sparse1', self.user_weights)

    def load_weights(self, file):
        self.user_weights = load_sparse(file)

    def make_prediction(self, targets, urm, num=5, exclude_seen=False):
        scores = -(self.user_weights.transpose()[targets, :].dot(urm))
        # recommendations = scores.argsort(axis=1)[:, scores.shape[1]-num:]
        recommendations = np.zeros((len(targets), num))

        for i in np.arange(len(targets)):
            recommendations[i, :] = np.array(scores[i, :].todense()).argsort(axis=1)[0][:num]
            # np.array(recommendations.getrow(i).todense())[0][::-1]
            print(i)

        return recommendations

        # this was fit function
        # self.dataset = X
        # compute the similarity matrix between users
        # self.user_weights = self.similarity.compute(X.T)
        # let try with a different approach, using sklearn library
        #self.nbrs = NearestNeighbors(n_neighbors=self.k, metric='cosine', algorithm='brute').fit(self.dataset)
        #self.idx_sorted = np.argsort(self.user_weights, axis=0)  # sort by column
        #  index of the items that DON'T BELONG
        # to the top-k similar items
        #not_top_k = self.idx_sorted[:-self.k, :]
        # zero-out the not top-k items for each column
        #self.user_weights[not_top_k, np.arange(self.user_weights.shape[1])] = 0.0

    def compute_recommendations(self, items, targets, nusers):

        target_profiles = self.dataset[targets, :]
        dist, idx = self.nbrs.kneighbors(target_profiles)

        shape = (nusers, nusers)
        rows = idx.ravel()
        cols = np.repeat(targets, self.k)
        ratings = dist.ravel()

        self.user_weights = sps.csr_matrix((ratings, (rows, cols)), shape=shape)

        rec_items = self.dataset.T[items, :]
        target_users_sim = self.user_weights[:, targets]
        for i in np.arange(len(targets)):
            if (sum(target_users_sim[:, i].toarray().ravel()) == self.k):
                self.new_users.append(i)


        self.recommendations = rec_items.dot(target_users_sim)

    def recommend(self, user_id, n=None, exclude_seen=True):
        # computing scores for the specified user
        if user_id in self.new_users:
            return ['p', 'p', 'p', 'p', 'p']
        else:

            scores = self.recommendations[:, user_id].toarray().ravel()
            # rank items
            ranking = scores.argsort()[::-1]
            if exclude_seen:
                ranking = self._filter_seen(user_id, ranking)
            return ranking[:n]

