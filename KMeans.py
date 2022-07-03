from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np

class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,debug=False):
        self.k = k
        self.debug = debug

    def fit(self, X, y=None):
        self.X = X
        clusters = []
        centroids = self.get_initial_centroids(X)
        centroid_with_min_dist = None

        while True:
            proximity_matrix = cdist(centroids, X, 'euclidean') # get distance from each centroid to each point
            centroid_with_min_dist = proximity_matrix.argsort(axis=0)[0] # find minimum distance to centroid for each point
            clusters = [X[centroid_with_min_dist == centroid_index] for centroid_index in range(self.k)] # group by centroid
            # using argsort above implies preferring the later index in ties
        
            new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

            if not self.debug and np.any(empty_clusters := [cluster_index for cluster_index in range(len(clusters)) if len(clusters[cluster_index]) == 0]):
                masked_centroids = np.ma.array(centroids, mask=False)
                masked_centroids[empty_clusters] = True # ignore empty clusters in the comparison

                masked_new_centroids = np.ma.array(new_centroids, mask=False)
                masked_new_centroids[empty_clusters] = True # ignore empty clusters in the comparison

                if np.array_equal(masked_centroids, masked_new_centroids):
                    break

                for cluster_index in empty_clusters:
                    new_centroids[cluster_index] = self.get_one_centroid() # re-roll centroids with empty clusters
            else:
                if np.array_equal(centroids, new_centroids):
                    break

            centroids = new_centroids

        self.centroids = centroids
        self.clusters = clusters
        self.y = centroid_with_min_dist
        return self

    def get_initial_centroids(self, X):
        if self.debug:
            return X[:self.k]
        else:
            bounding_box = self._calc_bounding_box(X)
            return np.random.uniform(low=bounding_box[:, 0], high=bounding_box[:, 1], size=(self.k, bounding_box.shape[0]))

    def get_one_centroid(self):
        return np.random.uniform(low=self.bounding_box[:, 0], high=self.bounding_box[:, 1], size=self.bounding_box.shape[0])

    def _calc_bounding_box(self, X):
        if not hasattr(self, 'bounding_box'):
            self.bounding_box = np.array([(X[:,dimension].min(), X[:,dimension].max()) for dimension in range(X.shape[1])])
        return self.bounding_box

    def compute_silhouette(self):
        return silhouette_score(self.X, self.y)

    def label_data_by_cluster(self):
        return self.y

    def print_clusters(self):
        print("Num clusters: {:d}\n".format(self.k))
        print("Silhouette score: {:.4f}\n\n".format(self.compute_silhouette()))
        for cluster, centroid in zip(self.clusters, self.centroids):
            centroid_string = np.array2string(centroid, precision=4, separator=",")
            print(f"Centroid: {centroid_string}")
            print(f"Length: {len(cluster)}\n")