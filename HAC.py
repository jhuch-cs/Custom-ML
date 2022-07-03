from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np

class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self, k=3, link_type='single'):
        self.link_type = link_type
        self.k = k
        
    def fit(self, X, y=None):
        self.X = X
        point_proximity_matrix = cdist(X, X, 'euclidean')
        # Initially, each point is its own cluster
        clusters = [Cluster([point], self.link_type) for point in range(len(X))]

        while len(clusters) > self.k:
            # Calculate distances between clusters
            #   could optimize by not re-computing entire table
            #   and by only computing above the diagonal
            cluster_distances = [(c1.distance_to_cluster(c2, point_proximity_matrix), i, j) for i, c1 in enumerate(clusters) for j, c2 in enumerate(clusters) if i != j]
            distance_between, c1, c2 = min(cluster_distances)
            
            # Combine the two closest clusters
            combined_cluster = Cluster([clusters[c1], clusters[c2]], link_type=self.link_type)
            del clusters[c2]
            del clusters[c1]
            clusters.append(combined_cluster)
        
        self.clusters = clusters
        return self

    def label_data_by_cluster(self):
        y = np.array([0] * len(self.X))
        for cluster_num, cluster in enumerate(self.clusters):
            y[cluster.collect_points()] = cluster_num
        return y

    def compute_silhouette(self):
        y = self.label_data_by_cluster()
        return silhouette_score(self.X, y)
    
    def print_clusters(self):
        print("Num clusters: {:d}\n".format(self.k))
        print("Silhouette score: {:.4f}\n\n".format(self.compute_silhouette()))
        for cluster in sorted(self.clusters, key=lambda cluster: len(cluster), reverse=True):
            print(f"Centroid: {cluster.get_centroid_string(self.X)}")
            print(f"Length: {len(cluster)}\n")

class Cluster():
    def __init__(self, initial_data,link_type='single'):
        self.clusters = [datum for datum in initial_data]
        self.link_type = link_type

    # Return all points (recursively)
    def collect_points(self): 
        if hasattr(self, 'collected_points'):
            return self.collected_points

        points = []
        for cluster in self.clusters:
            if isinstance(cluster, Cluster):
                points.extend(cluster.collect_points())
            else:
                points.append(cluster)
        
        points = np.array(points)

        self.collected_points = points
        return points

    def distance_to_cluster(self, other_cluster, proximity_matrix):
        if self == other_cluster:
            return 0
        link_func = max if self.link_type == 'complete' else min
        return link_func([proximity_matrix[point_in_self][point_in_other] for point_in_self in self.collect_points() for point_in_other in other_cluster.collect_points()])

    def get_centroid_string(self, X):
        points = self.collect_points()
        points = [X[point] for point in points]
        centroid = np.mean(points, axis=0)
        return np.array2string(centroid, precision=4, separator=",")

    def __len__(self):
        return len(self.collect_points())