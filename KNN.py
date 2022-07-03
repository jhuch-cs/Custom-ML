from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import math
import pandas as pd

class KNNClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, has_categorical=False, weight_type='inverse_distance'):
        self.has_categorical = has_categorical
        self.weight_type = weight_type

    def fit(self, instance_data, target_data):
        self.training_instance_data = instance_data
        self.training_target_data = target_data
        return self

    def _get_target(self, target_labels, distances):
        if self.weight_type == 'no_weight': 
            sorted_labels = np.sort(target_labels)
            majority_label = sorted_labels[np.argmax(np.unique(sorted_labels, return_counts=True)[1])]
            return majority_label
        elif self.weight_type == 'inverse_distance':
            # fancy weighting
            inv_distances = [1 / (dist ** 2) for dist in distances]
            df = pd.DataFrame({'label': target_labels, 'inv_dist': inv_distances }  )
            np_arr = df.groupby('label').sum().reset_index().to_numpy()
            most_weighted_value = np_arr[np.argmax(np_arr[:,1])][0]
            return most_weighted_value   
        else: 
            raise ValueError("Invalid weight_type provided to KNNCLassifier")

    def _difference_with_categorical_and_unknown(self, x, y):
        if x == y: # if the attributes are equal, regardless of type
            return 0
        elif x == b'?' or y == b'?': # if either attribute is unknown
            return 1
        else:
            try:
                val = x - y
                return val if not np.isnan(val) else 1
            except:
                return 1

    def _distance_with_categorical_and_unknown(self, training_row, test_row):
        differences = [self._difference_with_categorical_and_unknown(x, y)**2 for x, y in zip(training_row, test_row)]
        return math.sqrt(sum(differences))

    def _predict_one_with_categorical(self, test_row, k = 3):
        distances = np.array([self._distance_with_categorical_and_unknown(training_row, test_row) 
                                for training_row in self.training_instance_data])
        kth_distances_indices = np.argpartition(distances, k)[:k]

        kth_distances = distances[kth_distances_indices]
        kth_targets = self.training_target_data[kth_distances_indices]

        return self._get_target(kth_targets, kth_distances)

    def _predict_one(self, test_row, k = 3):
        differences = self.training_instance_data - test_row
        differences[differences == 0.0] = 0.0000001 # prevent numpy error with `astype` in the next line
        distances = np.linalg.norm(differences.astype(float), axis = 1)
        kth_distances_indices = np.argpartition(distances, k)[:k]

        kth_distances = distances[kth_distances_indices]
        kth_targets = self.training_target_data[kth_distances_indices]

        return self._get_target(kth_targets, kth_distances)
    
    def predict(self, test_data, k = 3):
        if not self.has_categorical: # save computation
            predictions = [self._predict_one(test_row, k) for test_row in test_data]
        else:
            predictions = [self._predict_one_with_categorical(test_row, k) for test_row in test_data]
        return predictions

    def score(self, X, y, k = 3):
        predictions = self.predict(X, k)
        num_correct = len([1 for prediction, target in zip(predictions, y) if prediction == target])
        return round(num_correct / len(y), 4)    

class KNNRegressor(KNNClassifier):
    def _get_target(self, target_labels, distances):
        if self.weight_type == 'no_weight':
            return np.mean(target_labels)
        elif self.weight_type == 'inverse_distance':
            inv_distances = [1 / (dist ** 2) for dist in distances]
            weighted_pairs = [w * val for w, val in zip(inv_distances, target_labels)]
            return sum(weighted_pairs) / sum(inv_distances)
        else:
            raise ValueError("Invalid weight_type provided to KNNRegressor")

    def score(self, X, y, k = 3):
        predictions = self.predict(X, k)
        error = mse(y, predictions)
        return error    